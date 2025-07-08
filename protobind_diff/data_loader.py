# Data loader for the protobind-diff.
# This version only supports ProtobindMaskedDiffusion with SMILES and ESM-2 protein encodings.
import os.path
import json
import logging
from pathlib import Path
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union
from zipfile import ZipFile

import lightning.pytorch as pl
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from .ligands.smiles_tokenizer import ChemformerTokenizer
from .ligands.rdkit_utils import randomize_smiles_rotated, cluster_fpsim2

logger = logging.getLogger("lightning")


class SplittingMethod(Enum):
    # enum that describes various train/val/test splitting methods.
    RANDOM = 1


def split_at_random(df: pd.DataFrame, valid_fraction=0.1, test_fraction=0.1, seed=777):
    """Randomly splits a DataFrame into training, validation, and test sets.

        Args:
            df (pd.DataFrame): The DataFrame to split.
            valid_fraction (float): The fraction of the data to allocate to the validation set.
            test_fraction (float): The fraction of the data to allocate to the test set.
            seed (int): The random seed for shuffling to ensure reproducibility.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the
            training, validation, and test DataFrames.
        """
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    valid_size = int(len(df) * valid_fraction)
    test_size = int(len(df) * test_fraction)
    train_size = len(df) - valid_size - test_size
    train_df = df[:train_size]
    valid_df = df[train_size:train_size + valid_size]
    test_df = df[train_size + valid_size:]
    return train_df, valid_df, test_df


class RandomizedSmilesDataset(object):
    """Creates a dataset of tokenized SMILES strings, with an option for on-the-fly randomization.

     This dataset maps integer indices to SMILES strings and provides tokenized
     representations. It can randomize SMILES strings during data retrieval to
     augment the training data.

     Attributes:
         smiles (pd.Series): A series of SMILES strings indexed by integers.
         tokenizer (ChemformerTokenizer): The tokenizer for converting SMILES to tokens.
         randomize (bool): If True, applies SMILES randomization at retrieval time.
     """
    def __init__(self, smiles: dict, tokenizer: ChemformerTokenizer,
                 randomize: bool = True):
        self.smiles = pd.Series(data=smiles.keys(), index=smiles.values()).sort_index()
        assert len(self.smiles) == self.smiles.index[-1] + 1, (f"{len(self.smiles)}"
                                                               f" {self.smiles.index[:5]} {self.smiles.index[-5:]}")
        self.tokenizer = tokenizer
        self.randomize = randomize
        logger.info(f"Molecular dataset initialized: RandomizedSmilesDataset {type(self.tokenizer)}"
                    f" random: {self.randomize}")

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        smi = self.smiles[item]
        if self.randomize:
            smi = randomize_smiles_rotated(smi)
        mol = self.tokenizer.encode(smi)[0]
        return mol

    @classmethod
    def from_json(cls, path, **kwargs):
        with open(path) as f:
            categorical_mappings = json.load(f)
            smiles = categorical_mappings['smiles']
            loaded = cls(smiles, **kwargs)
        return loaded


class RandomizedBatchSampler(torch.utils.data.Sampler):
    """A batch sampler that minimizes padding while maximizing batch randomness.

        To achieve this, the sampler employs a two-level shuffling strategy:
        1.  The data is first sorted by sequence length and grouped into buckets.
        2.  Within each bucket, the sample indices are shuffled.
        3.  Batches are created by slicing across the globally sorted list of indices,
            which keeps sequence lengths within a batch similar.
        4.  The order of these batches is then shuffled to ensure randomness across epochs.

        This approach balances the trade-off between minimizing padding (by batching
        similar-length sequences) and maintaining randomness required for effective training.
        """

    def __init__(self, sequence_length: np.ndarray, shuffle: bool, batch_volume: int,
                 generator: torch.Generator = None, num_ranges: int = 150, batch_size: int = 128):
        """Initializes the RandomizedBatchSampler.

           Args:
               sequence_length (np.ndarray): An array of sequence lengths for each item in the dataset.
               shuffle (bool): If True, shuffle batches and indices within length buckets.
               batch_volume (int): The maximum total number of elements (seq_len^2) per batch.
               generator (torch.Generator, optional): PyTorch random number generator. Defaults to None.
               num_ranges (int): The number of buckets to partition the sequence lengths into.
               batch_size (int): The maximum number of samples per batch.
        """
        self.shuffle = shuffle
        # For val/test (i.e. when we don't shuffle) we can fit more batches in memory as we don't need grads.
        batch_volume_factor = 1 if shuffle else 2
        self.batch_volume = batch_volume * batch_volume_factor
        assert max(sequence_length) ** 2 < self.batch_volume, \
            f"Cannot fit sequence {max(sequence_length)=} to {batch_volume=}"

        if generator is None:
            self.generator = self._init_generator()
        else:
            self.generator = generator
        self.num_ranges = num_ranges
        self.sequence_length = sequence_length
        self.sequence_length_2 = self.sequence_length ** 2
        self.batch_size = batch_size

        bins = np.linspace(np.min(sequence_length), np.max(sequence_length) + 1, num_ranges)
        digit_bins = np.digitize(sequence_length, bins=bins, right=True)
        self.sequence_length_buckets = [torch.tensor(np.where(digit_bins == i)[0],
                                                     dtype=torch.int32) for i in range(num_ranges)]
        self._prepared_batches = None

    def _get_sliced_batches(self):
        if self.shuffle:
            # reshuffle the sequence length buckets.
            for i in range(len(self.sequence_length_buckets)):
                self.sequence_length_buckets[i] = self.sequence_length_buckets[i][torch.randperm(
                    len(self.sequence_length_buckets[i]), generator=self.generator)]

        current_batch = []
        current_batch_volume = 0
        current_batch_size = 0
        for i in range(self.num_ranges):
            for idx in self.sequence_length_buckets[i]:
                if (current_batch_volume + self.sequence_length_2[idx] >= self.batch_volume
                        or current_batch_size >= self.batch_size):
                    yield current_batch
                    current_batch = []
                    current_batch_volume = 0
                    current_batch_size = 0
                current_batch.append(idx.item())
                current_batch_volume += self.sequence_length_2[idx]
                current_batch_size += 1
        if len(current_batch) > 0:
            yield current_batch

    @staticmethod
    def _init_generator():
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    @property
    def _length(self):
        if self._prepared_batches is None:
            self._prepared_batches = list(self._get_sliced_batches())
        return len(self._prepared_batches)

    def __len__(self):
        return self._length

    def __iter__(self):
        if self.shuffle:
            # Then get the batches and serve them in random order
            if self._prepared_batches is None:
                self._prepared_batches = list(self._get_sliced_batches())
            for batch_idx in torch.randperm(self._length, generator=self.generator):
                yield self._prepared_batches[batch_idx]
            self._prepared_batches = None  # Destroy _prepared_batches to recreate it again in __len__
        else:
            for batch in self._get_sliced_batches():
                yield batch


class ProtobindDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Protobind-diffusion datasets.

        This module handles the loading, processing, and batching of protein-ligand
        data. It is designed to work with ESM-2 protein embeddings and tokenized
        SMILES representations for ligands. The module manages data splitting,
        feature loading, and provides DataLoaders with an efficient batching
        strategy to minimize padding.

        Key Features:
        -   Loads pre-computed ESM-2 protein embeddings.
        -   Utilizes tokenized SMILES for ligands via `ChemformerTokenizer`.
        -   Implements a `RandomizedBatchSampler` to create efficient, low-padding batches.
        -   Handles dataset splitting into training, validation, and test sets.
    """
    MASK_VALUE = 0

    def __init__(self, *,
                 data_dir: Path,
                 exp_dir: Path,
                 splitting_method: SplittingMethod,
                 batch_volume: int,
                 num_workers: int,
                 sequence_type: str = 'esm_zip',
                 esm_model_name: str = "esm2_t33_650M_UR50D",
                 max_size_batch: int = 16,
                 dataset_params: Optional[dict] = None,
                 float_type: str = 'float32'):
        super().__init__()
        """Initializes the ProtobindDataModule.

             Args:
                 data_dir (Path): The directory containing the raw dataset files (e.g., data.csv, embeddings).
                 exp_dir (Path): The directory to save experiment artifacts, including data splits.
                 splitting_method (SplittingMethod): The method for splitting data (e.g., RANDOM).
                 batch_volume (int): The target batch volume for the RandomizedBatchSampler.
                 num_workers (int): The number of workers for the DataLoader.
                 sequence_type (str): The type of protein sequence data. Must be 'esm_zip'.
                 esm_model_name (str): The specific ESM model name for embeddings.
                 max_size_batch (int): The maximum number of samples in a batch.
                 dataset_params (Optional[dict]): Parameters for the underlying molecular dataset.
                 float_type (str): The floating-point precision to use.
        """
        self.csv_path = data_dir / "data.csv"
        self.categorical_mappings_path = data_dir / "categorical_mappings.json"

        # Validate sequence type - only allow ESM variants
        if sequence_type not in ['esm_zip']:
            raise ValueError(f"DataModule only supports only 'esm_zip' sequence type, got: {sequence_type}")

        # directory structure:
        # output_dir / split / exp_dir_prefix
        self.exp_dir: Path = Path(exp_dir)
        self.split_dir: Path = self.exp_dir.parent
        self.exp_data_dir: Path = self.split_dir.parent
        self.data_dir = data_dir

        if dataset_params is None:
            dataset_params = {}

        # Create simplified SMILES dataloader
        self.molecular_dataloader = MolecularDataloaderSMILES(
            data_dir=data_dir,
            dataset_options=dataset_params,
        )

        self.float_type = float_type
        self.batch_volume = batch_volume
        self.max_size_batch = max_size_batch
        self.num_workers = num_workers
        self.splitting_method = splitting_method
        self.esm_model_name = esm_model_name

        # Only support ESM embeddings (float type data)
        self.sequence_dtype = getattr(torch, self.float_type)

        # Will be initialized in setup()
        self.train_dataset: Optional[torch.utils.data.Dataset] = None
        self.val_dataset: Optional[torch.utils.data.Dataset] = None
        self.test_dataset: Optional[torch.utils.data.Dataset] = None

        self.datasets: Dict[str, pd.DataFrame] = {}
        self.torch_datasets: Dict[str, torch.utils.data.Dataset] = {}

    @staticmethod
    def _read_df(csv_path: Path) -> pd.DataFrame:
        _use_columns = ['smiles', 'sequence', 'log_IC50', 'log_Ki', 'log_Kd', 'log_EC50', 'label', 'split',
                        'cluster_smi']
        df = pd.read_csv(csv_path, nrows=1)
        _use_columns = df.columns.intersection(_use_columns)

        dtypes = {"smiles": int, "sequence": int, "log_IC50": float,
                  "log_Ki": float, "log_Kd": float, "log_EC50": float,
                  "label": float, "split": str, "cluster_smi": str}

        df = pd.read_csv(csv_path, dtype=dtypes, usecols=_use_columns)
        return df

    @staticmethod
    def _read_df_and_compute_sequence_lengths(csv_path: Path, length_dict: dict) -> pd.DataFrame:
        # to reduce RAM load only necessary columns
        df = ProtobindDataModule._read_df(csv_path)
        df['sequence_length'] = df["sequence"].map(length_dict)

        # sort by sequence length to increase the batching efficiency.
        df.sort_values(by="sequence_length", inplace=True)
        return df

    def check_splits_exist(self):
        """ Tries to find that train-test split exist """
        if (self.split_dir / "train.csv").exists():
            assert (self.split_dir / "valid.csv").exists()
            assert (self.split_dir / "test.csv").exists()
            logger.info(f"train.csv/valid.csv/test.csv exist, "
                        f"no new splits will be created for {self.splitting_method}")
            return True

        return False

    def prepare_data_split(self, seed=777, valid_fraction=0.1, test_fraction=0.1):
        """ Create train.csv, val.csv and test.csv in the experiment dir """

        if self.check_splits_exist():
            return

        # Check that data exists
        for path in [self.csv_path, self.categorical_mappings_path]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Could not find {path}. Please download the data.")

        # load label data
        data_df = pd.read_csv(self.csv_path)

        # add clusters
        distance_data = list(self.csv_path.parent.glob('all_smiles_sparse_*.npz'))
        if len(distance_data) > 0:
            logger.info(f"Calculating clusters for SMILES and distance data {distance_data[0]}")
            clusters_smi = cluster_fpsim2(distance_data[0])
            len_ = len(data_df)
            data_df = data_df.merge(pd.Series(clusters_smi, name='cluster_smi'), left_on='smiles', right_index=True)
            assert data_df.shape[0] == len_, (f"Failed to merge clusters, {len_=} {data_df.shape=}"
                                              f" {clusters_smi.min()} {clusters_smi.max()}")
        else:
            raise FileNotFoundError(f'Could not find any all_smiles_sparse_*.npz in {str(self.csv_path.parent)}')

        # Create splits
        if self.splitting_method == SplittingMethod.RANDOM:
            train, valid, test = split_at_random(data_df, valid_fraction=valid_fraction,
                                                 test_fraction=test_fraction, seed=seed)
        else:
            raise NotImplementedError(
                f"Splitting method {self.splitting_method} is not implemented in simplified version.")

        train.to_csv(self.split_dir / "train.csv", index=False)
        valid.to_csv(self.split_dir / "valid.csv", index=False)
        test.to_csv(self.split_dir / "test.csv", index=False)

    def prepare_data(self, **kwargs):

        if kwargs.get('load', False):
            return

        if self.exp_dir.exists():
            logger.info(f"Experiment directory {self.exp_dir} already exists. All existing files "
                        f" will be kept. To create new data/split remove {self.exp_data_dir} or {self.split_dir}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Make train-test split
        default_split_kwargs = {'seed': 777,
                                'valid_fraction': 0.1,
                                'test_fraction': 0.1,
                                }
        # update from kwargs
        for key in default_split_kwargs.keys():
            if key in kwargs:
                default_split_kwargs[key] = kwargs[key]
        # Create new split or skip if exist
        self.prepare_data_split(**default_split_kwargs)

        # Prepare smiles (simplified - only tokenized smiles)
        self.molecular_dataloader.prepare_molecular_features()

    def setup(self, stage=None):
        """Loads and prepares the datasets for a given stage.

                This method is called by PyTorch Lightning. It performs the following steps:
                1.  Loads molecular features (tokenized SMILES).
                2.  Loads protein features (pre-computed ESM embeddings).
                3.  Loads data splits (train/val/test) from CSV files.
                4.  Initializes the PyTorch Datasets for each split.

                Args:
                    stage (str, optional): The stage to setup ('fit', 'validate', 'test', 'predict').
        """
        logger.info("Loading molecular features")

        # Load molecular features (simplified - only SMILES)
        self.molecular_dataloader.load_molecular_features()

        # Load protein features (only ESM embeddings)
        logger.info(f"Loading protein features {self.esm_model_name}")
        prot_embbeding_pt = self.data_dir / f'all_prots_{self.esm_model_name}.pt'

        if prot_embbeding_pt.exists():
            self.idx_to_sequence_data = torch.load(prot_embbeding_pt, map_location='cpu', weights_only=False)
            length_dict = {idx: emb.shape[0] for idx, emb in self.idx_to_sequence_data.items()}
            self.sequence_embedding_dim = next(iter(self.idx_to_sequence_data.values())).shape[1]
        else:
            raise FileNotFoundError(
                f"Packed proteins `all_prots_{self.esm_model_name}.pt` is not found in {self.data_dir}")

        # load data. Use integer dtypes for categorical features and float for labels.
        logger.info("Loading activity table")

        self.datasets = dict(zip(["train", "val", "test"],
                                 [self._read_df_and_compute_sequence_lengths(self.split_dir / f"{split}.csv",
                                                                             length_dict)
                                  for split in ["train", "valid", "test"]]))


        # initialise self.train_dataset, self.val_dataset, self.test_dataset
        for ds in ['train', 'val', 'test']:
            df_ds = self.datasets[ds]
            assert len(ds) > 0, f"{ds=} is empty"
            ds_proto = self.create_dataset(df_ds)
            ds_proto._is_train = (ds == 'train')
            self.torch_datasets[ds] = ds_proto

    def create_dataset(self, df, **kwargs):
        dataset_kwargs = self.molecular_dataloader.dataset_kwargs
        dataset_class = DatasetMolecularEmbeddings

        cluster_smi = None
        sample_smiles = dataset_kwargs.get('sample_smiles', False)
        if sample_smiles:
            cluster_smi = df['cluster_smi'].values

        logger.info(f"Creating dataset: using {dataset_class=}")
        ds_proto = dataset_class(
            sequence_embedding=(self.idx_to_sequence_data),
            smiles_embeddings=self.molecular_dataloader.get_features(),
            sequences=df['sequence'].values,
            sequences_length=df['sequence_length'].values,
            smiles=df['smiles'].values,
            dtype=self.float_type,
            cluster_smi=cluster_smi,
            **dataset_kwargs,
            **kwargs,
        )
        return ds_proto

    def get_dataloader(self, dataset, shuffle, use_sampler=True, pin_memory=True):
        if use_sampler:
            sampler = RandomizedBatchSampler(sequence_length=dataset.sequences_length,
                                             shuffle=shuffle,
                                             batch_volume=self.batch_volume,
                                             batch_size=self.max_size_batch)
            return DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=dataset.collate_fn,
                              num_workers=self.num_workers, pin_memory=pin_memory)
        else:
            return DataLoader(dataset=dataset, collate_fn=dataset.collate_fn, batch_size=self.max_size_batch,
                              num_workers=self.num_workers, pin_memory=pin_memory, shuffle=shuffle)

    def train_dataloader(self, use_sampler=True, shuffle=True):
        return self.get_dataloader(self.torch_datasets['train'], shuffle=shuffle, use_sampler=use_sampler)

    def val_dataloader(self, use_sampler=True, shuffle=False):
        return self.get_dataloader(self.torch_datasets['val'], shuffle=shuffle, use_sampler=use_sampler)

    def test_dataloader(self, use_sampler=True, shuffle=False):
        return self.get_dataloader(self.torch_datasets['test'], shuffle=shuffle, use_sampler=use_sampler)

    def predict_dataloader(self, dataset='test', use_sampler=False, shuffle=False):
        return self.get_dataloader(self.torch_datasets[dataset], shuffle=shuffle, use_sampler=use_sampler)

    def get_smiles_embedding_dim(self):
        return self.molecular_dataloader.embedding_size

    def get_sequence_embedding_dim(self):
        return self.sequence_embedding_dim


class DatasetNumpy(Dataset):
    """ Dataset for feeding model with sequences and ligands embeddings """

    def __init__(self, *, sequence_embedding: Tuple[np.array, np.array],
                 smiles_embeddings: np.ndarray,
                 sequences: np.ndarray,
                 sequences_length: np.ndarray,
                 smiles: np.ndarray,
                 dtype='float16',
                 **kwargs,
                 ):
        """
        Args:
            sequence_embedding: embedding for sequences - 1 per each sequence
            smiles_embeddings: embedding for smiles - 1 per each smile
            sequences: sequence label in the dataset - 1 per sample
            sequences_length: sequence length in the dataset - 1 per sample
            smiles: smile label in the dataset - 1 per sample
        """
        assert len(sequences) == len(sequences_length), f"{len(sequences)=}  {len(sequences_length)=}"
        assert len(sequences) == len(smiles), f"{len(sequences)=}  {len(smiles)=}"

        self.data_sequence = sequence_embedding
        self.smiles_embeddings = self.init_smiles_embeddings(smiles_embeddings)
        self.sequences_length = sequences_length
        self.sequences = sequences
        self.smiles = smiles
        self.float_type = getattr(torch, dtype)

        # Only support ESM embeddings (float type)
        self.sequence_dtype = self.float_type
        self._is_train = False  # this parameter is assigned in during model.setup()

        # SMILES SAMPLER
        sample_smiles = kwargs.get('sample_smiles', False)
        self.cluster_smiles = kwargs.get('cluster_smi', None)
        self.smiles_to_cluster = None
        if sample_smiles:
            self.group_smiles(self.cluster_smiles)
            self.get_smiles_id = self._smiles_id_sample
        else:
            self.get_smiles_id = self._smiles_id_as_ind

    def init_smiles_embeddings(self, smiles_embeddings):
        return smiles_embeddings

    def group_smiles(self, clusters):
        """ for each sequence group similar smiles to list for random sampling during training """

        len_ = len(self.sequences)
        df = pd.DataFrame(data={'smiles': self.smiles, 'sequence': self.sequences, 'cluster_smi': clusters,
                                'sequences_length': self.sequences_length}
                          ).groupby(['cluster_smi', 'sequence', 'sequences_length'], as_index=False).agg(list)
        self.smiles_to_cluster = df['smiles'].values
        self.sequences = df['sequence'].values
        self.cluster_smiles = df['cluster_smi'].values
        self.sequences_length = df['sequences_length'].values
        logger.info(f"Sampling from similar smiles is ON, dataset size reduced from {len_} to {len(self.sequences)}")

    def _smiles_id_as_ind(self, idx: int) -> int:
        """ Get smiles is from array self.smiles """
        return self.smiles[idx]

    def _smiles_id_sample(self, idx) -> int:
        """ Sample smile id from grouped SMILES from same cluster"""
        return np.random.choice(self.smiles_to_cluster[idx])

    def __len__(self) -> int:
        # the number of entries in the dataset
        return len(self.sequences)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, int]:

        seq_id = self.sequences[idx]
        smi_id = self.get_smiles_id(idx)

        return (self.parametrize_sequence(seq_id),
                self.parametrize_smiles(smi_id),
                self.sequences_length[idx])

    def parametrize_smiles(self, smiles_id: int) -> np.array:
        return self.smiles_embeddings[smiles_id]

    def parametrize_sequence(self, sequence_id: int) -> np.array:
        return self.data_sequence[sequence_id]

    @staticmethod
    def _collate_fn_pack(batch):
        """ Pack dataset samples to sequences of sequences, smiles, sequence_lengths  """
        return zip(*batch)

    def _pad_sequence(self, sequences: List[np.ndarray]) -> torch.Tensor:
        return pad_sequence([torch.tensor(s, dtype=self.sequence_dtype) for s in sequences], batch_first=True,
                            padding_value=ProtobindDataModule.MASK_VALUE)

    def collate_fn(self, batch: Tuple[np.ndarray, np.ndarray, int ]) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Collates samples into a single batch, padding sequences to the same length.

        Args:
            batch : A tuple of samples, where each sample is the output of `__getitem__`.

        Returns:
            Tuple: A tuple containing batched tensors:
                - ((torch.Tensor, torch.Tensor)): A tuple of padded protein sequences
                  and a tensor of their original lengths.
                - (torch.Tensor): A batch of SMILES embeddings.
        """

        sequences, smiles, sequence_lengths = self._collate_fn_pack(batch)

        padded_sequences = self._pad_sequence(sequences)

        return ((padded_sequences, torch.tensor(sequence_lengths, dtype=torch.int32)),
                torch.tensor(np.array(smiles), dtype=self.float_type))


class DatasetMolecularEmbeddings(DatasetNumpy):
    """A dataset for masked diffusion models using protein embeddings and tokenized SMILES.

    This class extends `DatasetNumpy` to handle variable-length, tokenized SMILES
    representations from a `RandomizedSmilesDataset`. It overrides methods for
    SMILES parameterization and batch collation to support this token-based approach,
    which is required for diffusion models.
    """

    def parametrize_smiles(self, smiles_id: int) -> Tuple[np.array, int]:
        mol = self.smiles_embeddings[smiles_id]
        return mol, len(mol)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.array, int, int, int, int]:
        """Retrieves a single data sample with tokenized SMILES.

                Unlike the parent class, this method returns the tokenized SMILES
                and its length instead of a fixed-size embedding.
         """
        seq_id = self.sequences[idx]
        smi_id = self.smiles[idx]
        return (self.parametrize_sequence(seq_id),) + self.parametrize_smiles(smi_id) + (
            self.sequences_length[idx], seq_id, smi_id)

    def init_smiles_embeddings(self, smiles_embeddings):
        if isinstance(smiles_embeddings, RandomizedSmilesDataset):
            return smiles_embeddings
        else:
            raise ValueError("version only supports RandomizedSmilesDataset")

    def collate_fn(self, batch: List[Tuple[np.ndarray, np.array, int, int, int, int]]) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor, torch.Tensor]:

        """Collates samples into a batch, padding both protein and SMILES sequences.

        Args:
            batch (list): A list of samples, where each sample is the output of __getitem__.

        Returns:
            Tuple: A tuple containing the final batched tensors for the model:
                - ((torch.Tensor, torch.Tensor)): Padded protein sequences and their lengths.
                - ((torch.Tensor, torch.Tensor)): Padded tokenized SMILES and their lengths.
                - (torch.Tensor): A batch of sequence IDs.
                - (torch.Tensor): A batch of SMILES IDs.
        """

        sequences, atom, atom_lengths, sequence_lengths, seq_id, smi_id \
            = self._collate_fn_pack(batch)

        padded_sequences = self._pad_sequence(sequences)  # padding proteins sequences
        padded_atom = pad_sequence([s.to(dtype=self.float_type) for s in atom], batch_first=True)
        atom_lengths = torch.tensor(atom_lengths, dtype=torch.int32)

        return ((padded_sequences, torch.tensor(sequence_lengths, dtype=torch.int32)),
                (padded_atom, atom_lengths),
                torch.tensor(seq_id, dtype=torch.int32),
                torch.tensor(smi_id, dtype=torch.int32),
                )


class MolecularDataloaderSMILES(object):
    """
    molecular dataloader that only supports tokenized SMILES
    with ChemformerTokenizer for masked diffusion models.
    """

    def __init__(self, *,
                 data_dir: Path,
                 dataset_options: Optional[dict] = None):
        """
        Args:
            data_dir: path to data folder containing tokenizer files and dict with all smiles and fasta sequences
            dataset_options: dictionary with additional parameters used to create pytorch Dataset
        """
        self.data_dir = data_dir
        if dataset_options is None:
            logger.info('Setting tokenizer file name to tokenizer_smiles_diffusion.json')
            dataset_options = {'tokenizer_json_name': 'tokenizer_smiles_diffusion'}
        self.dataset_options = dataset_options

        self.tokenizer_path = self.data_dir / f"{dataset_options['tokenizer_json_name']}.json"
        self.tokenizer = ChemformerTokenizer(filename=str(self.tokenizer_path))
        self.randomize = dataset_options.get('randomize', False)
        self.smiles_embedding_dim = 1  # For tokenized SMILES, embedding dim is 1
        self.baseline_dim = 0  # this version doesn't support baseline features

    def prepare_molecular_features(self):
        """Prepare molecular features"""
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(
                f"Could not find tokenizer at {self.tokenizer_path}. Please ensure the tokenizer file exists.")
        logger.info(f"Found ChemformerTokenizer at {self.tokenizer_path}")

    def load_molecular_features(self):
        """Load molecular features - loads SMILES mappings"""
        categorical_mappings_path = self.data_dir / 'categorical_mappings.json'
        if not categorical_mappings_path.exists():
            raise FileNotFoundError(f"categorical_mappings.json not found in data_dir: {self.data_dir}")

        self.smiles_dataset = RandomizedSmilesDataset.from_json(
            categorical_mappings_path,
            tokenizer=self.tokenizer,
            randomize=self.randomize
        )

    def get_features(self):
        """Get the SMILES dataset for tokenized molecular features"""
        return self.smiles_dataset

    @property
    def dataset_kwargs(self):
        """Return dataset options for creating pytorch datasets"""
        return self.dataset_options

    @property
    def embedding_size(self):
        """Get embedding size for tokenized SMILES"""
        return self.smiles_embedding_dim


class InferenceDataset(Dataset):
    """Creates a dataset for running inference on a single protein embedding.

    This utility dataset repeatedly yields the same batch, created by expanding
    a single input embedding. It's designed for generating a large number of
    ligand samples for one protein target without a traditional dataset structure.
    """
    def __init__(self, embedding: torch.Tensor, batch_size: int, n_batches: int):
        """Initializes the inference dataset.

        Args:
            embedding (torch.Tensor): The single protein embedding tensor to be used.
            batch_size (int): The number of times to repeat the embedding in each batch.
            n_batches (int): The total number of identical batches the dataset should yield.
        """
        self.embedding_single = embedding
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.seq_len = embedding.shape[1]

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, idx: int) -> Tuple:
        """Generates a full batch ready for model inference.

        Note: This method ignores the `idx` argument and always returns the same
        batch, which is constructed by expanding the stored protein embedding.
        It includes dummy values to match the data structure expected by the model.

        Returns:
            Tuple: A tuple containing pre-batched tensors:
                - ((torch.Tensor, torch.Tensor)): Expanded protein embeddings and their lengths.
                - (torch.Tensor): A dummy NaN tensor (placeholder for SMILES).
                - (torch.Tensor): A batch of placeholder sequence IDs (-1).
                - (torch.Tensor): A dummy NaN tensor (placeholder for smiles IDs).
        """
        embedding = self.embedding_single.expand(self.batch_size, -1, -1).contiguous()
        lengths = torch.full((self.batch_size,), self.seq_len, dtype=torch.int32)
        seq_ids = torch.full((self.batch_size,), -1, dtype=torch.int32) #seq_ids dont exist for new sequences
        return (
            (embedding, lengths),
            torch.tensor(float('nan')),
            seq_ids,
            torch.tensor(float('nan')),
        )