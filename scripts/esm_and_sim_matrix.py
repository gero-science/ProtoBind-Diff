#!/usr/bin/env python3
import argparse
import os
import re
import numpy as np
from pathlib import Path
import torch
import esm
from tqdm import tqdm
import json
import pandas as pd
from FPSim2.io import create_db_file
from FPSim2 import FPSim2Engine
from scipy import sparse
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def calculate_distance_matrix(*args, **kwargs):
    """Calculates a sparse Tanimoto distance matrix for all SMILES in the dataset.

        This function reads SMILES from the categorical mappings file, creates an
        FPSim2 fingerprint database if it doesn't exist, and then computes a
        symmetric distance matrix, saving it as a sparse NumPy array.

        Args:
            **kwargs: A dictionary of arguments, expected to contain `data_dir`,
                `splitting_cutoff`, and `num_workers`.
        """
    splitting_cutoff = 1 - kwargs["splitting_cutoff"]
    DATA_DIR = kwargs["data_dir"]
    num_workers = kwargs["num_workers"]

    if not os.path.exists(DATA_DIR / 'all_smiles.h5'):
        with open(DATA_DIR / "categorical_mappings.json", "rt") as f:
            categorical_mappings = json.load(f)

        all_smiles = pd.DataFrame.from_dict(categorical_mappings['smiles'], orient='index').reset_index()
        all_smiles.columns = ['SMILES', 'Name']
        all_smiles = all_smiles.sort_values('Name')
        smi_path = DATA_DIR / 'all_smiles.smi'
        all_smiles.to_csv(smi_path, index=False, sep='\t', header=False)
        create_db_file(str(smi_path),
                       str(DATA_DIR / 'all_smiles.h5'),
                       'Morgan', {'radius': 2, 'nBits': 2048, 'useFeatures': True})
    fp_filename = DATA_DIR / 'all_smiles.h5'
    fpe = FPSim2Engine(fp_filename)

    csr_matrix = fpe.symmetric_distance_matrix(splitting_cutoff, n_workers=num_workers)
    sparse.save_npz(DATA_DIR / f"all_smiles_sparse_{splitting_cutoff}.npz", csr_matrix)


def calculate_esm_embeddings(data_dir, out_dir: Path, model_name: str = "esm2_t36_3B_UR50D"):
    """Generates and saves ESM embeddings for all protein sequences in the dataset.

        It reads sequences from the categorical mappings file, loads a specified ESM
        model, and iterates through all sequences to compute their embeddings.
        The resulting embeddings are saved as a dictionary in a .pt file.

        Args:
            data_dir (Path): The directory containing `categorical_mappings.json`.
            out_dir (Path): The directory where the final embeddings file will be saved.
            model_name (str): The name of the ESM model to use for embeddings.
        """

    # Set up output directory
    out_dir.mkdir(exist_ok=True)

    # download model data from the hub
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    number_layers = re.search(r'_t(\d+)_', model_name)
    number_layers = int(number_layers.group(1))

    # Determine the device
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use CUDA if available
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Use MPS for Apple Silicon if available
    else:
        device = torch.device("cpu")  # Fallback to CPU

    model = model.to(device)

    batch_converter = alphabet.get_batch_converter()
    model.eval()

    file_path = data_dir / "categorical_mappings.json"
    with open(file_path, "r") as f:
        seq_map = json.load(f)["sequence"]

    sequence_idx_pairs = [(int(idx), seq) for seq, idx in seq_map.items()]

    failed_seq = []
    emb_dict = {}

    for idx, seq in tqdm(sequence_idx_pairs, desc="Downloading ESM-2 embeddings", total=len(sequence_idx_pairs)):

        try:
            data = [(idx, seq)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)

            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[number_layers])
            token_representations: torch.Tensor = results["representations"][number_layers]

            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            # NOTE: the last token is <eos>
            token_representations = token_representations[:, 1:-1, :]
            assert token_representations.shape[1] == len(seq), f"{token_representations.shape[1]=} {len(seq)=}"

            emb_dict[idx] = token_representations.squeeze().cpu().numpy().astype(np.float16)
            del token_representations, results, batch_tokens

            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            failed_seq.append(seq)
            continue

    del model
    del alphabet

    fname = out_dir / f'all_prots_{model_name}.pt'
    with open(fname, 'wb') as f:
        torch.save(emb_dict, f)
        del emb_dict

    np.savetxt(out_dir / Path(f"failed_sequence_{model_name}.txt"), failed_seq, fmt="%s")


if __name__ == "__main__":


    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=Path, default=project_root / 'data' / 'experiments' / 'diffusion',
                        help="Data directory with `categorical_mappings.json` file")
    parser.add_argument('-o', '--out_dir', type=Path, default=project_root / 'data' / 'experiments' / 'diffusion',
                        help="Data directory where save embeddings. For ESM-2 embeddings creates in this directory "
                             "a file with name ${model_name}.pt. For Tanimoto similarity matrix creates in this "
                             "directory a file with name ${1-splitting_cutoff}.pt")

    parser.add_argument('--model_name', type=str, default='esm2_t33_650M_UR50D',
                        help="ESM model name. See https://github.com/facebookresearch/esm")

    parser.add_argument('--cache', type=str, default='./cache',  help='Cache folder for ckpt')
    parser.add_argument('--splitting_cutoff', type=float, default=0.4,
                        help="Threshold value for the Tanimoto similarity matrix calculation")
    parser.add_argument('--num_workers', type=int, default=8,
                        help="Number of workers for the Tanimoto similarity matrix calculation")
    args = parser.parse_args()

    torch.hub.set_dir(args.cache) # Setup cache dir

    if args.model_name != 'all':
        calculate_esm_embeddings(data_dir=args.data_dir, out_dir=args.out_dir, model_name=args.model_name)
    else:
        esm_models = ['esm2_t36_3B_UR50D', 'esm2_t33_650M_UR50D', 'esm2_t30_150M_UR50D']
        for model in esm_models:
            calculate_esm_embeddings(data_dir=args.data_dir, out_dir=args.out_dir, model_name=model)
    print("Calculating similarity matrix")
    calculate_distance_matrix(**vars(args))