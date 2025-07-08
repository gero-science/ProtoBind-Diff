#!/usr/bin/env python3

"""
Training script for ProtobindMaskedDiffusion model.

Example usage:
python train.py -o ./experiment_dir --exp_dir_prefix experiment_name --yaml ./configs/masked_diffusion.yaml
"""
import json
import logging
import warnings
from pathlib import Path
import numpy as np
import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.strategies import DDPStrategy

from protobind_diff.parsers import argparse_config_train
from protobind_diff.model import ModelGenerator
from protobind_diff.data_loader import ProtobindDataModule, SplittingMethod


warnings.filterwarnings("ignore", message="Trying to infer the `batch_size` from an ambiguous collection.",
                        category=UserWarning)
logger = logging.getLogger("lightning")


def setup_dataloader(*args, **kwargs):
    """Initializes and prepares the ProtobindDataModule.

        This function handles creating the data splits on disk (if necessary)
        and setting up the dataset for use by the Trainer.

        Args:
            **kwargs: A dictionary of arguments, typically from argparse.

        Returns:
            ProtobindDataModule: The fully set up data module.
    """

    data_dir = Path(kwargs["data_dir"])
    exp_dir = kwargs.get('exp_dir', None)
    exp_dir = ModelGenerator.get_exp_dir(
        exp_dir=exp_dir, output_dir=kwargs["output_dir"],
        exp_dir_prefix=kwargs["exp_dir_prefix"], split=kwargs["split"]
    )
    sm_dict = {sm.name: sm for sm in SplittingMethod}

    # dataset params for masked diffusion
    dataset_params = {
        'randomize': kwargs.get('randomize_smiles', False),
        'sample_smiles': kwargs.get('sample_smiles', False),
        'tokenizer_json_name': kwargs['tokenizer_json_name']
    }

    data_module = ProtobindDataModule(
        data_dir=data_dir,
        exp_dir=exp_dir,
        splitting_method=sm_dict[kwargs["split"]],
        batch_volume=kwargs["batch_volume"],
        num_workers=kwargs["num_workers"],
        sequence_type=kwargs.get('sequence_type', 'esm_zip'),
        esm_model_name=kwargs.get('esm_model_name', 'esm2_t33_650M_UR50D'),
        max_size_batch=kwargs['max_size_batch'],
        dataset_params=dataset_params,
        float_type=kwargs['float'],
    )

    logger.info(f"Experiment will run in {exp_dir.absolute().resolve()}.")

    # Note: prepare_data will create file only if it does not exist
    if not kwargs['load']:
        data_module.prepare_data(
            cutoff=kwargs["splitting_cutoff"],
            seed=kwargs['seed'],
            split_target_key=kwargs.get("split_target_key", "sequence"),
            valid_fraction=kwargs['valid_fraction'],
            test_fraction=kwargs['test_fraction'],
        )
    data_module.setup()
    return data_module

def main():
    args = argparse_config_train()
    torch.set_float32_matmul_precision('medium')

    # --- Data and Model Setup ---
    exp_dir = ModelGenerator.get_exp_dir(
        exp_dir=None, output_dir=args.output_dir,
        exp_dir_prefix=args.exp_dir_prefix, split=args.split)

    if args.version < 0:
        version = None
    else:
        version = args.version

    data_module = setup_dataloader(**vars(args))

    # --- Load Model from Checkpoint or Initialize New ---
    last_checkpoint_path = None
    if args.load:
        if args.best:
            last_checkpoint_path = exp_dir / f"lightning_logs/version_{version}/checkpoints/"
            last_checkpoint_path = list(last_checkpoint_path.glob('protobind-*-*=*.ckpt'))
            last_checkpoint_path = sorted(last_checkpoint_path)[0]
        else:
            last_checkpoint_path = exp_dir / f"lightning_logs/version_{version}/checkpoints/last.ckpt"
        if not last_checkpoint_path.exists():
            raise ValueError(f'{last_checkpoint_path=} not found')

        model = ModelGenerator.load_from_checkpoint(
            last_checkpoint_path,
            max_size_batch=args.max_size_batch,
            batch_volume=args.batch_volume,
            float=args.float,
            seed=args.seed,
            learning_rate=args.learning_rate,
            tokenizer_json_name=args.tokenizer_json_name,
            seq_embedding_dim=data_module.get_sequence_embedding_dim(),
        )
    else:
        params = vars(args)
        params['seq_embedding_dim'] = data_module.get_sequence_embedding_dim()
        model = ModelGenerator(**params)

    # --- Setup Callbacks and Logger ---
    tb_logger = pl.loggers.TensorBoardLogger(exp_dir, version=version)

    monitor = "val_loss"

    checkpoint_callback_top_k = pl.callbacks.ModelCheckpoint(
        save_top_k=2,
        monitor=monitor,
        mode="min",
        save_last=True,
        filename=f"protobind-simplified-{{epoch:04d}}-diff_loss={{{monitor}:.4f}}",
        auto_insert_metric_name=False
    )

    try:
        gpus = int(args.gpus)
    except ValueError:
        gpus = json.loads(args.gpus)
    distributed = False
    if isinstance(gpus, int):
        if gpus > 1:
            distributed = True
    elif isinstance(gpus, list):
        if len(gpus) > 1:
            distributed = True
    else:
        raise ValueError(f"{type(gpus)=}")

    if distributed:
        logger.info("Running in a distributed mode, sampler disabled")
        use_sampler = False
        strategy = DDPStrategy(gradient_as_bucket_view=True,
                               static_graph=True,
                               find_unused_parameters=True)
    else:
        use_sampler = True
        strategy = 'auto'

    val_dataloader = data_module.val_dataloader(use_sampler=use_sampler)
    train_dataloader = data_module.train_dataloader(use_sampler=use_sampler)

    progress_bar = RichProgressBar()
    # --- Initialize and Run Trainer ---
    trainer = pl.Trainer(max_epochs=args.epochs,
                         val_check_interval=len(train_dataloader),
                         check_val_every_n_epoch=None,
                         log_every_n_steps=min(50, len(train_dataloader)),
                         precision=args.precision,
                         default_root_dir=exp_dir,
                         callbacks=[checkpoint_callback_top_k, progress_bar],
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         gradient_clip_algorithm="norm",
                         gradient_clip_val=1.,
                         logger=tb_logger,
                         devices=gpus,
                         accelerator="gpu",
                         strategy=strategy,
                         )

    # For efficient processing of variable-length sequences, this model implements a strategy of flexible batching.
    # The procedure consists of two main steps: first, sorting the entire dataset by sequence length,
    # and second, shuffling the indices within each bucket of uniform sequence length.
    batch_sizes = []
    for batch_idx, data in enumerate(train_dataloader):
        batch_sizes.append(len(data[0][0]))
        if batch_idx > 1000:
            break
    batch_sizes = np.array(batch_sizes)
    logger.info(f'min/max/mean train batch length:'
                f' {np.min(batch_sizes)}/{np.max(batch_sizes)}/{np.mean(batch_sizes).astype(int)}')

    batch_sizes = np.array(batch_sizes)
    logger.info(f'min/max/mean val batch length:'
                f' {np.min(batch_sizes)}/{np.max(batch_sizes)}/{np.mean(batch_sizes).astype(int)}')
    del batch_sizes

    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path=last_checkpoint_path)

if __name__ == "__main__":
    main()
