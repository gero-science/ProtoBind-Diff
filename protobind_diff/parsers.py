import argparse
import logging
from pathlib import Path
from protobind_diff.data_loader import SplittingMethod
import yaml
from easydict import EasyDict as edict

logger = logging.getLogger("lightning")


def default_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    
    # Global config
    parser.add_argument('--yaml', type=Path, help="Load global config from yaml")

    # Experiment parameters
    parser.add_argument('--exp_dir_prefix', type=Path, required=True, help="Name for the directory where logs and checkpoints will be saved.")
    parser.add_argument('-d', '--data_dir', type=Path, default=Path('./data'), help="Path to directory with input data")
    parser.add_argument('-o', '--output_dir', type=Path, default=Path('.'), help="Path to global output directory")
    parser.add_argument('-l', '--load', action='store_true', help="Load the latest checkpoint")
    parser.add_argument('--version', default=-1, type=int, help="Version of model to load from the latest checkpoint")
    parser.add_argument('--best', action='store_true',
                        help="if `True` load best model (with best metrics) instead of the last")
    parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for the dataloader")
    parser.add_argument('--seed', type=int, default=777, help="Seed for reproducibility")
    
    # Core training parameters
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout value")
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Flat learning rate")
    parser.add_argument('--weight_decay', type=float, default=0., help="Weight decay")
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'AdamW'],
                        default='AdamW', help="Optimizer")

    # Trainer options
    parser.add_argument('--float', type=str, default='bfloat16', choices=['float16', 'float32', 'bfloat16'],
                        help="Float precision")
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help="Accumulate gradients over this number of batches")
    parser.add_argument('--gpus', type=str, default="[0]",
                        help="Number of gpus to use")

    # Dataloader hyperparameters
    parser.add_argument('--split', type=str, default='RANDOM',
                        choices=[sm.name for sm in SplittingMethod], required=False, help="Approach to split data. Only RANDOM split is implemented")
    parser.add_argument('--batch_volume', type=int, default=3 * (2000 ** 2),
                        help="Upper bound for BATCH_SIZE * (SEQ_LEN ^ 2) in a given batch")
    parser.add_argument('--max_size_batch', type=int, default=16,
                        help="Upper bound for the batch size")
    parser.add_argument('--splitting_cutoff', type=float, default=0.4,
                        help="Threshold value for the Tanimoto similarity matrix calculation")
    parser.add_argument('--split_target_key', type=str, default='sequence',
                        choices=['sequence', 'pfam_str', 'cluster'],
                        help="Which column use for defining unique targets")
    parser.add_argument('--valid_fraction', type=float, default=0.1, 
                        help="Fraction of data in a validation dataset")
    parser.add_argument('--test_fraction', type=float, default=0.1, 
                        help="Fraction of data in a test dataset")

    # Sequence parameters (only ESM variants supported)
    parser.add_argument('--sequence_type', type=str, choices=['esm_zip'],
                        help="Type of sequence. Supports only esm_zip embeddings in this version",
                        default='esm_zip')
    parser.add_argument('--esm_model_name', type=str, default='esm2_t33_650M_UR50D', help="ESM-2 embeddings name")

    # Ligand parameters
    parser.add_argument('--tokenizer_json_name', type=str, default='tokenizer_smiles_diffusion',
                        help='Name of tokenizer json file')
    parser.add_argument('--randomize_smiles', action='store_true',
                        help="Randomize SMILES during training")
    parser.add_argument('--sample_smiles', action='store_true',
                        help="Sample SMILES during training")

    # Decoder hyperparameters
    parser.add_argument('--num_heads_decoder', type=int, default=8,
                        help="Number of self-attention heads in the decoder")
    parser.add_argument('--num_decoder_layers', type=int, default=3, help="Number of decoder layers")
    parser.add_argument('--decoder_hidd_dim', type=int, default=512, help="Hidden dimension of decoder")
    parser.add_argument('--expand_feedforward', type=int, default=3, help="Multiplier for the decoder's feed-forward layer size")
    parser.add_argument('--decoder_name', type=str, choices=['decoder_re'],
                        default='decoder_re', help="Decoder type (this version supports only decoder_re)")
    return parser


def modify_argparser(args):
    """Validates and modifies parsed arguments after loading."""

    # Validate sequence type
    if args.sequence_type not in ['esm_zip']:
        raise ValueError(f"This version only supports 'esm_zip' sequence type, got: {args.sequence_type}")

    if args.num_decoder_layers <= 0:
        raise ValueError("Number of decoder layers must be greater than 0.")

    # Set precision based on float type
    if args.float == 'float32':
        args.precision = "32-true"
    elif args.float == 'float16':
        args.precision = "16-mixed"
    elif args.float == 'bfloat16':
        args.precision = "bf16-mixed"
    else:
        raise ValueError(f'{args.float=} is not supported')

    # Handle optimizer
    if args.optimizer == 'Adam' and args.weight_decay > 0:
        args.optimizer = 'AdamW'
        logger.warning("Optimizer changed from Adam to AdamW with non-zero weight_decay")

    return args


def parse_with_args(parser, args):
    """ Apply parser for default command line arguments or custom string/list (e.g. in jupyter notebooks) """
    if isinstance(args, str):
        args = parser.parse_args(args.split())
    elif isinstance(args, list):
        args = parser.parse_args(args)
    elif args is None:
        args = parser.parse_args()
    else:
        raise ValueError(f"Unexpected args type {type(args)=}")
    changed_args = {
        arg: value for arg, value in vars(args).items()
        if value != parser.get_default(arg)
    }
    return args, changed_args


def argparse_config_train(args=None):
    """ Get config for train. Can be used with `args` to run code from jupyter """
    parser = default_argparser()
    args, changed_args = parse_with_args(parser, args)
    args = load_yaml(args, changed_args)
    args = modify_argparser(args)  # should be called after `parser.parse_args` to modify some defaults
    return args


def load_yaml(args, changed_args):
    """Loads a YAML config, allowing command-line arguments to override it."""
    yaml_path = args.yaml
    if yaml_path is None:
        return args
    args = edict(vars(args))
    config_updates = edict(yaml.load(open(yaml_path, 'r'), Loader=yaml.FullLoader))

    for key, val in config_updates.items():
        if key not in changed_args:
            args[key] = val
        else:
            logger.info(f"Yaml value replaced from cmd line {key}={changed_args[key]}")
    return args