import argparse, sys
from typing import Optional, Tuple
from pathlib import Path
import esm
import os
import torch
import numpy as np
import re
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from protobind_diff.model import ModelGenerator
from protobind_diff.data_loader import InferenceDataset
from huggingface_hub import hf_hub_download

REPO_ID = "ai-gero/ProtoBind-Diff"
FILENAME = "model.ckpt"
TOKENIZER_FILENAME = "tokenizer_smiles_diffusion.json"

class ProtobindInference():
    """
    Simplified inference class that only supports ProtobindMaskedDiffusion model.
    """

    def __init__(self, checkpoint_path, tokenizer_path,
                 sequence_embedding_dim, lig_max_length: int=170, nucleus_p: float=0.9,
                 eta: float=0.1, sampling_steps: int=250,
                 **kwargs):
        self.checkpoint_path = Path(checkpoint_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.sequence_embedding_dim = sequence_embedding_dim

        # Set up sampler params
        self.lig_max_length = lig_max_length
        self.nucleus_p = nucleus_p
        self.eta = eta
        self.sampling_steps = sampling_steps

        # Load model
        self.model = self.load_model()

    def predict_on_dataloader(self, dl, devices=1, accelerator='cuda') -> Tuple[np.ndarray, np.ndarray]:
        if accelerator == 'cuda':
            torch.set_float32_matmul_precision('medium')
            precision = "16-mixed"
        else:
            precision = "32-true"
        trainer = pl.Trainer(precision=precision, use_distributed_sampler=False,
                             inference_mode=True, accelerator=accelerator, devices=devices)
        predictions_batches = trainer.predict(model=self.model, dataloaders=dl)
        return predictions_batches

    def load_model(self):
        """Simplified model loading - only supports ModelGenerator"""
        model = ModelGenerator.load_from_checkpoint(
            self.checkpoint_path,
            tokenizer_path=self.tokenizer_path,
            seq_embedding_dim=self.sequence_embedding_dim,
            load=True,
        )
        model.model_length = self.lig_max_length
        model.nucleus_p = self.nucleus_p
        model.eta = self.eta
        model.sampling_steps = self.sampling_steps
        model.model.eval()
        return model

def get_esm_embedding(sequence: str, model_name: str, device: torch.device) -> torch.Tensor:
    """Generates a protein embedding using a pre-trained ESM model.

    Args:
        sequence (str): The amino acid sequence.
        model_name (str): The name of the ESM model to use.
        device (torch.device): The device to run the model on.

    Returns:
        torch.Tensor: The final residue-level embedding tensor, with start/end tokens removed.
    """
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    number_layers = re.search(r'_t(\d+)_', model_name)
    number_layers = int(number_layers.group(1))

    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("protein", sequence)])
    tokens = tokens.to(device)
    with torch.no_grad():
        out = model(tokens, repr_layers=[number_layers])
    return out["representations"][number_layers][:, 1:-1, :]  # [1, seq_len, emb_dim]

def download_from_hub_hf(cache: Path, filename) -> Path:
    """
    Fetch file from Hugging Face into `cache`.
    Returns the local path to the file inside HF’s cache structure.
    """
    cache.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        cache_dir=cache,
    )
    return Path(local_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence",  help="Amino acid sequence (1-letter code)")
    parser.add_argument("--output_dir", default="./outputs", help="Output dir for SMILES")
    parser.add_argument("--output", default="generated_smiles.txt", help="Output file for generated SMILES")
    parser.add_argument("--n_batches", type=int, default=5, help="Number of batches to generate for this sequence")
    parser.add_argument("--batch_size", type=int, default=10, help="Max number of generated molecules per batch")
    parser.add_argument("--fasta_file", default="./examples/input.fasta",  help="Input FASTA file")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the model checkpoint")
    parser.add_argument('--model_name', type=str, default='esm2_t33_650M_UR50D',
                        help="ESM model name. See https://github.com/facebookresearch/esm")
    parser.add_argument('--tokenizer_path', help='Path to tokenizer.json file. If not provided, uses a default path and downloads if needed.')
    parser.add_argument('--cache', type=str, default = "./cache", help='Cache folder for ckpt')

    parser.add_argument("--sampling_steps", type=int, default=250, help="Number of steps during sampling")
    parser.add_argument("--lig_max_length", type=int, default=170, help="Max length of generated molecules")
    parser.add_argument("--nucleus_p", type=float, default=0.9,
                        help="Value of the nucleus sampling parameter. For more details, see https://arxiv.org/abs/2503.00307")
    parser.add_argument("--eta", type=float, default=0.1,
                        help="Value of the probability of remasking. For more details, see https://arxiv.org/abs/2503.00307")

    args = parser.parse_args()
    if args.fasta_file:
        sequence = str(next(SeqIO.parse(args.fasta_file, "fasta")).seq)
    elif args.sequence:
        sequence = args.sequence.strip().upper()
    else:
        sys.exit("Error: provide --sequence of --fasta_file")

    if args.checkpoint_path:
        ckpt_path = Path(args.checkpoint_path)
    else:
        torch.hub.set_dir(args.cache)  # for ESM model
        ckpt_path = download_from_hub_hf(Path(args.cache), FILENAME)

    if args.tokenizer_path:
        tokenizer_path = Path(args.tokenizer_path)
        if not tokenizer_path.exists():
            sys.exit(f"Error: Tokenizer file not found at specified path: {tokenizer_path}")
    else:
        tokenizer_path = download_from_hub_hf(Path(args.cache), TOKENIZER_FILENAME)

    # Determine the device
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use CUDA if available
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Use MPS for Apple Silicon if available
    else:
        device = torch.device("cpu")  # Fallback to CPU

    embedding = get_esm_embedding(sequence, args.model_name, device).to(dtype=torch.bfloat16)
    sequence_embedding_dim = embedding.shape[2]
    dataset = InferenceDataset(embedding, batch_size=args.batch_size, n_batches=args.n_batches)
    loader = DataLoader(dataset, batch_size=None)
    model = ProtobindInference(ckpt_path, tokenizer_path, sequence_embedding_dim,
                               sampling_steps=args.sampling_steps, nucleus_p=args.nucleus_p,
                               eta=args.eta, lig_max_length=args.lig_max_length,)

    predictions = model.predict_on_dataloader(loader, accelerator=str(device))

    all_smiles = [smi for batch in predictions for smi in batch[0]]
    out_dir = Path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / args.output, "w") as f:
        f.write("SMILES\n")
        for smi in all_smiles:
            f.write(smi + "\n")


if __name__ == "__main__":
    main()
