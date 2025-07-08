import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "ai-gero/ProtoBind-Diff"

FILES_TO_DOWNLOAD = [
    "data.csv",
    "categorical_mappings.json",
    "tokenizer_smiles_diffusion.json"
]

def download_data(destination_dir: Path):
    """
    Downloads all required data files from the Hugging Face Hub
    to the specified destination directory.

    Args:
        destination_dir (Path): The target directory to save the files.
    """
    print(f"Target directory: {destination_dir.resolve()}")

    destination_dir.mkdir(parents=True, exist_ok=True)

    for filename in FILES_TO_DOWNLOAD:
        print(f"\nDownloading '{filename}'...")
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=destination_dir,
            )
            print(f"Successfully downloaded '{filename}' to {destination_dir}")
        except Exception as e:
            print(f"Failed to download '{filename}'. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/experiments/diffusion",
        help="The directory where data files will be saved."
    )
    args = parser.parse_args()

    destination_path = Path(args.output_dir)

    download_data(destination_path)
