# ProtoBind-Diff: A Structure-Free Diffusion Language Model for Protein Sequence-Conditioned Ligand Design

[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ai-gero/ProtoBind-Diff)

Implementation of [ProtoBind-Diff: A Structure-Free Diffusion Language Model for Protein Sequence-Conditioned Ligand Design](https://www.biorxiv.org/content/10.1101/2025.06.16.659955v1)  by Lukia Mistryukova*, Vladimir Manuilov*, Konstantin Avchaciov*, and Peter O. Fedichev.


ProtoBind-Diff is a masked diffusion language model that generates target-specific, high-quality small-molecule ligands. Trained entirely without structural input, it enables structure-independent ligand design across the full proteome and matches structure-based methods in docking and Boltz-1 benchmarks.


If you have questions, feel free to open an issue or send us an email at lukiia.mistriukova@gero.ai, konstantin.avchaciov@gero.ai, vladimir.manuylov@gero.ai, and peter.fedichev@gero.ai.

![Alt Text](graphical-abstract.png)

You can also try out the model on [Hugging Face Spaces](https://huggingface.co/spaces/ai-gero/ProtoBind-Diff).

## Overview

This repository contains the evaluation toolkit and supporting data for **ProtoBind-Diff**. It is organized as follows:

##### `data/`
We selected 12 protein targets to benchmark molecular generation quality across different models.
- One folder per model (`pocket2mol/`, `targetdiff/`, etc.) containing SMILES files of generated ligands.
- `bindingdb/` – sets of random molecules (used as reference inactive molecules).
- `bindingdb_active/` – sets of true active molecules.
- `actives_bindingdb_cl` – clustered true actives; one representative per similarity cluster.
- `CrossDocked2020/` – cleaned PDB files and corresponding ligands for the targets.
- `fasta/` – FASTA sequences of the targets.

##### `notebooks/`
Jupyter notebooks to reproduce the paper figures: docking/Boltz-1 score distributions, interpretation of attention maps, chemical property distributions, and UMAP-based target specificity analysis. We also added `allign.ipynb` to show differences between canonical sequences and PDB receptor sequences.
 

##### `results/`
- Raw docking and Boltz-1 ipTM score tables for each method.

##### `protobind_diff/`
- Model code, train and inference scripts. 

##### `scripts/`
- Scripts to download and prepare data for training. 

## Usage

### Setup Environment

Clone the current repo

    git clone https://github.com/gero-science/ProtoBind-Diff.git

You can install the project locally in editable mode:

    python -m venv protodiff_env
    source protodiff_env/bin/activate    
    pip install -e .
Then you'll be able to run:

    protobind-infer #inference
    protobind-train #train


### Inference
This script generates potential binding molecules (in SMILES format) for a given protein sequence by computing the protein embeddings on-the-fly.

**Note**: The script automatically downloads the model checkpoint from Hugging Face Hub and uses the best available hardware.

You need to specify the protein sequence with `--fasta_file` or `--sequence`. 
And you are ready to run inference:

    protobind-infer --fasta_file examples/input.fasta

<details>
<summary>View All Command-Line Options</summary>

* **`--output_dir`**: Specify a different output folder (default: `../outputs`).
* **`--output`**: Change the output filename (default: `generated_smiles.txt`).
* **`--n_batches`**: Set the number of generation batches (default: 5).
* **`--batch_size`**: Set the number of molecules generated per batch (default: 10).
* **`--checkpoint_path`**: Use a local model checkpoint file.
* **`--tokenizer_path`**: Use a local tokenizer file.
* **`--model_name`**: Specify the ESM model for embeddings (default: `esm2_t33_650M_UR50D`).
* **`--cache`**: Set a custom cache folder for downloads (default: `../cache`).
* **`--sampling_steps`**: Set the number of steps during sampling (default: 250).
* **`--lig_max_length`**: Set the max length of generated molecules (default: 170).
* **`--nucleus_p`**: Set the value of the nucleus sampling parameter (default: 0.9).
* **`--eta`**: Set the value of the probability of remasking (default: 0.1).

</details>


### Model Training

Training the model involves a three-step pipeline: 1) downloading the dataset, 2) pre-processing the data to generate embeddings and similarity matrices, and 3) running the training script with a configuration file.

#### Step 1: Download Raw Data

First, download the necessary dataset files (protein/ligand pairs, tokenizer, etc.) from Hugging Face Hub.

**Run the command:**

    python scripts/download.py --output_dir ./data/experiments/diffusion

This will create a directory containing the raw data.csv and other necessary files.

#### Step 2: Pre-process Data
Next, you need to generate protein embeddings and a molecular similarity matrix from the raw data. This script performs both of these tasks. **Note**: This is a computationally intensive step. Calculating the ESM embeddings requires a GPU and at least 64 GB of RAM (depending on the size of the `categorical_mappings.json` file).

Run the command:

    python scripts/esm_and_sim_matrix.py --data_dir ./data/experiments/diffusion --out_dir ./data/experiments/diffusion

This uses the files in the `--data_dir` and saves the new files: `all_prots_*.pt` (embeddings) and `all_smiles_sparse_*.npz` (Tanimoto matrix).

####  Step 3: Start Training
You are ready to train your model. Training is configured using .yaml files located in the `configs/` directory.

    protobind-train -o ./experiment_dir --exp_dir_prefix experiment_name --yaml ./configs/masked_diffusion.yaml


This command uses the following arguments:

- --yaml: Specifies the main model and data configuration file.

- --output_dir: Defines the parent directory where all experiment results will be saved.

- --exp_dir_prefix: Creates a specific folder for this run (e.g., ./experiments/my_first_run).


To tune the model, you can edit the parameters in configs/masked_diffusion.yaml. Details on all tunable hyperparameters can be found in `protobind_diff/parsers.py`.

## Citations 
If you use this code or the models in your research, please cite the following paper:

```bibtex
@article {Mistryukova2025.06.16.659955,
	author = {Mistryukova, Lukia and Manuilov, Vladimir and Avchaciov, Konstantin and Fedichev, Peter O.},
	title = {ProtoBind-Diff: A Structure-Free Diffusion Language Model for Protein Sequence-Conditioned Ligand Design},
	year = {2025},
	journal = {bioRxiv}
}
```

## License 
The code and model weights are released under CC BY-NC 4.0 license. See the [LICENSE](LICENSE) file for details.

