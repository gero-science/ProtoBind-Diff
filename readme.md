# ProtoBind-Diff: A Structure-Free Diffusion Language Model for Protein Sequence-Conditioned Ligand Design



Implementation of [ProtoBind-Diff: A Structure-Free Diffusion Language Model for Protein Sequence-Conditioned Ligand Design](https://docsend.com/view/bseub85ffi5yyh92)  by Lukia Mistryukova*, Vladimir Manuilov*, Konstantin Avchaciov*, and Peter O. Fedichev.


ProtoBind-Diff is a masked diffusion language model that generates target-specific, high-quality small-molecule ligands. Trained entirely without structural input, it enables structure-independent ligand design across the full proteome and matches structure-based methods in docking and Boltz-1 benchmarks.


If you have questions, feel free to open an issue or send us an email at lukiia.mistriukova@gero.ai, konstantin.avchaciov@gero.ai, vladimir.manuylov@gero.ai, and peter.fedichev@gero.ai.

![Alt Text](graphical-abstract.png)



## Overview

This repository contains the evaluation toolkit and supporting data for **ProtoBind-Diff**. It is organized as follows:

### `data/`
We selected 12 protein targets to benchmark molecular generation quality across different models.
- One folder per model (`pocket2mol/`, `targetdiff/`, etc.) containing SMILES files of generated ligands
- `bindingdb/` – sets of random molecules (used as reference inactive molecules)
- `bindingdb_active/` – sets of true active molecules 
- `actives_bindingdb_cl` - clustered true actives; one representative per similarity cluster
- `CrossDocked2020/` – cleaned PDB files and corresponding ligands for the targets 
- `fasta/` – FASTA sequences of the targets 

### `notebooks/`
Jupyter notebooks to reproduce the paper figures: docking/Boltz-1 score distributions, interpretation of attention maps, chemical property distributions, and UMAP-based target specifity analysis. We also added `allign.ipynb` to show differences between canonical sequences and PDB receptor seqeunces.
 

### `results/`
- Raw docking and Boltz-1 score tables for each method.

---

A quick-start guide, environment setup instructions, license, and a citation will be added later.
[Subscribe for notification of future updates](https://docs.google.com/forms/d/e/1FAIpQLSdWJkWVT5qZC2Ukplc5Ej7Bxi2a62QeD0I8jFeHqvZIFJVEtA/viewform?usp=preview)

