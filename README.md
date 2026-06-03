# perfDSA: Automatic Perfusion Imaging in Cerebral Digital Subtraction Angiography

## Description

This repository contains the code release for **[perfDSA](https://link.springer.com/article/10.1007/s11548-025-03359-4)**, a fully automatic cerebral perfusion imaging tool for digital subtraction angiography. You find an overview of the automatic framework below.

<img width="1961" height="1645" alt="image" src="https://github.com/user-attachments/assets/962aab28-53a9-46ae-a38d-3d6cdc1a7721" />

## Quick Start

### 1. Setup

```bash
git clone https://github.com/RuishengSu/perfDSA.git
cd perfDSA
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run example

Run:

```
python example.py
```

This assumes there's a DICOM file example.dicom in the root folder. Output is stored as perfusion_parameters.png.

## Project Structure

```text
perfDSA/
|-- data/                       # Folder for DSA training data
|-- models/                     # Trained model weights for segmentation of ICA top for AIF extraction
|-- unet/                       # UNet Network architecture
|-- utils/                      # Utility functions
|-- scripts/                    # Scripts for visualization and analysis
```

## Citation

Please cite our paper if you find it useful.

```bibtex
@article{su2025perfdsa,
  title={perfDSA: Automatic perfusion imaging in cerebral digital subtraction angiography},
  author={Su, Ruisheng and van der Sluijs, P Matthijs and Marc, Flavius-Gabriel and Te Nijenhuis, Frank and Cornelissen, Sandra AP and Roozenbeek, Bob and van Zwam, Wim H and van der Lugt, Aad and Ruijters, Danny and Pluim, Josien and others},
  journal={International journal of computer assisted radiology and surgery},
  volume={20},
  number={6},
  pages={1195--1203},
  year={2025},
  publisher={Springer}
}
```

## Contact

Feel free to contact us with questions or for collaboration.

- Name: Ruisheng Su
- Email: r.su@tue.nl
