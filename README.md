# perfDSA: Automatic Perfusion Imaging in Cerebral Digital Subtraction Angiography

## Description

This repository contains the code release for **[perfDSA](https://link.springer.com/article/10.1007/s11548-025-03359-4)**, a fully automatic cerebral perfusion imaging tool for digital subtraction angiography. You find an overview of the automatic framework below.

<img width="1961" height="1645" alt="image" src="https://github.com/user-attachments/assets/962aab28-53a9-46ae-a38d-3d6cdc1a7721" />

## Quick Start

### 1. Setup

Tested Python version: 3.13

```bash
git clone https://github.com/RuishengSu/perfDSA.git
cd perfDSA
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Usage

Run the script with an input DICOM and optional arguments:

```
python perfDSA.py -i path/to/input.dcm -o path/to/output.png
```

Examples:

- Basic:

```
python perfDSA.py -i xxxx.dcm -o perfusion_parameters.png
```

- Specify model and device:

```
python perfDSA.py -i data/example.dcm -o out.png -m models/best_model_ica_top.pt -d cpu
```

- Set frame interval (milliseconds):

```
python perfDSA.py -i data/example.dcm -f 250
```

Options:

- `-i`: Input DICOM file to be processed (required).
- `-o`: Output image path (default: `./perfusion_parameters.png`).
- `-m`: Path to the ICA top segmentation model (default: `./models/best_model_ica_top.pt`).
- `-d`: Device to run the model on (`cuda` or `cpu`, default: `cuda`).
- `-f`: Desired frame interval in milliseconds for the uniformly-timed sequence (default: `250`).

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

@article{su2023towards,
  title={Towards quantitative digital subtraction perfusion angiography: an animal study},
  author={Su, Ruisheng and van der Sluijs, P Matthijs and Bobi, Joaquim and Taha, Aladdin and van Beusekom, Heleen MM and van der Lugt, Aad and Niessen, Wiro J and Ruijters, Danny and van Walsum, Theo},
  journal={Medical Physics},
  volume={50},
  number={7},
  pages={4055--4066},
  year={2023},
  publisher={Wiley Online Library}
}
```
## Acknowledgement
Many thanks to Fer Fadstake for helping improve the code.

## Contact

Feel free to contact us with questions or for collaboration.

- Name: Ruisheng Su
- Email: r.su@tue.nl
