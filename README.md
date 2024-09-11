# Belief embedding project 

This project contains the source codes and the dataset used to generate results of the study "Neural embedding of beliefs reveals the role of relative dissonance in human decision-making"  (https://arxiv.org/abs/2408.07237).

Authors: Byunghwee Lee<sup>1</sup>, Rachith Aiyappa<sup>1</sup>, Yong-Yeol Ahn<sup>1</sup>, Haewoon Kwak<sup>1</sup>, Jisun An<sup>1</sup>

<sup>1</sup> <sub>Center for Complex Networks and Systems Research, Luddy School of Informatics, Computing, and Engineering, Indiana University, Bloomington, Indiana, USA, 47408</sub>

## Introduction
This repository contains the source code for the analysis. Users can process datasets, fine-tune pre-trained models, and generate main results using these codes.

## Installation

Installation using [Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html):

```bash
git clone https://github.com/ByunghweeLee-IU/Belief-Embedding.git
cd Belief-Embedding
conda create -y --name belief python=3.8
conda activate belief
pip install -r requirements.txt
python -m ipykernel install --name belief
```

## System requirements
* **Software dependencies**:
  * Supported platforms: MacOS and Ubuntu (with Python 3.8)

* **Versions tested on** 
   * The following libraries need to be installed and are typically compatible with Python 3.8 or higher:
     * `torch = 2.2.1`
     * `sentence-transformers = 2.6.0`
     * `pandas = 2.2.2`
     * `numpy = 1.24.3`
     * See `requirements.txt` for full list of necessary libraries. 
     
* **Hardware requirements**
  * A GPU is recommended for faster training 
  * Our analysis was run with GPU NVIDIA A100 80GB PCIe

* **License**
  * This project is licensed under the MIT License - see the LICENSE file for details.

## Quickstart

```bash
jupyter notebook
```

Select `belief` kernel in the jupyter Notebook.
Open `Main_result.ipynb` to view the aggregated analysis process.