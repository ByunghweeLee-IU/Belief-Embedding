# Belief embedding project 

This repository contains the source code and dataset used in the study:

- Lee et al. **[A semantic embedding space based on large language models for modelling human beliefs](https://www.nature.com/articles/s41562-025-02228-z)**, *Nature Human Behaviour* (2025) (https://doi.org/10.1038/s41562-025-02228-z). 

- arXiv preprint also available at: [https://arxiv.org/abs/2408.07237](https://arxiv.org/abs/2408.07237)

**Authors**:  
Byunghwee Lee<sup>1</sup>, Rachith Aiyappa<sup>1</sup>, Yong-Yeol Ahn<sup>1</sup>, Haewoon Kwak<sup>1</sup>, Jisun An<sup>1</sup>  

<sup>1</sup> <sub>Center for Complex Networks and Systems Research, Luddy School of Informatics, Computing, and Engineering, Indiana University, Bloomington, Indiana, USA</sub>



## Introduction
 - This repository provides the source code necessary for reproducing the results presented in the paper.  
- The core implementation of experimental results is found in **`src/Main_results.ipynb`**.  
- Both the **raw Debate.org dataset** and the **pre-processed dataset** used in this study can be downloaded from [Figshare](https://figshare.com/articles/dataset/Dataset_for_Neural_embedding_of_beliefs_reveals_the_role_of_relative_dissonance_in_human_decision-making_/28327019).  
- The original Debate.org dataset was obtained from [https://esdurmus.github.io/ddo.html](https://esdurmus.github.io/ddo.html).


## Installation

Installation using [Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html):

```bash
git clone https://github.com/ByunghweeLee-IU/Belief-Embedding.git
cd Belief-Embedding
conda create -y --name belief python=3.8
conda activate belief
pip install -r requirements.txt
```

## System requirements
* **Software dependencies**:
  * Supported platforms: MacOS and Ubuntu (with Python 3.8)
  * See requirements.txt for a complete list of necessary libraries.

* **Tested Versions** 
   * The following libraries have been tested with Python 3.8 or higher:
     * `torch = 2.2.1`
     * `sentence-transformers = 2.6.0`
     * `pandas = 2.2.2`
     * `numpy = 1.24.3`
     * See `requirements.txt` for full list of necessary libraries. 


* **Quickstart**

  * In terminal, run 
  ```bash 
  jupyter notebook
  ```
  * Select `belief` kernel in the jupyter Notebook.
  * Open `Main_result.ipynb` 

* **Hardware requirements**
  * A GPU is recommended for faster training 
  * All experiments were conducted using an **Nvidia A100 80GB PCIe** GPU.




## Estimated time of fine-tuning
 * Fine-tuning a single dataset for one epoch takes approximately 70 minutes on an Nvidia A100 GPU.

 * This study employs 5-fold cross-validation, training five distinct models, each for three epochs. Consequently, the total estimated training time ranges from 15 to 20 hours. 
 
 * To simplify experimentation, Main_results.ipynb provides a convenient way to download a **fine-tuned S-BERT model** from Hugging Face, enabling users to obtain core results without performing the full fine-tuning process.


## Using the Fine-tuned belief model
Below is a simple example of how to load the fine-tuned model from Hugging Face:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("Byunghwee/roberta_belief_finetuned")
```


## License
  * This project is licensed under the **MIT License** - see the `LICENSE` file for details.
