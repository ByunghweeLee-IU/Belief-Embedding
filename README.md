# Belief embedding project 

This project contains the source codes and the dataset used to generate results of the study "Neural embedding of beliefs reveals the role of relative dissonance in human decision-making"  (https://arxiv.org/abs/2408.07237).

Byunghwee Lee, Rachith Aiyappa, Yong-Yeol Ahn, Haewoon Kwak, Jisun An

Beliefs serve as the foundation for human cognition and decision-making. They guide individuals in deriving meaning from their lives, shaping their behaviors, and forming social connections. Therefore, a model that encapsulates beliefs and their interrelationships is crucial for quantitatively studying the influence of beliefs on our actions. Despite its importance, research on the interplay between human beliefs has often been limited to a small set of beliefs pertaining to specific issues, with a heavy reliance on surveys or experiments. Here, we propose a method for extracting nuanced relations between thousands of beliefs by leveraging large-scale user participation data from an online debate platform and mapping these beliefs to an embedding space using a fine-tuned large language model (LLM). This belief embedding space effectively encapsulates the interconnectedness of diverse beliefs as well as polarization across various social issues. We discover that the positions within this belief space predict new beliefs of individuals. Furthermore, we find that the relative distance between one's existing beliefs and new beliefs can serve as a quantitative estimate of cognitive dissonance, allowing us to predict new beliefs. Our study highlights how modern LLMs, when combined with collective online records of human beliefs, can offer insights into the fundamental principles that govern human belief formation and decision-making processes.

## Notes
This repository includes the initial processed dataset, which is essential for generating other results. 
Users can generate further necessary datasets, pre-trained models, and other results using the source codes in this repository.  

## System requirements
* **Software dependencies**:
  * Python (version 3.x)
  * Required libraries:
     * `torch >= 2.2.1`
     * `sentence-transformers >= 2.6.0` 
     * `pandas`
     * `numpy`
* **Pre-trained models**:
  * `roberta-base-nli-stsb-mean-tokens` and `bert-base-uncased` from the `sentence-transformer` package.
* **Operating systems**:
  * The code should run on any operating system that supports Python and the above libraries, including:
    *  Ubuntu 20.04.6 LTS
    *  macOS
    *  Windows 10
* **Versions tested on** 
   * The following libraries need to be installed and are typically compatible with Python 3.8 or higher:
     * `torch >= 2.2.1`
     * `sentence-transformers >= 2.6.0`
     * `pandas >= 2.2.2`
     * `numpy >= 1.24.3`
     * `seaborn >= 0.13`
     * `matplotlib >= 3.8`
* **Hardware requirements**
  * A GPU is recommended for faster training (CUDA-capable NVIDIA GPU recommended).
  * GPU usage: NVIDIA A100 80GB PCIe
