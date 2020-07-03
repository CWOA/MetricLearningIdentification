# Identification via Metric Learning

This repository contains the source code that accompanies our paper "Visual Identification of Individual Holstein Friesian Cattle via Deep Metric Learning" - available at [https://arxiv.org/abs/2006.09205](https://arxiv.org/abs/2006.09205).
At its core, the code in this repository is adapted and extended (with permission) from Lagunes-Fortiz, M. et al's work on "Learning Discriminative Embeddings for Object Recognition on-the-fly" published in ICRA 2019 - [paper](https://ieeexplore.ieee.org/document/8793715), [source code](https://github.com/MikeLagunes/Supervised-Triplet-Network).

Within our paper, the code in this repository relates to section 5 on the "Open-Set Individual Identification via Metric Learning" and the experiments conducted in section 6.
A selective set of the highest-performing weights from the experiments on open-set identification are included in this repository at `weights`.

### Installation

Simply clone this repository to your desired local directory: `git clone https://github.com/CWOA/MetricLearningIdentification.git` and
install any missing requirements via `pip` or `conda`: [numpy](https://pypi.org/project/numpy/), [PyTorch](https://pytorch.org/), [tqdm](https://pypi.org/project/tqdm/), [sklearn](https://pypi.org/project/scikit-learn/)

### Usage

To replicate the results obtained in our paper, please download the OpenCows2020 dataset at: [https://www.data.bris.ac.uk/data](https://www.data.bris.ac.uk/data) and searching for `OpenCows2020`.
Place the identification folder in `datasets/OpenCows2020/`.
A selective set of weights from the paper are included in the `weights` folder.

To train the model, use `python train.py -h` to get help with setting command line arguments. To train on your own dataset, write your own dataset class for managing loading the data (similarly to `datasets/OpenCows2020/`), import it into `utilities/utils.py` and add the case to the `def selectDataset(args)` method.

To test a trained model, use `python test.py -h` to get help with setting command line arguments.

To visualise inferred embeddings using T-SNE, use `python utilities/visualse_embeddings.py -h` to get help with setting relevant command line arguments.

### Citation

Consider citing ours and Miguel's works in your own research if this repository has been useful:
```
@article{andrew2020visual,
  title={Visual Identification of Individual Holstein Friesian Cattle via Deep Metric Learning},
  author={Andrew, William and Gao, Jing and Campbell, Neill and Dowsey, Andrew W and Burghardt, Tilo},
  journal={arXiv preprint arXiv:2006.09205},
  year={2020}
}

@inproceedings{lagunes2019learning,
  title={Learning discriminative embeddings for object recognition on-the-fly},
  author={Lagunes-Fortiz, Miguel and Damen, Dima and Mayol-Cuevas, Walterio},
  booktitle={2019 International Conference on Robotics and Automation (ICRA)},
  pages={2932--2938},
  year={2019},
  organization={IEEE}
}
```
