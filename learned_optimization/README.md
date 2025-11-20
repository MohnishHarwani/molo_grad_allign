# Update

This is a bumped up fork of `learned_optimization` library which works with latest jax and python 3.11. Also uses wandb in [summary writer](learned_optimization/summary.py) for logging instead of tensorboard.

Install commands (tested on RTX 8000 with cuda 12.8):
```bash
# create new conda env
conda create -n glo python=3.11

# install pip packages
pip install "jax[cuda12]==0.6.2"
pip install dm-haiku==0.0.14
pip install optax==0.2.9
pip install tqdm==4.67.1
pip install absl-py==2.2.2
pip install --no-cache-dir tensorflow[and-cuda]
pip install flax==0.10.6
pip install tensorflow-datasets==4.9.8
pip install gin-config==0.5.0
pip install pandas==2.2.3
pip install seqio==0.0.19
pip install git+https://github.com/amoudgl/learned_optimization.git
pip install nvidia-cudnn-cu12==9.8.0.87
pip install tensorflow-probability==0.24.0
pip install tf-keras==2.19.0
pip install wandb==0.21.0
```

To enable logging of intermediate values in jit functions, install `oryx` from a latest git commit since its pip version (v0.2.9) was last updated in Dec 2024 and hence is not compatible with jax at the time of writing (v0.6.2, July 2025):
```
pip install git+https://github.com/jax-ml/oryx.git@2619298bbda423ffb0923d69acaeb1cccd7d7e44
```

The code runs even if you don't install oryx except that intermediate values will not logged by the summary writer.

Original package README below:

---

# learned\_optimization: Meta-learning optimizers and more with JAX

[![Documentation Status](https://readthedocs.org/projects/learned-optimization/badge/?version=latest)](https://learned-optimization.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

*learned\_optimization* is a research codebase for training, designing, evaluating, and applying learned
optimizers, and for meta-training of dynamical systems more broadly. It implements hand-designed and learned optimizers, tasks to meta-train and meta-test them, and outer-training algorithms such as ES, PES, and truncated backprop through time.

To get started see our [documentation](https://learned-optimization.readthedocs.io/en/latest/).

## Quick Start Colab Notebooks
Our [documentation](https://learned-optimization.readthedocs.io/en/latest/) can also be run as colab notebooks! We recommend running these notebooks with a free accelerator (TPU or GPU) in colab (go to `Runtime` -> `Change runtime type`).

### *learned\_optimization* tutorial sequence

1. Introduction : <a href="https://colab.research.google.com/github/google/learned_optimization/blob/main/docs/notebooks/Part1_Introduction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
2. Creating custom tasks: <a href="https://colab.research.google.com/github/google/learned_optimization/blob/main/docs/notebooks/Part2_CustomTasks.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
3. Truncated Steps: <a href="https://colab.research.google.com/github/google/learned_optimization/blob/main/docs/notebooks/Part3_Truncation_TruncatedStep.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
4. Gradient estimators: <a href="https://colab.research.google.com/github/google/learned_optimization/blob/main/docs/notebooks/Part4_GradientEstimators.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
5. Meta training: <a href="https://colab.research.google.com/github/google/learned_optimization/blob/main/docs/notebooks/Part5_Meta_training_with_GradientLearner.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
9. Custom learned optimizers: <a href="https://colab.research.google.com/github/google/learned_optimization/blob/main/docs/notebooks/Part6_custom_learned_optimizers.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### Build a learned optimizer from scratch

Simple, self-contained, learned optimizer example that does not depend on the *learned\_optimization* library:
<a href="https://colab.research.google.com/github/google/learned_optimization/blob/main/docs/notebooks/no_dependency_learned_optimizer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## Local Installation
We strongly recommend using virtualenv to work with this package.

```
pip3 install virtualenv
git clone git@github.com:google/learned_optimization.git
cd learned_optimization
python3 -m venv env
source env/bin/activate
pip install -e .
```

## Train a learned optimizer example
To train a learned optimizer on a simple inner-problem, run the following:

`python3 -m learned_optimization.examples.simple_lopt_train --train_log_dir=/tmp/logs_folder --alsologtostderr`

This will first use tfds to download data, then start running. After a few minutes you should see numbers printed.

A tensorboard can be pointed at this directory for visualization of results. Note this will run very slowly without an accelerator.

## Need help? Have a question?
File a github issue! We will do our best to respond promptly.

## Publications which use *learned\_optimization*
Wrote a paper or blog post that uses *learned\_optimization*? Add it to the list!

* Vicol, Paul, Luke Metz, and Jascha Sohl-Dickstein. ["Unbiased gradient estimation in unrolled computation graphs with persistent evolution strategies."](https://arxiv.org/abs/2112.13835) International Conference on Machine Learning (Best paper award). PMLR, 2021.
* Metz, Luke*, C. Daniel Freeman*, Samuel S. Schoenholz, and Tal Kachman. ["Gradients are Not All You Need."](https://arxiv.org/abs/2111.05803) arXiv preprint arXiv:2111.05803 (2021).

## Development / Running tests
We locate test files next to the related source as opposed to in a separate `tests/` folder.
Each test can be run directly, or with pytest (e.g. `python3 -m pytest learned_optimization/outer_trainers/`). Pytest can also be used to run all tests with `python3 -m pytest`, but this will take quite some time.

If something is broken please file an issue and we will take a look!

## Citing *learned\_optimization*

To cite this repository:

```
@inproceedings{metz2022practical,
  title={Practical tradeoffs between memory, compute, and performance in learned optimizers},
  author={Metz, Luke and Freeman, C Daniel and Harrison, James and Maheswaranathan, Niru and Sohl-Dickstein, Jascha},
  booktitle = {Conference on Lifelong Learning Agents (CoLLAs)},
  year = {2022},
  url = {http://github.com/google/learned_optimization},
}
```

## Disclaimer

*learned\_optimization* is not an official Google product.