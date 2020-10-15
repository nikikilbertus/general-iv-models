# A class of algorithms for general instrumental variable models

This is the Python code accompanying the paper

[A Class of General Instrumental Variable Models](https://arxiv.org/abs/2006.06366)\
[Niki Kilbertus](https://sites.google.com/view/nikikilbertus), [Matt J. Kusner](https://mkusner.github.io/), [Ricardo Silva](http://www.homepages.ucl.ac.uk/~ucgtrbd/)\
Neural Information Processing Systems (NeurIPS) 2020

## Setup

First clone this repository and navigate to the main directory

```sh
git clone git@github.com:nikikilbertus/general-iv-models.git
cd general-iv-models
```

To run the code, please first create a new Python3 environment (Python version >= 3.6 should work).
For example, if you are using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/) run

```sh
mktmpenv -n
```

Then install the required packages into your newly created environment via

```sh
python -m pip install -r requirements.txt
```

## Run experiments

There are three executable scripts in the `general-iv-models` directory to run different subsets of the experiments in the paper:

* `linear_experiments.sh`
* `non_linear_experiments.sh`
* `sigmoid_design.sh`

Running any of them will create a directory `general-iv-models/results` where all results (plots) will be stored.

To run all experiments, simply run

```sh
./linear_experiments.sh
./non_linear_experiments.sh
./sigmoid_design.sh
```