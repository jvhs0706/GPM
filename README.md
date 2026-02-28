# GPM: The Gaussian Pancake Mechanism for Planting Undetectable Backdoors in Differential Privacy

Welcome to the official implementation of [GPM](https://arxiv.org/abs/2509.23834).

## Reproduction Instructions

Follow the steps below to reproduce the experimental results.

### Step 0: Fork and Star This Repository

This repository is set up to automatically generate raw experimental logs. If you do not fork it, the code may attempt to push logs to the original repository, which could fail or interfere with existing data.

To get started:

1. **Fork** this repository on GitHub (a star is appreciated).
2. **Clone your fork locally**:

```bash
git clone --recurse-submodules <your-fork-url> GPM-reproduction
cd GPM-reproduction
```

### Step 1: Set Up the Environment

Create and activate the Conda environment:

```bash
conda env create --file env-start.yaml && conda activate GPM_ENV
```

### Step 2: Run the Reproduction Scripts

First, grant execution permissions and download the data:

```bash
chmod +x ./*.sh && cd ./mnist && python __init__.py && cd ../cifar10_models && chmod +x *.sh && ./download_weights.sh && cd ../
```

Then run the following script from the repository root directory (where this `README.md` is located):

```bash
./run.sh
```

### Step 3: Generate the Plots

Regenerate all plots:

```bash
./plot.sh
```

### Step 4: Submit Your Reproduction Report

After completing reproduction, [open a new issue using the Reproduction Report template](https://github.com/jvhs0706/GPM/issues/new?template=reproduction-report.md). Proper attribution will be provided for all submissions.
