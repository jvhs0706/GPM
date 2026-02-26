# GPM: The Gaussian Pancake Mechanism for Planting Undetectable Backdoors in Differential Privacy

Welcome to the official implementation of [GPM](https://arxiv.org/abs/2509.23834).

## Reproduction Instructions

Follow the steps below to reproduce the experimental results.

### Step 0: Fork and Star This Repository

This repository is designed to automatically generate raw experimental logs. If you do not fork the repository, the code may attempt to push logs to the original repository, which could fail or interfere with existing data.

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

First, grant execution permissions:

```bash
chmod +x ./*.sh
```

Then execute the following scripts from the repository root directory (where this `README.md` is located):

```bash
./hist-da.sh
./mnist-da-new.sh
./cifar10-da.sh
./gm.sh
./hist-rt.sh
```

For systems using **SLURM**, submit the scripts as batch jobs instead:

```bash
sbatch hist-da.sh
sbatch mnist-da-new.sh
sbatch cifar10-da.sh
sbatch gm.sh
sbatch hist-rt.sh
```

### Step 3: Generate the Plots

Regenerate all plots:

```bash
./plot.sh
```

### Step 4: Submit Your Reproduction Report

After completing reproduction, [open a new issue using the Reproduction Report template](https://github.com/jvhs0706/GPM/issues/new?template=reproduction-report.md). Proper attribution will be provided for all submissions.
