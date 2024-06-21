# PyGenomics Project Setup

## Setup Environment and Jupyter Notebook Kernel

### Step 1: Create Conda Environment

First, create the `pygenomics` environment from the provided YAML file.

```bash
cd pygenomics
conda env create -f ./pygenomics.yml
```

### Step 2: Activate Conda Environment

Activate the pygenomics environment.

```bash
conda activate pygenomics
```

### Step 3: Install Jupyter Kernel

Install the environment as a Jupyter Notebook kernel.

```bash
pip install ipykernel
python -m ipykernel install --user --name pygenomics --display-name "PyGenomics"
```

### Step 4: Verify Installation

Verify that the kernel is installed correctly.

```bash
jupyter kernelspec list
```

### Step 5: Start Jupyter Notebook

Start Jupyter Notebook to begin working with PyGenomics.

```bash
jupyter notebook
```