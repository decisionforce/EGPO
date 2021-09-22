# Expert Guided Policy Optimization (EGPO)

Official implementation of CoRL 2021 paper: Safe Driving via Expert Guided Policy Optimization.

[**Webpage**](https://decisionforce.github.io/EGPO) | [**Paper (OpenReview)**](https://openreview.net/pdf?id=KnOYrZf17CQ)

## Installation

```bash
# Clone the code to local
git clone https://github.com/decisionforce/EGPO.git
cd EGPO

# Create virtual environment
conda create -n egpo python=3.7
conda activate egpo

# Install basic dependency
pip install -e .

# Now you can run the training script of EGPO.
# If you wish to run other baselines, some extra environmental
# setting is required as follows:

# To run CQL/BC, ray needs to be updated to 1.2.0
pip install ray==1.2.0

# To run GAIL/DAgger, please install GPU-version of torch:
conda install pytorch==1.5.0 torchvision==0.6.0 -c pytorch
conda install condatoolkit==9.2
```

## Training

```bash
cd EGPO/training_script/
python train_egpo.py
```

You can also run other baselines by running the training scripts directly.

## Citation

This part is working in progress!



