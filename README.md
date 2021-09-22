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
conda install pytorch==1.5.0 torchvision==0.6.0 -c pytorch
conda install condatoolkit==9.2
pip install -e .

# To run CQL/BC, ray needs to be updated to 1.2.0
pip install ray==1.2.0
```

## Training

```bash
cd EGPO/training_script/
python train_egpo.py
```

You can also run other baselines by running the training scripts directly.

## Citation

This part is working in progress!



