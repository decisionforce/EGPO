# expert-guided-policy-optimization
Official implementation of CoRL 2021 paper: Safe Driving via Expert Guided Policy Optimization.

## Installation
```
# Create virtual environment
conda create -n egpo python=3.7
conda activate egpo

# Install basic dependency
conda install pytorch==1.5.0 torchvision==0.6.0 -c pytorch
conda install condatoolkit==9.2
pip install -e .

# for CQL/BC, ray needs to be updated to 1.2.0
pip install ray==1.2.0
```


