# Expert Guided Policy Optimization (EGPO)

Official implementation of CoRL 2021 paper: Safe Driving via Expert Guided Policy Optimization.

[**Webpage**](https://decisionforce.github.io/EGPO) | 
[**Code**](https://github.com/decisionforce/EGPO) | 
[**Video**](https://www.youtube.com/embed/mu2WO--B5C8) | 
[**Poster**](https://decisionforce.github.io/EGPO/images/egpo_poster.png) | 
[**Paper**](https://arxiv.org/pdf/2110.06831.pdf)

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

## Reference

```latex
@inproceedings{peng2021safe,
  title={Safe Driving via Expert Guided Policy Optimization},
  author={Peng, Zhenghao and Li, Quanyi and Liu, Chunxiao and Zhou, Bolei},
  booktitle={5th Annual Conference on Robot Learning},
  year={2021}
}
```

