
Littletitan is a project built upon [torchtitan](https://github.com/pytorch/torchtitan), with a focus on MoE pretraining.


## Quick Start

Clone the repo and install the dependencies.
```bash
git clone --recurse-submodules https://github.com/xffxff/littletitan.git

cd third_party/torchtitan
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124 --force-reinstall
pip install -e .
```

Run the training script.
```bash
NGPU=2 bash train.sh
```
