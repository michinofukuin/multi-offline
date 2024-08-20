# Offline Multi-Agent Reinforcement Learning with Implicit Global-to-Local Value Regularization (NeurIPS 2023)
The official implementation of "[Offline Multi-Agent Reinforcement Learning with Implicit Global-to-Local Value Regularization](https://arxiv.org/abs/2307.11620)". OMIGA provides a principled framework to convert global-level value regularization into equivalent implicit local value regularizations and simultaneously enables in-sample learning, thus elegantly bridging multi-agent value decomposition and policy learning with offline regularizations. This repository is inspired by the [TRPO-in-MARL](https://github.com/cyanrain7/TRPO-in-MARL) library for online Multi-Agent RL.

**This repo provides the implementation of OMIGA in Multi-agent MuJoCo.**

## Installation
``` Bash
conda create -n env_name python=3.9
conda activate OMIGA
git clone https://github.com/ZhengYinan-AIR/OMIGA.git
cd OMIGA-master
pip install -r requirements.txt
```

## How to run
Before running the code, you need to download the necessarpython run_mujoco.py  --scenario="HalfCheetah-v2" --agent_conf="6x1" --data_type="medium-expert"y offline datasets ([Download link](https://cloud.tsinghua.edu.cn/d/dcf588d659214a28a777/)). Then, make sure the config file at [configs/config.py](https://github.com/ZhengYinan-AIR/Offline-MARL/blob/master/configs/config.py) is correct. Set the **data_dir** parameter as the storage location for the downloaded data, and configure parameters **scenario**, **agent_conf**, and **data_type**. You can run the code as follows:
``` Bash
# If the location of the dataset is at: "/data/Ant-v2-2x4-expert.hdf5"
cd OMIGA
python run_mujoco.py  --scenario="Ant-v2" --agent_conf="2x4" --data_type="expert"
python run_mujoco.py  --scenario="Ant-v2" --agent_conf="2x4" --data_type="medium"
python run_mujoco.py  --scenario="Ant-v2" --agent_conf="2x4" --data_type="medium-expert"
python run_mujoco.py --scenario="Hopper-v2" --agent_conf="3x1" --data_type="medium"
python run_mujoco.py --scenario="Hopper-v2" --agent_conf="3x1" --data_type="expert"
python run_mujoco.py --scenario="HalfCheetah-v2" --agent_conf="6x1" --data_type="expert"
python run_mujoco.py --scenario="HalfCheetah-v2" --agent_conf="6x1" --data_type="medium"
python run_mujoco.py --scenario="HalfCheetah-v2" --agent_conf="6x1" --data_type="medium-expert"
kill -SIGCONT 2946802
tmux attach-session -t 3

kill -STOP 147733
#seed=0,1,2,42,3407
/mnt/data/chenhaosheng/OMIGA-master/data/Hopper-v2-3x1-expert.hdf5
## Weights and Biases Online Visualization Integration
This codebase can also log to [W&B online visualization platform](https://wandb.ai/site). To log to W&B, you first need to set your W&B API key environment variable:
```
wandb online
export WANDB_API_KEY='YOUR W&B API KEY HERE'
```
Then you can run experiments with W&B logging turned on:
```
python run_mujoco.py --wandb=True
```


## Bibtex
If you find our code and paper can help, please cite our paper as:
```
@article{wang2023offline,
  title={Offline Multi-Agent Reinforcement Learning with Implicit Global-to-Local Value Regularization},
  author={Wang, Xiangsen and Xu, Haoran and Zheng, Yinan and Zhan, Xianyuan},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```
