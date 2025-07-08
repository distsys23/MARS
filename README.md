# Cooperative Multi-Agent Deep Reinforcement Learning for Workflow Scheduling in Hybrid Clouds with Privacy and Security (MARS)
This is an implementation code for our paper entitled MARS: Cooperative Multi-Agent Deep Reinforcement Learning for Workflow Scheduling in Hybrid Clouds with Privacy and Security Constraints in IEEE ICPADS 2024.

## Requirements

see in [requirements.txt](https://github.com/distsys23/MARS/blob/main/requirements.txt)

install packages with `pip install -r requirements.txt`

## Quick Start

```
$ python main.py
```

Directly run the `main.py`, the approach will run with the default settings.

## Data
Workflows data with .xml format located in folder [XML_Scientific_Workflow](https://github.com/distsys23/MARS/tree/main/XML_Scientific_Workflow), including CyberShake, Epigenomics, LIGO, Montage, and SIPHT.

## Code Structure
- `agent.py`: function for building agents.
- `buffer.py`: function for building buffer, where some trained data would be saved.
- `env.py`: codes for creating a multi-agent hybrid cloud environment with VMs.
- `matd3.py`: function for building Multi-Agent Twin Delayed Deep Deterministic Policy Gradient (MATD3) algorithm.
- `networks.py`: function for building neural networks.
- `noise.py`: codes for creating noise.
- `parameter.py`: parameters of the approach.
- `workflow.py`: scientific workflows for scheduling.
- `workflow_preprocess.py`: preprocess of workflow xml files.
- `main.py`: main function.

## Citation
This work is accepted by 2024 IEEE 30th International Conference on Parallel and Distributed Systems (ICPADS).

```bash
@inproceedings{cheng2024mars,
  title={MARS: Multi-Agent Deep Reinforcement Learning for Real-Time Workflow Scheduling in Hybrid Clouds with Privacy Protection},
  author={Cheng, Long and He, Haoyang and Gu, Yan and Liu, Qingzhi and Zhao, Zhiming and Fang, Fang},
  booktitle={2024 IEEE 30th International Conference on Parallel and Distributed Systems (ICPADS)},
  pages={657--666},
  year={2024},
  organization={IEEE}
}
```
