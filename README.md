
[![arXiv](https://img.shields.io/badge/arXiv-2503.15111-b31b1b.svg)](https://arxiv.org/abs/2503.15111)

**This is the official implementation of the ICLR 2025 paper "[FedLWS: Federated Learning with Adaptive Layer-wise Weight Shrinking](https://openreview.net/pdf?id=6RjQ54M1rM)".**

## FedLWS
You can run FedLWS with the following command:

```
python main.py --dataset cifar10 --local_model ResNet20 --server_method fedlws --client_method local_train #FedLWS on CIFAR-100 dataset with ResNet20 model
```



## Citing This Repository

Please cite our paper if you find this repo useful in your work:

```
@inproceedings{
shi2025fedlws,
title={Fed{LWS}: Federated Learning with Adaptive Layer-wise Weight Shrinking},
author={Changlong Shi and Jinmeng Li and He Zhao and Dan dan Guo and Yi Chang},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=6RjQ54M1rM}
}
```


We would like to thank the authors for releasing the public repository: [ICML-2023-FedLAW](https://github.com/ZexiLee/ICML-2023-FedLAW/tree/main)
