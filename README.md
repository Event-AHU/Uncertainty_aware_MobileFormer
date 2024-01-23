# Uncertainty_aware_MobileFormer
**Uncertainty-aware Bridge based Mobile-Former Network for Event-based Pattern Recognition**, Haoxiang Yang, Chengguo Yuan, Yabin Zhu, Lan Chen, Xiao Wang, Jin Tang 
[[Paper](https://arxiv.org/abs/2401.11123)]

## Overview
![image](https://github.com/Event-AHU/Uncertainty_aware_MobileFormer/blob/main/IMG/Overview.png)

An overview of our proposed uncertain-aware bridge based Mobile-Former framework for event-based action recognition. Given the event streams, we first adopt a StemNet to get the feature embeddings. Then, a MobileNet is proposed to learn the local feature representations and a Transformer branch is adopted to capture the long-range relations. The input of the Transformer branch is random initialized tokens. More importantly, these two branches focus on different types of feature learning, and the information from different samples or the same sample at different time steps may be asymmetrical. The decision of which branch should transmit richer information to the other branch carries a certain level of uncertainty. To address this issue, we design a novel uncertain-aware bridge module to control the information propagation between the dual branches.
 
## Result on ASL-DVS,N-Caltech101 and DVS128-Gait-Day

![image](https://github.com/Event-AHU/Uncertainty_aware_MobileFormer/blob/main/IMG/ASL_DVS_result.png)

![image](https://github.com/Event-AHU/Uncertainty_aware_MobileFormer/blob/main/IMG/N-Caltech101_result.png)

![image](https://github.com/Event-AHU/Uncertainty_aware_MobileFormer/blob/main/IMG/DVS128_Gait-Day_result.png)

## Installation

- clone this repository

```shell
git clone https://github.com/Event-AHU/Uncertainty_aware_MobileFormer.git
```

- Install the dependencies


## datasets

- Ncaltech101: [download link](https://1drv.ms/f/c/9168ed6fce3e99fd/EvPOD9f7LjNNo0QFSQv_5BkBsmKcl6nUnsa1MZEdzICIZA?e=MOc70f)
