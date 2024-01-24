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

- install the virtual environment and pytorch:
   ```
  conda create --name env_name python=3.7
  source activate env_name
  pip install torch==1.7.1+cu110 torchvision=0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
   pip install torch_cluster==1.5.8
   pip install torch_scatter==2.0.7
   pip install torch_sparse==0.6.9
   pip install torch_spline_conv==1.2.0
   pip install torch-geometric
   pip install -r requirements.txt
  ```


## datasets

- Ncaltech101: [download link](https://1drv.ms/f/c/9168ed6fce3e99fd/EvPOD9f7LjNNo0QFSQv_5BkBsmKcl6nUnsa1MZEdzICIZA?e=B3daNo)
-  please place the data folder based on the following structure:
    ```
	MobileFormer_3D
	├── data
	│   ├── Ncaltech101
	│   │   │── rawframes
	│   │   │   │── accordion
	│   │   │   │── ....
        │   │   │── Ncal_train.txt
        │   │   │── Ncal_test.txt
	├── datasets
	```

## Training
 ```
  #--root: database path
  bash train.sh
  ```
The time cost for an epoch is around 8 minutes
