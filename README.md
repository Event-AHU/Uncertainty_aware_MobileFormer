# Uncertainty_aware_MobileFormer
Uncertainty-aware Bridge based Mobile-Former Network for Event-based Pattern Recognition

# Overview
![image](https://github.com/Event-AHU/Uncertainty_aware_MobileFormer/blob/main/IMG/Overview.png)

An overview of our proposed uncertain-aware bridge based Mobile-Former framework for event-based action recognition. Given the event streams, we first adopt a StemNet to get the feature embeddings. Then, a MobileNet is proposed to learn the local feature representations and a Transformer branch is adopted to capture the long-range relations. The input of the Transformer branch is random initialized tokens. More importantly, these two branches focus on different types of feature learning, and the information from different samples or the same sample at different time steps may be asymmetrical. The decision of which branch should transmit richer information to the other branch carries a certain level of uncertainty. To address this issue, we design a novel uncertain-aware bridge module to control the information propagation between the dual branches.
 
