# AD-DRL

This is the Pytorch implementation for our ACM MM 2024 paper:

>ACM MM 2024. Zhenyang Li, Fan Liu, Yinwei Wei, Zhiyong Cheng, Liqiang Nie, Mohan Kankanhalli(2024). Attribute-driven Disentangled Representation Learning for Multimodal Recommendation, [Paper in arXiv](https://arxiv.org/abs/2312.14433).

Author: Dr. Zhenyang Li 

## Introduction

Many Recommendation methods focus on learning robust and independent representations by disentangling the intricate factors within interaction data across various modalities in an unsupervised manner. However, such an approach obfuscates the discernment of how specific factors (e.g., category or brand) influence the outcomes, making it challenging to regulate their effects. In response to this challenge, we introduce a novel method called Attribute-Driven Disentangled Representation Learning (short for AD-DRL), which explicitly incorporates attributes from different modalities into the disentangled representation learning process. By assigning a specific attribute to each factor in multimodal features, AD-DRL can disentangle the factors at both attribute and attribute-value levels. To obtain robust and independent representations for each factor associated with a specific attribute, we first disentangle the representations of features both within and across different modalities. Moreover, we further enhance the robustness of the representations by fusing the multimodal features of the same factor. Empirical evaluations conducted on three public real-world datasets substantiate the effectiveness of AD-DRL, as well as its interpretability and controllability.


## Enviroment Requirement

`pip install -r requirements.txt`


## Dataset

We provide three processed datasets: Amazon-Baby, Amazon-Sports and Amazon-ToysGames.

* The dataset is released at [Google Drive](https://drive.google.com/drive/folders/18LRHDZhcX2KYJ-f_ThZdQgRmKls8WvUH?usp=sharing).
* Please download the three datasets and place them in the `AD-DRL/AmazonData` folder.
* see more in `amazon.py`

## Checkpoints

We provide checkpoints on three datasets. Please download them from [Google Drive](https://drive.google.com/drive/folders/11JM0Iw3dsy_vUAVGKjMgqyA6VJCtNhmi?usp=sharing) and place them in the `AD-DRL/checkpoints` folder.

## Model Training

Train ADRRL on **Amazon-Baby** dataset:

` python main.py --mode "train" --dataset "Baby" --attribute_dataset "item_attribute_label" --learning_rate 0.0001 --decay_r 0 --decay_f 1 --decay_a 50 --decay_n 1 --temp 1 --num_neg 4 --gpu "0" --n_factors 4 --emb_dim 128`

Train ADRRL on **Amazon-Sports** dataset:

`python main.py --mode "train" --dataset "Sports" --attribute_dataset "item_attribute_label" --learning_rate 0.0001 --decay_r 1 --decay_f 5 --decay_a 10 --decay_n 0.01 --temp 1 --num_neg 8 --gpu "0" --n_factors 5 --emb_dim 160`

Evaluate ADRRL on **Amazon-Sports** dataset:

`python main.py --mode "train" --dataset "ToysGames" --attribute_dataset "item_attribute_label" --learning_rate 0.0001 --decay_r 1 --decay_f 1 --decay_a 10 --decay_n 0.01 --temp 1 --num_neg 8 --gpu "0" --n_factors 5 --emb_dim 160`

## Model Evaluation

Evaluate ADRRL on **Amazon-Baby** dataset:

` python main.py --mode "test" --dataset "Baby" --attribute_dataset "item_attribute_label" --num_neg 4 --gpu "0" --n_factors 4 --emb_dim 128`

Evaluate ADRRL on **Amazon-Sports** dataset:

`python main.py --mode "test" --dataset "Sports" --attribute_dataset "item_attribute_label" --num_neg 8 --gpu "1" --n_factors 5 --emb_dim 160`

Evaluate ADRRL on **Amazon-Sports** dataset:

`python main.py --mode "test" --dataset "ToysGames" --attribute_dataset "item_attribute_label" --num_neg 8 --gpu "0" --n_factors 5 --emb_dim 160`

## Please consider to cite our paper if our work helps you, thanks:
```
@inproceedings{ADDRL,
  author       = {Zhenyang Li and
                  Fan Liu and
                  Yinwei Wei and
                  Zhiyong Cheng and
                  Liqiang Nie and
                  Mohan S. Kankanhalli},
  title        = {Attribute-driven Disentangled Representation Learning for Multimodal
                  Recommendation},
  booktitle    = {Proceedings of the {ACM} International Conference on Multimedia},
  pages        = {9660--9669},
  publisher    = {{ACM}},
  year         = {2024}
}
```