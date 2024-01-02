# COMMA: Co-Articulated Multi-Modal Learning [AAAI 2024]

Official implementation of the paper "[COMMA: Co-Articulated Multi-Modal Learning](https://arxiv.org/abs/2401.00268)".
<hr />

## Main Contributions

1) **Correlated prompt generation:** The prompts of the vision and language branches in these methods are usually separated or uni-directionally correlated. To better guide and align the representations of two branches, we present to compute prompts based on preceding prompts of both branches to aggregate beneficial multi-modal information.
2) **Alleviating Forgetting Generic Knowledge:** The essential generic knowledge learned in the pretraining stage is partly forgotten in the fine-tuning process. We propose to alleviate forgetting generic knowledge by minimizing the feature discrepancy between the learnable prompts and hand-crafted prompts of the pretrained CLIP in the last several layers.


## :ballot_box_with_check: Supported Methods

[comment]: <> (| Language Prompting            | MaPLe |  [link]&#40;configs/trainers/IVLP/vit_b16_c2_ep5_batch4_4ctx_language_only.yaml&#41;      |      |)

| Method                    | Paper                                         |                             Configs                             |          Training Scripts          |
|---------------------------|:----------------------------------------------|:---------------------------------------------------------------:|:----------------------------------:|
| MaPLe                     | [CVPR 2023](https://arxiv.org/abs/2210.03117)                                     | [link](configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml)  |       [link](scripts/maple)        |
| CoOp                      | [IJCV 2022](https://arxiv.org/abs/2109.01134) |                  [link](configs/trainers/CoOp)                  |        [link](scripts/coop)        |
| Co-CoOp                   | [CVPR 2022](https://arxiv.org/abs/2203.05557) |                 [link](configs/trainers/CoCoOp)                 |       [link](scripts/cocoop)       |
| Deep Vision Prompting     | -                                             |    [link](configs/trainers/VPT/vit_b16_c2_ep5_batch4_4.yaml)    |        [link](scripts/vpt)         |
| Deep Language Prompting   | -                                             |                 [link](configs/trainers/IVLP/vit_b16_c2_ep5_batch4_4ctx_language_only.yaml)                  | [link](scripts/language-prompting) |
| Independent V-L Prompting | -                                             | [link](configs/trainers/IVLP/vit_b16_c2_ep5_batch4_2+2ctx.yaml) |  [link](scripts/independent-vlp)   |
| COMMA (ours) | [AAAI2024]((https://arxiv.org/abs/2401.00268)  )                                             | [link](configs/trainers/COMMA/vit_b16_c2_ep5_batch4_2+2ctx.yaml) |  [link](scripts/comma)   |
<hr />

## Results
### MaPLe in comparison with existing methods
Results reported below show accuracy for base and novel classes for across 11 recognition datasets averaged over 3 seeds.

| Name                                                      | Base Acc. | Novel Acc. |    HM     | Epochs | 
|-----------------------------------------------------------|:---------:|:----------:|:---------:|:------:|
| [CLIP](https://arxiv.org/abs/2103.00020)                  |   69.34   |   74.22    |   71.70   |   -    |  
| [CoOp](https://arxiv.org/abs/2109.01134)                  | **82.69** |   63.22    |   71.66   |  200   | 
| [CoCoOp](https://arxiv.org/abs/2203.05557) |   80.47   |   71.69    |   75.83   |   10   | 
| [KgCoOp](https://arxiv.org/abs/2303.13283) |   80.73   |   73.60    |   77.00   |   10  |  
| [MaPLe ](https://arxiv.org/abs/2210.03117)  |   82.28   | 75.14  | 78.55 |   5    |  
[ COMMA (ours)](https://arxiv.org/abs/2401.00268)  |   82.42   | 75.87  | 79.04 |   5    |  

## Installation 
For installation and other package requirements, please follow the instructions detailed in [INSTALL.md](docs/INSTALL.md). 

## Data preparation
Please follow the instructions at [DATASETS.md](docs/DATASETS.md) to prepare all datasets.


## Training and Evaluation
Please refer to the [RUN.md](docs/RUN.md) for detailed instructions on training and evaluating.

## Acknowledgements

Our code is based on [Co-CoOp/CoOp](https://github.com/KaiyangZhou/CoOp) and [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning) repositories. We thank the authors for releasing their code. 

