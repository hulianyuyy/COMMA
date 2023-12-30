# Training and Evaluation

We provide bash scripts in [scripts/](../scripts) for each prompting variant including COMMA, MaPLe, vision, language and independent V-L prompting.
Make sure to configure the dataset paths in environment variable `DATA` and run the commands from the main directory `multimodal-prompt-learning/`.
Below we provide training and evaluation instructions for COMMA. The same instructions applies for all other variants including *MaPLe, Vision (VPT), Language and independent V-L prompting* .


### Training time and compute
We train COMMA on each dataset with a batch size of 4 using a **single** NVIDIA 3090 GPU.
Training COMMA on ImageNet for 5 epochs takes about 1.5 hours for a single seed. So results for 3 seeds takes around 5 hours. For all remaining 10 datasets, it combinedly takes around 6 hours (for all 3 seeds) on a single 3090 GPU. 

## COMMA

#### (1) Base-to-Novel class generalization setting
The default training settings are provided in config file at `configs/trainers/COMMA/vit_b16_c2_ep5_batch4_2ctx.yaml`. All hyper-parameters such as prompt length, prompt depth, etc., can be modified using this config file.

Below, we provide instructions to train COMMA on imagenet. 


```bash
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

# seed=1
# trains and evaluates on base classes
bash scripts/comma/base2new_train_comma.sh imagenet 1
# evaluates on novel classes
bash scripts/comma/base2new_test_comma.sh imagenet 1

# seed=2
# trains and evaluates on base classes
bash scripts/comma/base2new_train_comma.sh imagenet 2
# evaluates on novel classes
bash scripts/comma/base2new_test_comma.sh imagenet 2

# seed=3
# trains and evaluates on base classes
bash scripts/comma/base2new_train_comma.sh imagenet 3
# evaluates on novel classes
bash scripts/comma/base2new_test_comma.sh imagenet 3
```

You could directly use ``` bash base2new_train.sh``` to train your model, and modify the dataset in the bash file accordingly. 

#### Averaging results over 3 seeds: 
Once the above trainings and evaluations are completed, the `output/` directory should have the following structure:

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– COMMA/
|   |   |   |   |   |–– vit_b16_c2_ep5_batch4_2ctx/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– COMMA/
|   |   |   |   |   |–– vit_b16_c2_ep5_batch4_2ctx/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```

Now use the script `parse_test_res.py` and run the commands below to calculate the averaged results:
```bash
# prints averaged results for base classes
python parse_test_res.py output/base2new/train_base/imagenet/shots_16/COMMA/vit_b16_c2_ep5_batch4_2ctx
# averaged results for novel classes
python parse_test_res.py output/base2new/test_new/imagenet/shots_16/COMMA/vit_b16_c2_ep5_batch4_2ctx --test-log
```

You could directly use ``` bash base2new_test.sh``` to test your model, and modify the dataset in the bash file accordingly. 

The above steps can be repeated for other individual datasets.

For simplicity, you can directly perform training on all dataset by running ``` bash base2new_train_all.sh```, and then evaluate your model on all datasets by running ``` bash base2new_test_all.sh```. You can further run ``` bash base2new_train and_test_all.sh``` to train and evaluate your model with only one execution step.

The results are printed on the screen and stored in the log.txt in the output directory (e.g., "./output/base2new/test_new/shots_16/COMMA/vit_b16_c2_ep5_batch4_2ctx" and "./output/train_base/test_new/shots_16/COMMA/vit_b16_c2_ep5_batch4_2ctx").

#### (2) Cross-Dataset Transfer
We provide instructions to train COMMA on ImageNet using all 1000 classes and then evaluating it directory on new downstream datasets.
We provide cross-dataset config for COMMA: `configs/COMMA/vit_b16_c2_ep5_batch4_2ctx_cross_datasets.yaml`.
* Firstly, train COMMA on imagenet in few-shot manner (for all 3 seeds).

```bash
# seed=1 
bash scripts/comma/xd_train_comma.sh imagenet 1
# seed=2 
bash scripts/comma/xd_train_comma.sh imagenet 2
# seed=3 
bash scripts/comma/xd_train_comma.sh imagenet 3
```

You could directly use ``` bash cross_dataset_train.sh``` to train your model. 

* Now evaluate imageNet model on downstream datasets.

```bash
for SEED in 1 2 3
do
    bash scripts/comma/xd_test_comma.sh caltech101 ${SEED}
    bash scripts/comma/xd_test_comma.sh oxford_pets ${SEED}
    bash scripts/comma/xd_test_comma.sh stanford_cars ${SEED}
done
```
You could directly use ``` bash cross_dataset_test.sh``` to test your model, and then run ``` bash cross_dataset_evaluation.sh``` to view the results on each dataset. 

The results are printed on the screen and stored in the log.txt in the output directory (e.g., "./output/evaluation/COMMA/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots").

#### (3) Domain Generalization 
We use imagenet trained COMMA model for domain generalization experiments. The steps are similar to above cross-dataset experiments, however, model is evaluated on imagenet variants.

You could directly use ``` bash domain_generalization_train.sh``` to train your model, or simply use the model trained under the Cross-Dataset Transfer setting.

* Evaluate imageNet model on variants of imagenet (domain shift datasets).

```bash
for SEED in 1 2 3
do
    bash scripts/comma/xd_test_comma.sh imagenetv2 ${SEED}
    bash scripts/comma/xd_test_comma.sh imagenet_sketch ${SEED}
    bash scripts/comma/xd_test_comma.sh imagenet_a ${SEED}
    bash scripts/comma/xd_test_comma.sh imagenet_r ${SEED}
done
```

You could directly use ``` bash domain_generalization_test.sh``` to test your model, and then run ``` bash domain_generalization_evaluation.sh``` to view the results on each dataset. 

By combining "Cross-Dataset Transfer" and "Domain Generalization" together, you can run ``` bash cross_dataset_and_domain_ge_train.sh```, ``` bash cross_dataset_and_domain_ge_test.sh``` and ``` bash cross_dataset_and_domain_ge_evaluation.sh``` to finish the training, testing and evaluation steps, respectively. 

Further, you can run ``` bash cross_dataset_and_domain_ge_all.sh``` to finish the training, testing and evaluation steps upon both setting with only one execution step. 

The results are printed on the screen and stored in the log.txt in the output directory (e.g., "./output/evaluation/COMMA/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots").

<br>

#### Training and Evaluating other variants

For other variants including vision, language and independent V-L prompting techniques, we provide their corresponding configs and scripts as follows.

```
configs
|–– datasets/
|–– trainers/
|   |–– CoCoOp/
|   |–– CoOp/
|   |–– MaPLe/
|   |–– IVLP/
|   |–– VPT/
```

```
scripts
|–– cocoop/
|–– coop/
|–– language-prompting/
|–– maple/
|–– independent-vlp/
```

Please use the corresponding config and script files and follow the same instructions as provided for COMMA in order to train and evaluate the other variants. Same instructions can be followed to reproduce results of other variants using provided pretrained weights.
This repository also supports using official [CoOp](CoOp.md), [Co-CoOp](Co-CoOp.md) configs and models.
