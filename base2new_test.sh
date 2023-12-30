# Other possible dataset values includes [imagenet, caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]
# prints averaged results for base classes
python parse_test_res.py output/base2new/train_base/shots_16/COMMA/vit_b16_c2_ep5_batch4_2ctx/imagenet --test-log
# averaged results for novel classes
python parse_test_res.py output/base2new/test_new/shots_16/COMMA/vit_b16_c2_ep5_batch4_2ctx/imagenet --test-log