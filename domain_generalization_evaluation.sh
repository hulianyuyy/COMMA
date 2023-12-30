# Other possible dataset values includes [imagenet, caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]
# prints averaged results 
for dataset in imagenetv2 imagenet_sketch imagenet_a imagenet_r
do
    python parse_test_res.py output/evaluation/COMMA/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots/${dataset} --test-log
done

python parse_test_res.py output/imagenet/COMMA/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots --test-log
