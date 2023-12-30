CUDA=0
# seed=1 
CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_train_comma.sh imagenet 1
# seed=2 
CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_train_comma.sh imagenet 2
# seed=3 
CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_train_comma.sh imagenet 3

for SEED in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_test_comma.sh caltech101 ${SEED}
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_test_comma.sh oxford_pets ${SEED}
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_test_comma.sh stanford_cars ${SEED}
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_test_comma.sh oxford_flowers ${SEED}
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_test_comma.sh food101 ${SEED}
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_test_comma.sh fgvc_aircraft ${SEED}
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_test_comma.sh sun397 ${SEED}
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_test_comma.sh dtd ${SEED}
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_test_comma.sh eurosat ${SEED}
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_test_comma.sh ucf101 ${SEED}
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_test_comma.sh imagenetv2 ${SEED}
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_test_comma.sh imagenet_sketch ${SEED}
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_test_comma.sh imagenet_a ${SEED}
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_test_comma.sh imagenet_r ${SEED}
done

for dataset in caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat imagenetv2 imagenet_sketch imagenet_a imagenet_r
do
    python parse_test_res.py output/evaluation/COMMA/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots/${dataset} --test-log
done

python parse_test_res.py output/imagenet/COMMA/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots --test-log