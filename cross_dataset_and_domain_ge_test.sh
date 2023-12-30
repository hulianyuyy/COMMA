CUDA=0
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