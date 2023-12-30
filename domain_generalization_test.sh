CUDA=0
for SEED in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_test_comma.sh imagenetv2 ${SEED}
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_test_comma.sh imagenet_sketch ${SEED}
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_test_comma.sh imagenet_a ${SEED}
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_test_comma.sh imagenet_r ${SEED}
done