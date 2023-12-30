CUDA=0
# seed=1 
CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_train_comma.sh imagenet 1
# seed=2 
CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_train_comma.sh imagenet 2
# seed=3 
CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/xd_train_comma.sh imagenet 3