# Other possible dataset values includes [imagenet, caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]
CUDA=0
# seed=1
# trains and evaluates on base classes
CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/base2new_train_comma.sh imagenet 1 
# evaluates on novel classes
CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/base2new_test_comma.sh imagenet 1

# seed=2
# trains and evaluates on base classes
CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/base2new_train_comma.sh imagenet 2
# evaluates on novel classes
CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/base2new_test_comma.sh imagenet 2

# seed=3
# trains and evaluates on base classes
CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/base2new_train_comma.sh imagenet 3
# evaluates on novel classes
CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/base2new_test_comma.sh imagenet 3