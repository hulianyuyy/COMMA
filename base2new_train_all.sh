# Other possible dataset values includes [imagenet, caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]
CUDA=0
for dataset in imagenet caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat
do
    # seed=1
    # trains and evaluates on base classes
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/base2new_train_comma.sh ${dataset} 1 
    # evaluates on novel classes
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/base2new_test_comma.sh ${dataset} 1

    # seed=2
    # trains and evaluates on base classes
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/base2new_train_comma.sh ${dataset} 2
    # evaluates on novel classes
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/base2new_test_comma.sh ${dataset} 2

    # seed=3
    # trains and evaluates on base classes
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/base2new_train_comma.sh ${dataset} 3
    # evaluates on novel classes
    CUDA_VISIBLE_DEVICES=$CUDA bash scripts/comma/base2new_test_comma.sh ${dataset} 3
done