# CUDA_VISIBLE_DEVICES=0 python  train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --trial 1 --dataset cifar10 --drop_class --drop_class_idx 9
# CUDA_VISIBLE_DEVICES=0 python  train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --trial 1 --dataset cifar10 --drop_class --drop_class_idx 9 7

CUDA_VISIBLE_DEVICES=0 python  train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd \
                                                --model_s resnet20 -r 0.1 -a 0.9 -b 0 --trial 1 --dataset cifar10 --drop_class --drop_class_idx 9 7 5 3 1 

CUDA_VISIBLE_DEVICES=0 python  train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd \
                                                --model_s resnet20 -r 0.1 -a 0.9 -b 0 --trial 1 --dataset cifar10 --drop_class --drop_class_idx 9 7 5 3 1 0
                                                