CUDA_VISIBLE_DEVICES=0 python  train_student.py --path_t ./save/models/resnet56_cifar10_FT_lr_0.05_decay_0.0005_CMI_-0.15_trial_0/ckpt_last.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --trial 1 --dataset cifar10 --drop_class --drop_class_idx 9 7 5
CUDA_VISIBLE_DEVICES=0 python  train_student.py --path_t ./save/models/resnet56_cifar10_FT_lr_0.05_decay_0.0005_CMI_-0.2_trial_0/ckpt_last.pth --distill kd \
                                                --model_s resnet20 -r 0.1 -a 0.9 -b 0 --trial 1 --dataset cifar10 --drop_class --drop_class_idx 9 7 5
CUDA_VISIBLE_DEVICES=0 python  train_student.py --path_t ./save/models/resnet56_cifar10_FT_lr_0.05_decay_0.0005_CMI_-0.25_trial_0/ckpt_last.pth --distill kd \
                                                --model_s resnet20 -r 0.1 -a 0.9 -b 0 --trial 1 --dataset cifar10 --drop_class --drop_class_idx 9 7 
