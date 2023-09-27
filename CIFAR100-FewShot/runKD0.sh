CUDA_VISIBLE_DEVICES=1 python  train_student.py --path_t ./save/models/resnet56_cifar100_FT_lr_0.05_decay_0.0005_CMI_-0.15_trial_0/resnet56_best.pth \
                                                --distill crd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --frac 0.35 --trial 3
CUDA_VISIBLE_DEVICES=1 python  train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth  \
                                                --distill kd --model_s resnet20 -a 0 -b 0.8 --frac 0.35 --trial 4
