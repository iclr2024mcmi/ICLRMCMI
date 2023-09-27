CUDA_VISIBLE_DEVICES=0 python  train_teacher.py --model resnet56 --dataset cifar10
CUDA_VISIBLE_DEVICES=0 python  FineTuneTeacher.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth  --dataset cifar10 --param 1 -0.25
CUDA_VISIBLE_DEVICES=0 python  train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --trial 1 --dataset cifar10 --drop_class --drop_class_idx 0 3

