CUDA_VISIBLE_DEVICES=1 python  train_teacher.py --model vgg11
CUDA_VISIBLE_DEVICES=1 python  FineTuneTeacher.py --path_t ./save/models/vgg11_vanilla/ckpt_epoch_240.pth --param 1 -0.25
CUDA_VISIBLE_DEVICES=1 python  train_student.py --path_t ./save/models/vgg11_vanilla/ckpt_epoch_240.pth --distill kd --model_s MobileNetV2 -r 0.1 -a 0.9 -b 0 --trial 1
CUDA_VISIBLE_DEVICES=1 python  train_student.py --path_t ./save/models/vgg11_vanilla/ckpt_epoch_240.pth --distill crd --model_s MobileNetV2 -a 0 -b 0.8 --trial 1
CUDA_VISIBLE_DEVICES=1 python  train_student.py --path_t ./save/models/vgg11_cifar100_FT_lr_0.05_decay_0.0005_CMI_-0.25_trial_0/ckpt_last.pth --distill kd --model_s MobileNetV2 -r 0.1 -a 0.9 -b 0 --trial 1
CUDA_VISIBLE_DEVICES=1 python  train_student.py --path_t ./save/models/vgg11_cifar100_FT_lr_0.05_decay_0.0005_CMI_-0.25_trial_0/ckpt_last.pth --distill crd --model_s MobileNetV2 -a 0 -b 0.8 --trial 1
