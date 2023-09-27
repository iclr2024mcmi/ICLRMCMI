# python  TSNE.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --title Vanilla_Teacher,_drop_class_10
# python  TSNE.py --path_t ./save/models/resnet56_cifar10_FT_lr_0.05_decay_0.0005_CMI_-0.25_trial_0/resnet56_best.pth --title Vanilla_Teacher,_drop_class_10

# python  TSNE.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --title tsne_Vanilla_Teacher
# python  TSNE.py --path_t ./save/models/resnet56_cifar10_FT_lr_0.05_decay_0.0005_CMI_-0.25_trial_0/resnet56_best.pth --title tsne_CMI_Teacher


python  TSNE.py --path_t ./save/student_model/S:resnet20_T:resnet56_CMI:vanilla_kd_r:0.1_a:0.9_b:0.0_trail_1_drop_9.0/resnet20_best.pth \
                --title Vanilla_Teacher,_drop_class_10
python  TSNE.py --path_t ./save/student_model/S:resnet20_T:resnet56_CMI:-0.25_kd_r:0.1_a:0.9_b:0.0_trail_1_drop_9.0/resnet20_best.pth \
                --title CMI_Teacher,_drop_class_10

# python  TSNE.py --path_t ./save/student_model/S:resnet20_T:resnet56_CMI:vanilla_kd_r:0.1_a:0.9_b:0.0_trail_1_drop_9.0_7.0_5.0/resnet20_best.pth \
#                 --title tsne_Vanilla_Teacher,drop_class_6_8_10 --dci 6 8 10
# python  TSNE.py --path_t ./save/student_model/S:resnet20_T:resnet56_CMI:-0.25_kd_r:0.1_a:0.9_b:0.0_trail_1_drop_9.0_7.0_5.0/resnet20_best.pth \
#                 --title tsne_CMI_Teacher,drop_class_6_8_10 --dci 6 8 10

# python  TSNE.py --path_t ./save/student_model/S:resnet20_T:resnet56_CMI:vanilla_kd_r:0.1_a:0.9_b:0.0_trail_1_drop_9.0_7.0/resnet20_best.pth \
#                 --title tsne_Vanilla_Teacher,drop_class_8_10 --dci 8 10
# python  TSNE.py --path_t ./save/student_model/S:resnet20_T:resnet56_CMI:-0.25_kd_r:0.1_a:0.9_b:0.0_trail_1_drop_9.0_7.0/resnet20_best.pth \
#                 --title tsne_CMI_Teacher,drop_class_8_10 --dci 8 10
# python  TSNE.py --path_t ./save/student_model/S:resnet20_T:resnet56_CMI:vanilla_kd_r:0.1_a:0.9_b:0.0_trail_1_drop_9.0/resnet20_best.pth \
#                 --title tsne_Vanilla_Teacher,drop_class_10 --dci 10
# python  TSNE.py --path_t ./save/student_model/S:resnet20_T:resnet56_CMI:-0.25_kd_r:0.1_a:0.9_b:0.0_trail_1_drop_9.0/resnet20_best.pth \
#                 --title tsne_CMI_Teacher,drop_class_10 --dci 10
