import numpy as np
import os
import torch
DIR = "./save/student_model"
for filename in os.listdir(DIR):
    if "_0.35_" in filename:
        for ckptfile in os.listdir(os.path.join(DIR, filename)):
            if "best" in ckptfile:
                ckpt = torch.load(os.path.join(DIR, filename,ckptfile))
                print("{} -> {:2.2f}".format(filename, float(ckpt['best_acc'])) , flush=True)


