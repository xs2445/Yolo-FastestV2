# Augmentation

from utils.augdetection import augmentation as Tdet
import uuid
import glob 
import os
import cv2
import random
from google.colab.patches import cv2_imshow
import numpy as np
import matplotlib.pyplot as plt
import math


if __name__ == '__main__':

    img_dir = './data/train'
    img_list = glob.glob(img_dir+'/*.png') + glob.glob(img_dir+'/*.jpg')
    aug_dir = './data/train/aug'
    if not os.path.exists(aug_dir):
        os.mkdir(aug_dir)

    train_file = open('./data/train.txt', 'a')
    train_file.write('\n')

    # random.shuffle(img_list)

    epoch = 1

    augseq = Tdet.AugmentationSequence([
                Tdet.RandomHorizontalFlip(prob=0.3),
                Tdet.RandomScale(0.8),
                Tdet.RandomRotation(10),
                Tdet.RandomTranslation(0.1),
                Tdet.RandomContrastBrightness(),
                Tdet.RandomMotionBlur(),
                Tdet.RandomHSV(),
            ], prob=1)

    count = 0

    for _ in range(epoch):
        for img_name in img_list:

            file_name = img_name.replace('png', 'txt').replace('jpg','txt')
            # get label
            annotation = []
            with open(file_name, 'r') as f:
                for line in f.readlines():
                    l = line.strip().split(" ")
                    annotation.append([l[i] for i in range(0,5)])
            annotation = np.array(annotation, dtype=np.float32)

            # get img
            img = cv2.imread(img_name)
            # img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)

            # augmentation
            img_aug, annotation_aug = augseq(img, annotation)

            if annotation_aug is not None:
                suffix = img_name.split('.')[-1]
                img_name_new = os.path.join(aug_dir, str(uuid.uuid1()) + '.' + suffix)
                file_name_new = img_name_new.replace('png', 'txt').replace('jpg','txt') 
                msg = "\n".join(["{:d} {:.6f} {:.6f} {:.6f} {:.6f}".format(int(x[0]), x[1], x[2], x[3], x[4]) for x in annotation_aug])
                # print(msg)
                # print()
                cv2.imwrite(img_name_new, img_aug)
                train_file.write(os.path.join('aug', os.path.basename(img_name_new)) + '\n')
                with open(file_name_new, 'w') as f:
                    f.write(msg)
                count += 1
            if count % 500 == 499:
                print(count)

    train_file.close()