%cd /content/Yolo-FastestV2-NCNN-RasPi4B/

import glob 
import os
import cv2
import random
from google.colab.patches import cv2_imshow
import numpy as np
import matplotlib.pyplot as plt
import math


def xywh2xyxy(x, w=None, h=None):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    if x is None:
        return x
    if w is not None or h is not None:
        assert w is not None and h is not None, "Only got either h, w."
    else:
        w = h = 1
    y = np.zeros_like(x).astype(np.float32)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2) * w  # top left x
    y[:, 1] = (x[:, 1] - x[:, 3] / 2) * h  # top left y
    y[:, 2] = (x[:, 0] + x[:, 2] / 2) * w  # bottom right x
    y[:, 3] = (x[:, 1] + x[:, 3] / 2) * h  # bottom right y
    return y

def xyxy2xywh(x, w=None, h=None):
    if x is None:
         return x
    if w is not None or h is not None:
        assert w is not None and h is not None, "Only got either h, w."
    else:
        w = h = 1
    y = np.zeros_like(x).astype(np.float32)
    y[:, 0] = ((x[:,0] + x[:,2]) / 2) / w   # x
    y[:, 1] = ((x[:,1] + x[:,3]) / 2) / h   # y
    y[:, 2] = (x[:,2] - x[:,0]) / w         # w
    y[:, 3] = (x[:,3] - x[:,1]) / h         # h
    return y

def draw_boxes(img, box_ary, color=(255,0,255)):
    h, w = img.shape[:2]
    img_boxed = img.copy()
    for box in box_ary:
        cv2.rectangle(
            img_boxed,
            (int(box[0]*w), int(box[1]*h)),
            (int(box[2]*w), int(box[3]*h)),
            color,
            2
        )
    return img_boxed


def bbox_area_xyxy(bbox):
    return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])

def bbox_area_xywh(bbox):
    return bbox[:,2]*bbox[:,3]


class AffineTransformation():
    def __init__(self, thresh_a=0.25):
        self.thresh_a=0.25

    @staticmethod
    def affineImage(img, warp_mat):
        return cv2.warpAffine(img, warp_mat, (img.shape[1], img.shape[0]))

    @staticmethod
    def affineBox(bboxes, warp_mat):
        # input: xyxy boxes in image scale (640,480)
        rot_mat = warp_mat[:,:-1]
        trans_mat = np.squeeze(warp_mat[:,-1])
        new_bboxes = []
        for bbox in bboxes:
            x1,y1,x2,y2 = bbox
            points = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ])
            points_warp = points @ rot_mat.T + trans_mat
            x1, y1 = points_warp.min(axis=0)
            x2, y2 = points_warp.max(axis=0)
            new_bboxes.append([x1, y1, x2, y2])

        return np.array(new_bboxes)
    
    @classmethod
    def affineTransform(cls, img, bboxes, warp_mat):
        img = cls.affineImage(img)
        bboxes = cls.affineBox(bboxes)
        bboxes = cls.clipBox(bboxes, cls.thresh_a)
        return img, bboxes
    
    @staticmethod
    def clipBox(bboxes, bboxes_original, thresh_a=0.1, thresh_b=0.25, thresh_area=0.02):
        # pass in xyxy bboxes
        area_original = bbox_area_xyxy(bboxes_original)
        area_unclip = bbox_area_xyxy(bboxes)
        bboxes = np.clip(bboxes, 0, 1)
        area = bbox_area_xyxy(bboxes)
        bboxes = [x for i, x in enumerate(bboxes) if area[i] > thresh_area**2 and area[i]/area_unclip[i] > thresh_b and  area[i]/area_original[i] >= thresh_a]

        return np.array(bboxes)


class RandomTranslation(AffineTransformation):
    def __init__(self, translate = 0.2, thresh_a=0.1, thresh_b=0.25, diff = False, prob=0.5):
        self.translate = translate
        self.thresh_a = thresh_a
        self.thresh_b = thresh_b
        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range"  
            assert self.translate[0] > 0 & self.translate[0] < 1
            assert self.translate[1] > 0 & self.translate[1] < 1
        else:
            assert self.translate > 0 and self.translate < 1
            self.translate = (-self.translate, self.translate)
        self.diff = diff
        self.prob = 0.5

    def __call__(self, img, bboxes):
        if random.uniform(0,1) < self.prob:
            return self.operation(img, bboxes)
        else:
            return img, bboxes
    
    def operation(self, img, bboxes):
        h, w, _ = img.shape
        translate_factor_x = random.uniform(*self.translate)
        translate_factor_y = random.uniform(*self.translate)
        if not self.diff:
            translate_factor_y = translate_factor_x
        warp_mat = np.float32([
            [1,0,translate_factor_x*w],
            [0,1,translate_factor_y*h]
        ])
        img_warp = self.affineImage(img, warp_mat)
        # input: xywh bbox
        bboxes_original = xywh2xyxy(bboxes.copy())
        bboxes = xywh2xyxy(bboxes)
        bboxes += np.array([translate_factor_x, translate_factor_y, translate_factor_x, translate_factor_y])
        bboxes = self.clipBox(bboxes, bboxes_original, thresh_a=self.thresh_a, thresh_b=self.thresh_b)
        bboxes = xyxy2xywh(bboxes) if len(bboxes)>0 else None
            
        return img_warp, bboxes


class RandomRotation(AffineTransformation):
    def __init__(self, rotation=10, thresh_a=0.1, thresh_b=0.25, prob=0.5):
        # input 0-360 degree
        self.rotation = rotation/180*math.pi
        self.thresh_a = thresh_a
        self.thresh_b = thresh_b
        self.prob = prob

    def __call__(self, img, bboxes):
        if random.uniform(0,1) < self.prob:
            return self.operation(img, bboxes)
        else:
            return img, bboxes
    
    def operation(self, img, bboxes):
        h, w, _ = img.shape
        angle = random.uniform(-self.rotation, self.rotation)
        warp_mat = np.float32([
            [math.cos(angle), -math.sin(angle), w/2*(1-math.cos(angle))+h/2*math.sin(angle)],
            [math.sin(angle), math.cos(angle), h/2*(1-math.cos(angle))-w/2*math.sin(angle)]
        ])
        img_warp = self.affineImage(img, warp_mat)
        bboxes_original = xywh2xyxy(bboxes.copy())
        bboxes = self.affineBox(xywh2xyxy(bboxes, w, h), warp_mat)
        bboxes[:,0] /= w
        bboxes[:,1] /= h
        bboxes[:,2] /= w
        bboxes[:,3] /= h
        bboxes = self.clipBox(bboxes, bboxes_original, self.thresh_a, self.thresh_b)
        bboxes = xyxy2xywh(bboxes)

        return img_warp, bboxes


class RandomScale(AffineTransformation):
    def __init__(self, scaleR=0.4, thresh_a=0.1, thresh_b=0.25, prob=0.5):
        self.scaleR = scaleR
        assert 0 <= scaleR <= 1, "Scale should be in [0,1]."
        self.thresh_a = thresh_a
        self.thresh_b = thresh_b
        self.prob = prob

    def __call__(self, img, bboxes):
        if random.uniform(0,1) < self.prob:
            return self.operation(img, bboxes)
        else:
            return img, bboxes
    
    def operation(self, img, bboxes):
        h, w, _ = img.shape
        scale = random.uniform(self.scaleR, 2-self.scaleR)
        warp_mat = np.float32([
            [scale, 0, w/2*(1-scale)],
            [0, scale, h/2*(1-scale)]
        ])
        img_warp = self.affineImage(img, warp_mat)
        warp_mat[0,2] /= w
        warp_mat[1,2] /= h
        bboxes_original = xywh2xyxy(bboxes.copy())
        bboxes = self.affineBox(xywh2xyxy(bboxes), warp_mat)
        bboxes = self.clipBox(bboxes, bboxes_original, thresh_a=self.thresh_a, thresh_b=self.thresh_b)
        bboxes = xyxy2xywh(bboxes) if len(bboxes)>0 else None

        return img_warp, bboxes


class RandomHorizontalFlip():
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, bboxes):
        if random.uniform(0,1) < self.prob:
            return self.operation(img, bboxes)
        else:
            return img, bboxes
    
    def operation(self, img, bboxes):
        bboxes[:,0] = 1 - bboxes[:,0]
        return img[:,::-1], bboxes


class RandomContrastBrightness():
    def __init__(self, alphaR=(0.25, 1.75), betaR=(0.25, 1.75), prob=0.5):
        self.alphaR = alphaR
        self.betaR = betaR
        self.prob = prob
    
    def __call__(self, img, bboxes):
        if random.uniform(0,1) < self.prob:
            return self.operation(img), bboxes
        else:
            return img, bboxes
        
    def operation(self, img):
        alpha = random.uniform(*self.alphaR)
        beta = random.uniform(*self.betaR)
        blank = np.zeros(img.shape, img.dtype)
        # dst = alpha * img + beta * blank
        dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
        
        return dst


class RandomMotionBlur():
    def __init__(self, degree=8, prob=0.5) -> None:
        assert 2<= degree, "degree should >= 2"
        self.degree = (2,degree)
        self.prob = prob

    def __call__(self, img, bboxes):
        if random.uniform(0,1) < self.prob:
            return self.operation(img), bboxes
        else:
            return img, bboxes
    
    def operation(self, img):
        degree = random.randint(*self.degree)
        angle = random.uniform(-360, 360)
        img = np.array(img)
        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(img, -1, motion_blur_kernel)
        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)

        return blurred


class RandomHSV():
    def __init__(self, hgain = 0.0138, sgain = 0.678, vgain = 0.36, prob=0.5) -> None:
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.prob = prob

    def __call__(self, img, bboxes):
        if random.uniform(0,1) < self.prob:
            return self.operation(img), bboxes
        else:
            return img, bboxes
    
    def operation(self, img):
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8
        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # no return needed

        return img
        

class AugmentationSequence():
    def __init__(self, operation_list, prob=0.5):
        self.operation_list = operation_list
        self.prob = prob
    
    def __call__(self, img, bboxes):
        if random.uniform(0,1) < self.prob:
            for func in self.operation_list:
                img, bboxes = func(img, bboxes)

        return img, bboxes


if __name__ == '__main__':
    img_dir = '/content/gdrive/MyDrive/22summer/lab/2dLidar/Data/ObjectDetection/general/general_0810_xinghua'
    img_list = glob.glob(img_dir+'/*.png')

    train_file = open('./data/train.txt', 'w+')

    random.shuffle(img_list)
    for i in range(5):
        train_file.write(img_list[i] + '\n')
        file_name = img_list[i].replace('png', 'txt').replace('jpg','txt')
        # get label
        label = []
        with open(file_name, 'r') as f:
            for line in f.readlines():
                l = line.strip().split(" ")
                label.append([l[i] for i in range(1,5)])
        label = np.array(label, dtype=np.float32)
        # get img
        img = cv2.cvtColor(cv2.imread(img_list[i]), cv2.COLOR_BGR2RGB)
        # augmentation
        augseq = AugmentationSequence([
            RandomHorizontalFlip(0.3),
            RandomScale(0.8),
            RandomRotation(15),
            RandomTranslation(0.1),
            RandomContrastBrightness(),
            RandomMotionBlur(),
            RandomHSV(),
        ], prob=0.7)
        img_aug, label_aug = augseq(img.copy(), label.copy())
        # draw boxes
        img = draw_boxes(img, xywh2xyxy(label))
        if label_aug is not None:
            img_aug = draw_boxes(img_aug, xywh2xyxy(label_aug), color=(0,255,0))
        # plot
        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        axs[0].imshow(img)
        axs[1].imshow(img_aug)

    train_file.close()