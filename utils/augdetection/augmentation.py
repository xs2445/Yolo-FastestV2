import glob 
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import math



def _xywh2xyxy(x, w=None, h=None):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    if x is None:
        return x
    if w is not None or h is not None:
        assert w is not None and h is not None, "Only got either h, w."
    else:
        w = h = 1
    y = x.copy()
    a=1
    y[:, 0+a] = (x[:, 0+a] - x[:, 2+a] / 2) * w  # top left x
    y[:, 1+a] = (x[:, 1+a] - x[:, 3+a] / 2) * h  # top left y
    y[:, 2+a] = (x[:, 0+a] + x[:, 2+a] / 2) * w  # bottom right x
    y[:, 3+a] = (x[:, 1+a] + x[:, 3+a] / 2) * h  # bottom right y
    return y

def _xyxy2xywh(x, w=None, h=None):
    if x is None:
         return x
    if w is not None or h is not None:
        assert w is not None and h is not None, "Only got either h, w."
    else:
        w = h = 1
    y = x.copy()
    a=1
    y[:, 0+a] = ((x[:,0+a] + x[:,2+a]) / 2) / w   # x
    y[:, 1+a] = ((x[:,1+a] + x[:,3+a]) / 2) / h   # y
    y[:, 2+a] = (x[:,2+a] - x[:,0+a]) / w         # w
    y[:, 3+a] = (x[:,3+a] - x[:,1+a]) / h         # h
    return y




def _draw_boxes(img, box_ary, color=(255,0,255)):
    colors = {
        0: (255,0,0),
        1: (0,255,0),
        2: (0,0,255)
    }
    h, w = img.shape[:2]
    img_boxed = img.copy()
    for box in box_ary:
        cv2.rectangle(
            img_boxed,
            (int(box[1]*w), int(box[2]*h)),
            (int(box[3]*w), int(box[4]*h)),
            colors[box[0]],
            2
        )
    return img_boxed


def _bbox_area_xyxy(bbox):
    return (bbox[:,3] - bbox[:,1])*(bbox[:,4] - bbox[:,2])

def _bbox_area_xywh(bbox):
    return bbox[:,3]*bbox[:,4]


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
            l,x1,y1,x2,y2 = bbox
            points = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ])
            points_warp = points @ rot_mat.T + trans_mat
            x1, y1 = points_warp.min(axis=0)
            x2, y2 = points_warp.max(axis=0)
            new_bboxes.append([l, x1, y1, x2, y2])

        return np.array(new_bboxes)
    
    # @classmethod
    # def affineTransform(cls, img, bboxes, warp_mat):
    #     img = cls.affineImage(img, warp_mat)
    #     bboxes = cls.affineBox(bboxes, warp_mat)
    #     bboxes = cls.clipBox(bboxes, cls.thresh_a)
    #     return img, bboxes
    
    @staticmethod
    def clipBox(bboxes, bboxes_original, thresh_a, thresh_b, thresh_area=0.05):
        # pass in xyxy bboxes
        area_original = _bbox_area_xyxy(bboxes_original)
        area_unclip = _bbox_area_xyxy(bboxes)
        bboxes[:,1:] = np.clip(bboxes[:,1:], 0, 1)
        area = _bbox_area_xyxy(bboxes)
        # bboxes = [x for i, x in enumerate(bboxes) if area[i] > thresh_area**2 and area[i]/area_unclip[i] > thresh_b and  area[i]/area_original[i] >= thresh_a]
        new_bboxes = []
        for i, x in enumerate(bboxes):
            if area[i] > thresh_area**2:
                if area[i]/area_unclip[i] > thresh_b and  area[i]/area_original[i] >= thresh_a:
                    new_bboxes.append(x)

        return np.array(new_bboxes)


class RandomTranslation(AffineTransformation):
    def __init__(self, translate = 0.2, thresh_a=0.1, thresh_b=0.25, diff = True, prob=0.5):
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
        self.prob = prob

    def __call__(self, img, bboxes):
        if random.uniform(0,1) <= self.prob and bboxes is not None:
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
        bboxes_original = _xywh2xyxy(bboxes.copy())
        bboxes = _xywh2xyxy(bboxes)
        bboxes += np.array([0, translate_factor_x, translate_factor_y, translate_factor_x, translate_factor_y])
        bboxes = self.clipBox(bboxes, bboxes_original, thresh_a=self.thresh_a, thresh_b=self.thresh_b)
        bboxes = _xyxy2xywh(bboxes) if len(bboxes)>0 else None
            
        return img_warp, bboxes


class RandomRotation(AffineTransformation):
    def __init__(self, rotation=10, thresh_a=0.1, thresh_b=0.25, prob=0.5):
        # input 0-360 degree
        self.rotation = rotation/180*math.pi
        self.thresh_a = thresh_a
        self.thresh_b = thresh_b
        self.prob = prob

    def __call__(self, img, bboxes):
        if bboxes is None:
            return img, bboxes 
        if random.uniform(0,1) <= self.prob and bboxes is not None:
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
        bboxes_original = _xywh2xyxy(bboxes.copy())
        bboxes = self.affineBox(_xywh2xyxy(bboxes, w, h), warp_mat)
        bboxes[:,1] /= w
        bboxes[:,2] /= h
        bboxes[:,3] /= w
        bboxes[:,4] /= h
        bboxes = self.clipBox(bboxes, bboxes_original, self.thresh_a, self.thresh_b)
        
        bboxes = _xyxy2xywh(bboxes) if len(bboxes) > 0 else None

        return img_warp, bboxes


class RandomScale(AffineTransformation):
    def __init__(self, scaleR=0.4, thresh_a=0.1, thresh_b=0.25, prob=0.5):
        self.scaleR = scaleR
        assert 0 <= scaleR <= 1, "Scale should be in [0,1]."
        self.thresh_a = thresh_a
        self.thresh_b = thresh_b
        self.prob = prob

    def __call__(self, img, bboxes):
        if random.uniform(0,1) <= self.prob and bboxes is not None:
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
        bboxes_original = _xywh2xyxy(bboxes.copy())
        bboxes = self.affineBox(_xywh2xyxy(bboxes), warp_mat)
        bboxes = self.clipBox(bboxes, bboxes_original, thresh_a=self.thresh_a, thresh_b=self.thresh_b)
        bboxes = _xyxy2xywh(bboxes) if len(bboxes)>0 else None

        return img_warp, bboxes


class RandomHorizontalFlip():
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, bboxes):
        if random.uniform(0,1) <= self.prob and bboxes is not None:
            return self.operation(img, bboxes)
        else:
            return img, bboxes
    
    def operation(self, img, bboxes):
        bboxes[:,1] = 1 - bboxes[:,1]
        return img[:,::-1], bboxes


class RandomContrastBrightness():
    def __init__(self, alphaR=(0.25, 1.75), betaR=(0.25, 1.75), prob=0.5):
        self.alphaR = alphaR
        self.betaR = betaR
        self.prob = prob
    
    def __call__(self, img, bboxes):
        if random.uniform(0,1) <= self.prob and bboxes is not None:
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
        if random.uniform(0,1) <= self.prob and bboxes is not None:
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
        if random.uniform(0,1) <= self.prob and bboxes is not None:
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
        if random.uniform(0,1) <= self.prob and bboxes is not None:
            for func in self.operation_list:
                img, bboxes = func(img, bboxes)

        return img, bboxes


if __name__ == '__main__':
    img_dir = '/content/gdrive/MyDrive/22summer/lab/2dLidar/Data/ObjectDetection/general/general_0810_xinghua'
    img_list = glob.glob(img_dir+'/*.png')

    train_file = open('./data/train.txt', 'w+')

    random.shuffle(img_list)
    for i in range(10):
        train_file.write(img_list[i] + '\n')
        file_name = img_list[i].replace('png', 'txt').replace('jpg','txt')
        # get label
        label = []
        with open(file_name, 'r') as f:
            for line in f.readlines():
                l = line.strip().split(" ")
                label.append([l[i] for i in range(0,5)])
        label = np.array(label, dtype=np.float32)
        # print(label)
        # get img
        img = cv2.cvtColor(cv2.imread(img_list[i]), cv2.COLOR_BGR2RGB)
        # augmentation
        prob=1
        augseq = AugmentationSequence([
            RandomHorizontalFlip(prob=prob),
            RandomScale(0.8, prob=prob),
            RandomRotation(15, prob=prob),
            RandomTranslation(0.1, prob=prob),
            RandomContrastBrightness(prob=prob),
            RandomMotionBlur(prob=prob),
            RandomHSV(prob=prob),
        ], prob=prob)
        img_aug, label_aug = augseq(img.copy(), label.copy())
        # print(label)
        print(np.isclose(label[:,0], label_aug[:,0]) if label.shape == label_aug.shape else 'dim diff')
        print()
        # draw boxes
        img = _draw_boxes(img, _xywh2xyxy(label))
        if label_aug is not None:
            img_aug = _draw_boxes(img_aug, _xywh2xyxy(label_aug), color=(0,255,0))
        # plot
        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        axs[0].imshow(img)
        axs[1].imshow(img_aug)

    train_file.close()