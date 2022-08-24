import ncnn
import cv2
import numpy as np
import os

BOXCOLOR = {
    0: (255,0,0),
    1: (0,255,0),
    2: (0,0,255)
}

CLSTHRESH = {
    0: 0.6, # person
    1: 0.5, # bottle
    2: 0.1  # smoke
}

def draw_boxes(img, box_list):
    img_boxed = img.copy()
    for box in box_list:
        if box.score >= CLSTHRESH[box.category]:
            cv2.rectangle(
                img_boxed,
                (box.x_left, box.y_top),
                (box.x_right, box.y_down),
                BOXCOLOR[box.category],
            )
        # text = "%s %.1f%%" % (box.category, box.score*100)
        # label_size, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    return img_boxed


#加载data
def load_datafile(data_path):
    #需要配置的超参数
    cfg = {"model_name":None,
    
           "epochs": None,
           "steps": None,           
           "batch_size": None,
           "subdivisions":None,
           "learning_rate": None,

           "pre_weights": None,        
           "classes": None,
           "width": None,
           "height": None,           
           "anchor_num": None,
           "anchors": None,

           "val": None,           
           "train": None,
           "names":None
        }

    assert os.path.exists(data_path), "请指定正确配置.data文件路径"

    #指定配置项的类型
    list_type_key = ["anchors", "steps"]
    str_type_key = ["model_name", "val", "train", "names", "pre_weights"]
    int_type_key = ["epochs", "batch_size", "classes", "width",
                   "height", "anchor_num", "subdivisions"]
    float_type_key = ["learning_rate"]
    
    #加载配置文件
    with open(data_path, 'r') as f:
        for line in f.readlines():
            if line == '\n' or line[0] == "[":
                continue
            else:
                data = line.strip().split("=")
                #配置项类型转换
                if data[0] in cfg:
                    if data[0] in int_type_key:
                       cfg[data[0]] = int(data[1])
                    elif data[0] in str_type_key:
                        cfg[data[0]] = data[1]
                    elif data[0] in float_type_key:
                        cfg[data[0]] = float(data[1])
                    elif data[0] in list_type_key:
                        cfg[data[0]] = [float(x) for x in data[1].split(",")]
                    else:
                        print("配置文件有错误的配置项")
                else:
                    print("%s配置文件里有无效配置项:%s"%(data_path, data))
    return cfg


def intersectionArea(box_a, box_b):
    # no intersection
    if box_a.x_left > box_b.x_right or box_b.x_left > box_a.x_right or box_a.y_top > box_b.y_down or box_b.y_top > box_a.y_down:
        return 0
    inter_width = min(box_a.x_right, box_b.x_right) - max(box_a.x_left, box_b.x_left)
    inter_height = min(box_a.y_down, box_a.y_down) - min(box_a.y_top, box_a.y_top)

    return inter_width * inter_height

def IoU(box_a, box_b):
    interArea = intersectionArea(box_a, box_b)
    return interArea / (box_a.area() + box_b.area() - interArea)


def NMS(box_list, iou_thresh, top_k=-1, candidate_size=500):
    # ascending sort
    sortFunc = lambda box: box.score
    box_list.sort(key=sortFunc)
    # limit the size of box list
    box_list = box_list[-candidate_size:]
    picked = []
    while len(box_list) > 0:
        # get the one with highest prob
        curr = box_list[-1]
        picked.append(box_list.pop(-1))
        # keep the boxes with top k prob
        if 0 < top_k and top_k == len(picked): # 0<top_k==len(picked)
            break
        box_list = [x for x in box_list if IoU(curr, x) <= iou_thresh or curr.category != x.category]

    return picked

class TargetBox:
    def __init__(self, x_left, x_right, y_top, y_down, category, score) -> None:
        self.x_left = int(x_left)
        self.x_right = int(x_right)
        self.y_top = int(y_top)
        self.y_down = int(y_down)
        self.category = category
        self.score = score

    def getWidth(self):
        return self.x_right - self.x_left
    
    def getHeight(self):
        return self.y_down - self.y_top
    
    def area(self):
        return self.getWidth() * self.getHeight()
    
    def __str__(self):
        return "{} {:.2f} \t{} {} {} {}".format(self.category, self.score*100, self.x_left, self.y_top, self.x_right, self.y_down)


class ncnnModel:
    # ncnn model loader for yolo-fastest-v2
    def __init__(self, param_path, bin_path, datafile_path, detThresh=0.2, nmsThresh=0.25, input_shape=352):
        self.input_name = 'input.1'
        self.input_shape = input_shape
        self.output_name = ['794', '796']
        self.mean_vals = [0] * 3
        self.norm_vals = [1/255.0] * 3
        self.net = ncnn.Net()
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        cfg = load_datafile(datafile_path)
        self.anchor = cfg["anchors"]
        self.numAnchor = cfg["anchor_num"]
        self.detThresh = detThresh
        self.nmsThresh = nmsThresh
        self.num_threads = 6

    def predHandle(self, mat_out_list, scaleW, scaleH, thresh):
        resultBoxes = []
        for i, mat_out in enumerate(mat_out_list):
            outH = mat_out.c
            outW = mat_out.h
            outC = mat_out.w
            stride = self.input_shape/outH
            mat_out = np.array(mat_out)     # shape: 22*22*95, float32
            for h in range(outH):
                for w in range(outW):
                    grid_value = mat_out[h,w]   # shape: 95, = 3*4+3+8
                    # (x,y,w,h)*numAnch + objS*numAnch + numCate
                    # idxs:
                    # objScore: grid_value[4*numAnchor + index]
                    # clsScore: grid_value[5*numAnchor:]
                    # score: objScore * clsScore
                    category = np.argmax(grid_value[5*self.numAnchor:])
                    maxClsScore = grid_value[category + 5*self.numAnchor]
                    for iA in range(self.numAnchor):
                        score = maxClsScore * grid_value[4*self.numAnchor + iA]
                        if score > thresh:
                            bcx = ((grid_value[iA * 4 + 0] * 2 - 0.5) + w) * stride
                            bcy = ((grid_value[iA * 4 + 1] * 2 - 0.5) + h) * stride
                            bw = ((grid_value[iA * 4 + 2] * 2) ** 2) * self.anchor[(i * self.numAnchor * 2) + iA * 2 + 0]
                            bh = ((grid_value[iA * 4 + 3] * 2) ** 2) * self.anchor[(i * self.numAnchor * 2) + iA * 2 + 1]

                            x_left = (bcx - 0.5 * bw) * scaleW
                            y_top = (bcy - 0.5 * bh) * scaleH
                            x_right = (bcx + 0.5 * bw) * scaleW
                            y_down = (bcy + 0.5 * bh) * scaleH

                            resultBoxes.append(TargetBox(x_left, x_right, y_top, y_down, category, score))

        return resultBoxes

    def detection(self, img):
        scaleH, scaleW = img.shape[0]/self.input_shape, img.shape[1]/self.input_shape
        mat_in = ncnn.Mat.from_pixels_resize(
            img,
            ncnn.Mat.PixelType.PIXEL_BGR,
            img.shape[1],
            img.shape[0],
            self.input_shape,
            self.input_shape,
        )
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)
        
        # feed data to model
        ex = self.net.create_extractor()
        ex.set_num_threads(self.num_threads)
        ex.input(self.input_name, mat_in)
        # inference
        mat_out_list = []
        for outname in self.output_name:
            ret, mat = ex.extract(outname)
            mat_out_list.append(mat)
        # prediction postprocess
        result = self.predHandle(mat_out_list, img.shape[1]/self.input_shape, img.shape[0]/self.input_shape, self.detThresh)
        # non-maximum suppression
        result = NMS(result, self.nmsThresh)

        return result
    
    def __call__(self, img):
        return self.detection(img)
    

        

        
        
