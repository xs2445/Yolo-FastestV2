import glob, cv2, time
from ncnn_utils import ncnnModel, draw_boxes


def main():
    param_path = './ncnn_model/yolo-fastestv2.param'
    bin_path = './ncnn_model/yolo-fastestv2.bin'
    datafile_path = './data/dla.data'

    model = ncnnModel(
        param_path, 
        bin_path,
        datafile_path,
        detThresh=0.1,
        nmsThresh=0.25,
        input_shape=352
    )
    
    name_list = glob.glob('./data/test/*.png')

    t_s = time.time()
    for input_name in name_list:
        img = cv2.imread(input_name)
        results = model(img)
    t_e = time.time()
    
    print("FPS:{:.2f}  Talt.Time:{:.2f}  Num.:{}".format(
        len(name_list)/(t_e-t_s),
        t_e-t_s,
        len(name_list)
        ))

if __name__ == '__main__':
    main()
    