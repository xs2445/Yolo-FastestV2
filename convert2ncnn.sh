# script to convert pytorch model to ncnn model
# input arguments:
# $1: yolo data file
# $2: pytorch model file
# example:
# !bash convert2ncnn.sh 'data/coco.data' 'modelzoo/coco2017-0.241078ap-model.pth'
python3 pytorch2onnx.py --data $1 --weights $2 --output yolo-fastestv2.onnx
python3 -m onnxsim yolo-fastestv2.onnx yolo-fastestv2-opt.onnx
./sample/ncnn/bin/onnx2ncnn yolo-fastestv2-opt.onnx yolo-fastestv2.param yolo-fastestv2.bin
./sample/ncnn/bin/ncnnoptimize yolo-fastestv2.param yolo-fastestv2.bin yolo-fastestv2-opt.param yolo-fastestv2-opt.bin 1
# cp yolo-fastestv2-opt* ./sample/ncnn/model