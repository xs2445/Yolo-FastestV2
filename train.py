import os
import math
import time
import argparse
import numpy as np
from numpy.lib.npyio import _save_dispatcher
from tqdm import tqdm
from numpy.testing._private.utils import print_assert_equal

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import dataset
from numpy.core.fromnumeric import shape

from torchsummary import summary

import utils.loss
import utils.utils
import utils.datasets
import model.detector as Detectors


if __name__ == '__main__':
    # 指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='',
                        help='Specify training profile *.data')
    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)

    save_path = cfg["save_path"]
    if save_path == None or not os.path.exists(save_path):
        save_path = 'weight'
        print("Model save path not sapecified, will save to ./{}".format(save_path))
    else:
        print("Model save path: {}".format(save_path))

    log_path = os.path.join(save_path, 'log.txt')
    if log_path == None or not os.path.exists(os.path.dirname(log_path)):
        log_path = 'weight'
        print("Log path not sapecified, will save to ./{}".format(log_path))
    else:
        with open(log_path, 'w') as f:
            pass
        print("Log path: {}".format(log_path))

    print("训练配置:")
    print(cfg)

    # 数据集加载
    train_dataset = utils.datasets.TensorDataset(cfg["train"], cfg["width"], cfg["height"], imgaug = True)
    val_dataset = utils.datasets.TensorDataset(cfg["val"], cfg["width"], cfg["height"], imgaug = False)

    batch_size = int(cfg["batch_size"] / cfg["subdivisions"])
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # 训练集
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=utils.datasets.collate_fn,
                                                   num_workers=nw,
                                                   pin_memory=True,
                                                   drop_last=True,
                                                   persistent_workers=True
                                                   )
    #验证集
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 collate_fn=utils.datasets.collate_fn,
                                                 num_workers=nw,
                                                 pin_memory=True,
                                                 drop_last=False,
                                                 persistent_workers=True
                                                 )

    # 指定后端设备CUDA&CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 判断是否加载预训练模型
    load_param = False
    premodel_path = cfg["pre_weights"]
    if premodel_path != None and os.path.exists(premodel_path):
        load_param = True
        numClass = cfg["pre_classes"]
    else:
        numClass = cfg["classes"]
    
    model = Detectors.Detector(numClass, cfg["anchor_num"], load_param)

    # 加载预训练模型参数
    if load_param == True:
        model.load_state_dict(torch.load(premodel_path, map_location=device), strict = False)
        if numClass != cfg["classes"]:
            model.output_cls_layers = nn.Conv2d(72, cfg["classes"], 1, 1, 0, bias=True)
        print("Loaded finefune model param: %s" % premodel_path)
    else:
        # 初始化模型结构
        print("Initialize weights: model/backbone/backbone.pth")
    
    model.to(device)
    summary(model, input_size=(3, cfg["height"], cfg["width"]))

    # 构建SGD优化器
    optimizer = optim.SGD(params=model.parameters(),
                          lr=cfg["learning_rate"],
                          momentum=0.949,
                          weight_decay=0.0005,
                          )

    # 学习率衰减策略
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg["steps"],
                                               gamma=0.1)

    print('Starting training for %g epochs...' % cfg["epochs"])

    batch_num = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        pbar = tqdm(train_dataloader)

        for imgs, targets in pbar:
            # 数据预处理
            imgs = imgs.to(device).float() / 255.0
            targets = targets.to(device)

            # 模型推理
            preds = model(imgs)
            # loss计算
            iou_loss, obj_loss, cls_loss, total_loss = utils.loss.compute_loss(preds, targets, cfg, device)

            # 反向传播求解梯度
            total_loss.backward()

            #学习率预热
            for g in optimizer.param_groups:
                warmup_num =  5 * len(train_dataloader)
                if batch_num <= warmup_num:
                    scale = math.pow(batch_num/warmup_num, 4)
                    g['lr'] = cfg["learning_rate"] * scale
                    
                lr = g["lr"]

            # 更新模型参数
            if batch_num % cfg["subdivisions"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            # 打印相关信息
            info = "Epoch:%d LR:%f CIou:%f Obj:%f Cls:%f Total:%f" % (
                    epoch, lr, iou_loss, obj_loss, cls_loss, total_loss)
            pbar.set_description(info)

            batch_num += 1

        # 模型保存
        if epoch % 10 == 0 and epoch > 0:
            model.eval()
            #模型评估
            print("computer mAP...")
            _, _, AP, _ = utils.utils.evaluation(val_dataloader, cfg, model, device)
            print("computer PR...")
            precision, recall, _, f1 = utils.utils.evaluation(val_dataloader, cfg, model, device, 0.3)
            print("Precision:%f Recall:%f AP:%f F1:%f"%(precision, recall, AP, f1))
            model_save_path = os.path.join(save_path, 
              "%s-%d-epoch-%fap-model.pth" % (cfg["model_name"], epoch, AP)) 
            torch.save(model.state_dict(), model_save_path)
            print('Model Saved to {}'.format(model_save_path))
            with open(log_path, 'a') as f:
                f.write("Time:%f Epoch:%d Precision:%f Recall:%f AP:%f F1:%f\n"%(time.time(), epoch, precision, recall, AP, f1))
            

        # 学习率调整
        scheduler.step()
