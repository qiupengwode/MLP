import numpy as np

import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from tensorboardX import SummaryWriter
import torch

import dataLoader
from helper import Configure, Quality, Log
import time
import math
from model import Net
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def test(model, times):
    ##加载数据
    Log.d('正在验证...')
    val_loader = dataLoader.getFBP500TestDataLoader(times)
    p = Configure()
    aug_val_loader = list(
        enumerate(dataLoader.getDataLoader(p.getFBP500TestTxt(times), p.FBP500_AUG_IMG, p.train_aug_transform)))

    out = list()
    lab = list()
    for j, d in enumerate(val_loader):
        val_ima, val_lab = d
        val_ima = val_ima.to(device)
        _,ad=aug_val_loader[j]
        val_aug_ima,val_aug_lab=ad
        val_out = model(val_aug_ima,val_ima)
        for i in range(val_out.size(0)):
            out.append(val_out[i].item())
            lab.append(val_lab[i].item())
    a = torch.tensor(out)
    b = torch.tensor(lab)
    pc = Quality.PC(a, b, isTensor=True)
    rmse = Quality.RMSE(a, b, isTensor=True)
    mae = Quality.MAE(a, b, isTensor=True)
    Log.i(f'total----PC ：{pc} RMSE ：{rmse} MAE ：{mae}')
    return pc, rmse, mae
    # plt.plot(a,c='red',label='out')
    # plt.plot(b,c='blue',label='label')
    # plt.show()


def train():
    """
    训练函数，运行前配置
    *savePath 保存位置,需要手动设置
    saveLogPath日志保存位置
    saveWeightPath模型权重保存位置
    saveTxtPath 日志Txt保存位置
    """

    savePath = '../logs/FBP5500/resNext50/'
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    Log.d(time.asctime(time.localtime(time.time())))

    p = Configure()

    Log.d('加载模型——————》》》》》》》》》》》》》')


    # pc,rmse,mae
    pcs = []
    rmses = []
    maes = []
    Log.d('开始5折训练——————》》》》》》》》》》》》》')
    for index in range(0, 5):
        Log.d('开始第%d次训练......' % (index))
        # 使用预训练的ResNet50
        Log.d("加载模型配置——————》》》》》》》》》》》》》")
        model =Net().to(device)
        model.train()
        # model=ResNet50_att.resnet50(pretrained=True).to(device)

        # model.load_state_dict(torch.load('E:/PyProjects/MLP/logs/FBP500/resNet50/weight/resNet50-8200-epochs-0.018446.pth'))
        mse = nn.MSELoss()

        # kl = nn.KLDivLoss(size_average=False, reduction='sum')
        optimizer = optim.SGD(model.parameters(), lr=p.LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
        # optimizer=optim.Adam(model.parameters(),lr=p.LEARNING_RATE,weight_decay=5e-4)
        lr_schedu = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1000, gamma=0.1)

        Log.d('加载存储器——————》》》》》》》》》》》》》')
        saveLogPath = savePath + f'/{index}/events'
        if not os.path.exists(saveLogPath):
            os.makedirs(saveLogPath)
        saveWriter = SummaryWriter(log_dir=saveLogPath)

        Log.d(time.asctime(time.localtime(time.time())))
        Log.d('开始加载训练数据——————》》》》》》》》》》》》》')
        orgin_train_loader = dataLoader.getFBP500TrainDataLoader(times=index)
        p=Configure()
        aug_train_loader=list(enumerate(dataLoader.getDataLoader(p.getFBP500TrainTxt(index),p.FBP500_AUG_IMG,p.train_aug_transform)))
        Log.i('损失函数：%s\t优化器：%s\t学习率衰减器：%s\t存储位置：%s' % ('MSE', 'SGD', 'stepLr', savePath))
        Log.d('开始训练...')
        pcm = 0
        rmsem = 0
        maem = 0
        lastWeight = ''
        for epoch in range(p.START_EPOCHS, p.END_EPOCHS):

            running_loss = 0.0
            for i, data in enumerate(orgin_train_loader):
                img, label = data
                label = label.to(device)
                j,augdata=aug_train_loader[i]
                augimg,auglabel=augdata
                # img = torch.stack([tor01.1
                # +ch.rot90(img, k, (2, 3)) for k in range(4)], 0).view((-1, 3, 224, 224))
                # label = torch.stack([label, label, label, label], 0).view(-1, 5)
                img = img.to(device)
                out = model(augimg.to(device),img)
                loss1 = mse(out, label)
                loss = loss1
                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
            saveWriter.add_scalar('trainLoss', scalar_value=running_loss, global_step=epoch)
            lr_schedu.step()
            print('Train {} epoch, Loss:{:.6f}'.format(epoch, running_loss))
            """
            开始验证val
            """
            if epoch % 5== 0:
                model.eval()
                pc, rmse, mae = test(model, times=index)
                model.train()
                if pc > pcm:
                    pcm = pc
                    rmsem = rmse
                    maem = mae
                    Log.d('正在保存模型...')
                    if os.path.exists(lastWeight):
                        os.remove(lastWeight)
                    saveWeightPathPro=savePath+f'{index}/weight/'
                    if not os.path.exists(saveWeightPathPro):
                        os.makedirs(saveWeightPathPro)
                    saveWeightPath = saveWeightPathPro+'resNext50-{}-epoch-{:.6f}-{:.4f}.pth'.format(index, epoch,
                                                                                                        running_loss,
                                                                                                        pc)
                    lastWeight = saveWeightPath
                    torch.save(model.state_dict(), saveWeightPath)

        pcs.append(pcm)
        rmses.append(rmsem)
        maes.append(maem)

    Log.d('训练完成！！!正在保存输出结果——————》》》》》》》》》》》》》')
    Log.d("pc")
    Log.d(pcs)
    Log.d("rmse")
    Log.d(rmses)
    Log.d("mae")
    Log.d(maes)
    Log.d('avg： PC %.4f  RMSE %.4f MAE %.4f' % (sum(pcs) / 5, sum(rmses) / 5, sum(maes) / 5))
    Log.i('损失函数：%s\t优化器：%s\t学习率衰减器：%s\t存储位置：%s' % ('MSE', 'SGD', 'stepLr', savePath))
    saveTxtPath = savePath + 'result.txt'
    file = open(saveTxtPath, 'w')
    file.writelines('损失函数：%s\t优化器：%s\t学习率衰减器：%s\t存储位置：%s' % ('MSE', 'SGD', 'stepLr', savePath))
    file.writelines('pc')
    for i in pcs:
        file.write(i.item() + ' ')
    file.write('\n')
    file.writelines('rmse')
    for i in rmses:
        file.write(i.item() + ' ')
    file.write('\n')
    file.writelines('mae')
    for i in maes:
        file.write(i.item() + ' ')
    file.write('\n')
    file.writelines('avg： PC %.4f  RMSE %.4f MAE %.4f' % (sum(pcs) / 5, sum(rmses) / 5, sum(maes) / 5))
    file.close()
    Log.d('训练结束啦，快看看结果吧！！！')


def predict(fileName: str):
    ima = Image.open(fileName).convert('RGB')
    # ima2=ima.resize(size=(128,128))
    # img=torch.tensor(np.array(ima)).unsqueeze(dim=0)
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img = trans(ima).unsqueeze(dim=0)
    # img=torch.transpose(img,dim0=1,dim1=3)
    model = torchvision.models.resnet50(pretrained=True)
    numtr = model.fc.in_features
    model.fc = torch.nn.Linear(numtr, 1)
    model.load_state_dict(
        torch.load('E:/PyProjects/MLP/logs/FBP5500/resNet50/weight/resNet50-4600-epochs-0.000157.pth'))
    res = model(img)
    print("预测结果：%.4f" % (res.item()))
    ima.show()


if __name__ == '__main__':
    train()
