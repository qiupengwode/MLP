from abc import ABCMeta, abstractmethod
import os
import numpy as np
from helper import Log, Configure, Quality
import re
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import xlrd
import random


class MyDataSet(Dataset):
    def __init__(self, data, y, transform=None):
        self.transform = transform
        self.data = data
        self.y = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.y[item]


class MyDataSet2(Dataset):
    def __init__(self, data, y, transform=None):
        self.transform = transform
        self.data = data
        self.y = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]

        if self.transform:
            sample = self.transform(sample)

        x1 = sample[:, :76, :76]
        x2 = sample[:, :76, 76:152]
        x3 = sample[:, :76, 152:228]
        x4 = sample[:, 76:152, :76]
        x5 = sample[:, 76:152, 76:152]
        x6 = sample[:, 76:152, 152:228]
        x7 = sample[:, 152:228, :76]
        x8 = sample[:, 152:228, 76:152]
        x9 = sample[:, 152:228, 152:228]
        return sample, self.y[item]


class Data():

    @staticmethod
    def splitData(mode=1):
        """
        :param mode:1交叉验证法 2留一法 3自助法
        :return:xtrain,ytrain,xtest,ytest
        """

    @staticmethod
    def loadDirData(path, start=3):
        if not os.path.exists(path):
            pass

    @staticmethod
    def read_img(root, filedir, transform=None):
        # Data loading
        with open(filedir, 'r') as f:
            lines = f.readlines()
        output = []
        for line in lines:
            linesplit = line.split('\n')[0].split(' ')
            addr = linesplit[0]
            target = torch.Tensor([float(linesplit[1])])
            img = Image.open(os.path.join(root, addr)).convert('RGB')

            if transform is not None:
                img = transform(img)

            output.append([img, target])

        return output

    # SCUPT-FBP-5500
    @staticmethod
    def loadTextData(txtPath, imaPath):
        '''

        :param txtPath imaPath:文件路径：01.jpg 2.5
        :return:tensor[nlists,scores]  nlists:(b,w,h,c),scores:[b,s]
        '''
        if not os.path.exists(txtPath):
            pass
        file = open(txtPath)
        lists = []
        y = []
        while True:
            str = file.readline()
            if str:
                s = str.split(' ')
                lists.append(s[0])
                y.append([float(s[1].rstrip('\n'))])
            else:
                break
        x = []
        for i in lists:
            wholePath = imaPath + '/' + i
            if not os.path.exists(wholePath):
                print(wholePath + ':wrong Path')
                exit()
            else:
                try:
                    a = Image.open(wholePath)
                except Warning:
                    print(wholePath)
                else:
                    x.append(a)
        return x, torch.tensor(y)

    # SCUPT-FBP-500
    @staticmethod
    def loadXlsxData(txtPath, imgPath):
        '''
        读取xlsx文件，加载数据集
        :param txtPath:xlsx文件路径
        :param imgPath:图片数据路径
        :return:torch类型：(x,y)
        '''
        table = xlrd.open_workbook(txtPath, 'r')
        sheet1 = table.sheet_by_index(0)
        imaNames = sheet1.col_values(0, start_rowx=1)
        imaNames = [imgPath + r'\SCUT-FBP-' + str(int(imaNames[i])) + '.jpg' for i in range(len(imaNames))]
        raters = sheet1.col_values(1, start_rowx=1)
        x = []
        for path in imaNames:
            if not os.path.exists(path):
                print(path + ':wrong path')
                exit()
            else:
                x.append(Image.open(path).convert('RGB'))
        y = torch.tensor(np.array(raters), dtype=torch.float32)
        y = y.unsqueeze(dim=1)
        return x, y

    @staticmethod
    def createFBPTxt():
        '''
        创建FBP500数据集训练txt、测试txt
        :return:
        '''
        txt1Path = r'F:\learnMaterials\Datasets\SCUT-FBP数据集\Rating_Collection\Rating_Collection\Attractiveness label.xlsx'
        table = xlrd.open_workbook(txt1Path, 'r')
        sheet1 = table.sheet_by_index(0)
        imaNames = sheet1.col_values(0, start_rowx=1)
        imaNames = ['SCUT-FBP-' + str(int(imaNames[i])) + '.jpg' for i in range(len(imaNames))]
        raters = sheet1.col_values(1, start_rowx=1)
        res = []
        for i in range(len(imaNames)):
            res.append([imaNames[i], raters[i]])
        random.shuffle(res)
        s1 = -100
        s2 = 0
        for index in range(5):
            s1 += 100
            s2 += 100
            with open(f'F:/learnMaterials/Datasets/SCUT-FBP数据集/Rating_Collection/Rating_Collection/train{index}.txt',
                      'w') as file1:
                e1 = 0
                e2 = s1
                for _ in range(2):
                    for j in range(e1, e2):
                        file1.writelines(str(res[j][0]) + ' ' + str(res[j][1]) + '\n')
                    e1 = s2
                    e2 = 500
                file1.close()
            with open(f'F:/learnMaterials/Datasets/SCUT-FBP数据集/Rating_Collection/Rating_Collection/test{index}.txt',
                      'w') as file2:
                for j in range(s1, s2):
                    file2.writelines(str(res[j][0]) + ' ' + str(res[j][1]) + '\n')
                file2.close()

    @staticmethod
    def createFBP500LDLTXT():
        from random import shuffle
        txtPath = r'F:\learnMaterials\Datasets\SCUT-FBP数据集\Rating_Collection\Rating_Collection\Ratings of all raters.xlsx'
        table = xlrd.open_workbook(txtPath, 'r')
        sheet1 = table.sheet_by_index(0)
        imgNames = sheet1.col_values(1, start_rowx=1)
        scores = sheet1.col_values(2, start_rowx=1)
        pre = '1'
        res = [[0, 0, 0, 0, 0] for _ in range(500)]
        for i in range(len(imgNames)):
            if imgNames[i] != pre:
                pre = imgNames[i]
            res[int(pre) - 1][int(scores[i]) - 1] += 1

        sort = [i for i in range(500)]
        shuffle(sort)

        s1 = -100
        s2 = 0
        for index in range(5):
            s1 += 100
            s2 += 100
            saveFile = f'F:/learnMaterials/Datasets/SCUT-FBP数据集/Rating_Collection/Rating_Collection/LDLTrain{index}.txt'
            file = open(saveFile, 'w')
            e1 = 0
            e2 = s1
            for _ in range(2):
                for k in range(e1, e2):
                    i = sort[k]
                    count = 0
                    for j in range(5):
                        count += int(res[i][j])
                    rate1 = str(int(res[i][0]) / count)
                    rate2 = str(int(res[i][1]) / count)
                    rate3 = str(int(res[i][2]) / count)
                    rate4 = str(int(res[i][3]) / count)
                    rate5 = str(int(res[i][4]) / count)
                    file.writelines('SCUT-FBP-' + str(
                        i + 1) + '.jpg' + ' ' + rate1 + ' ' + rate2 + ' ' + rate3 + ' ' + rate4 + ' ' + rate5 + '\n')
                e1 = s2
                e2 = 500
            file.close()
            saveFile = f'F:/learnMaterials/Datasets/SCUT-FBP数据集/Rating_Collection/Rating_Collection/LDLTest{index}.txt'
            file = open(saveFile, 'w')
            for k in range(s1, s2):
                i = sort[k]
                count = 0
                maxi = 0
                for j in range(5):
                    count += int(res[i][j])
                rate1 = str(int(res[i][0]) / count)
                rate2 = str(int(res[i][1]) / count)
                rate3 = str(int(res[i][2]) / count)
                rate4 = str(int(res[i][3]) / count)
                rate5 = str(int(res[i][4]) / count)
                file.writelines('SCUT-FBP-' + str(i + 1) + '.jpg' + ' ' + str(rate1) + ' ' + str(rate2) + ' ' + str(
                    rate3) + ' ' + str(rate4) + ' ' + str(rate5) + ' ' + str(maxi + 1) + '\n')
            pass

    @staticmethod
    def loadFBP500LDLTxt(txtPath, imaPath):
        file = open(txtPath)
        lists = []
        y = []
        while True:
            str = file.readline()
            if str:
                s = str.strip('\n').split(' ')
                lists.append(s[0])
                y.append([[float(s[i]) for i in range(1, 6)]])
            else:
                break

        x = []
        for i in lists:
            wholePath = imaPath + '/' + i
            if not os.path.exists(wholePath):
                print(wholePath + ':wrong Path')
                exit()
            else:
                try:
                    a = Image.open(wholePath).convert("RGB")
                    x.append(a)
                except:
                    print(wholePath)
        return x, torch.tensor(y)

    @staticmethod
    def loadAVAData():
        pass


def _loader(batchSize=4, txtPath='', imgPath='', transform=None):
    x, y = Data.loadTextData(txtPath, imgPath)
    dataSet = MyDataSet(x, y, transform=transform)
    return torch.utils.data.DataLoader(dataset=dataSet, batch_size=batchSize, shuffle=False)


def getFBP500TrainDataLoader(times=0):
    config = Configure()
    txtPath = config.getFBP500TrainTxt(times)
    imgPath = config.FBP500_IMG
    transformer = config.train_transform
    return _loader(batchSize=config.BATCH_SIZE,txtPath=txtPath, imgPath=imgPath, transform=transformer)


def getFBP500TestDataLoader(times=0):
    config = Configure()
    txtPath = config.getFBP500TestTxt(times)
    imgPath = config.FBP500_IMG
    transformer = config.test_transform
    return _loader(batchSize=config.BATCH_SIZE,txtPath=txtPath, imgPath=imgPath, transform=transformer)


def getFBP5500TrainDataLoader(times=0):
    config = Configure()
    txtPath = config.getFBP5500TrainTxt(times)
    imgPath = config.FBP5500_IMG
    transformer = config.train_transform
    return _loader(batchSize=config.BATCH_SIZE,txtPath=txtPath, imgPath=imgPath, transform=transformer)


def getFBP5500TestDataLoader(times=0):
    config = Configure()
    txtPath = config.getFBP5500TestTxt(times)
    imgPath = config.FBP5500_IMG
    transformer = config.test_transform
    return _loader(batchSize=config.BATCH_SIZE,txtPath=txtPath, imgPath=imgPath, transform=transformer)


def getDataLoader(txtPath='', imgPath='',transformer=None):
    config = Configure()
    return _loader(batchSize=config.BATCH_SIZE,txtPath=txtPath, imgPath=imgPath, transform=transformer)


#
#
# def _loader(times=0, flag='train', data='FBP500'):
#     config = Configure()
#     path = ''
#     imgPath = ''
#     if data == 'FBP500':
#         imgPath = config.FBP500_IMG
#         if flag == 'train':
#             transform = config.train_transform
#             path = config.getFBP500TrainTxt(times)
#         elif flag == 'test':
#             transform = config.test_transform
#             path = config.getFBP500TestTxt(times)
#     elif data == 'FBP5500':
#         imgPath = config.FBP5500_IMG
#         if flag == 'train':
#             transform = config.train_transform
#             path = config.getFBP5500TrainTxt(times)
#         elif flag == 'test':
#             transform = config.test_transform
#             path = config.getFBP5500TestTxt(times)
#     x,y=Data.loadTextData(path, imgPath)
#     dataSet = MyDataSet(x,y, transform=transform)
#     return torch.utils.data.DataLoader(dataset=dataSet, batch_size=config.BATCH_SIZE, shuffle=False)
#
#
# def getFBP500TrainDataLoader(times=0):
#     return _loader(times=times, flag='train', data='FBP500')
#
#
# def getFBP500TestDataLoader(times=0):
#     return _loader(times=times, flag='test', data='FBP500')
#
#
# def getFBP5500TrainDataLoader(times=0):
#     return _loader(times=times, flag='train', data='FBP5500')
#
#
# def getFBP5500TestDataLoader(times=0):
#     return _loader(times=times, flag='train', data='FBP5500')


if __name__ == '__main__':
    l = getFBP500TrainDataLoader(1)
    a = getFBP500TrainDataLoader(1)
    b = list(enumerate(l))
    c = list(enumerate(a))
    for i, v in b:
        j, d = c[i]
        print(i,j)
