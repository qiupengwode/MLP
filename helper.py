from abc import abstractmethod, ABCMeta
import numpy as np
import torch
from torchvision.transforms import transforms
import torch


class Quality():

    @staticmethod
    def RMSE(x, y, isTensor=True):
        if isTensor:
            return torch.sqrt((1 / len(x)) * (torch.sum(torch.square(x - y))))

        return np.sqrt((1 / len(x)) * (np.sum(np.square(x - y))))

        # return np.sqrt(np.mean(np.square(y-x)))

    @staticmethod
    def MAE(x, y, isTensor=True):
        if isTensor:
            return (1 / len(x)) * torch.sum(torch.abs(x - y))
        return (1 / len(x)) * np.sum(np.abs(x - y))
        # return np.mean(np.abs(y-x))

    @staticmethod
    def PC(x, y, isTensor=True):
        if isTensor:
            x_ = torch.mean(x)
            y_ = torch.mean(y)
            sx = torch.sqrt(torch.sum(torch.square(x - x_)))
            sy = torch.sqrt(torch.sum(torch.square(y - y_)))
            pc = torch.sum(torch.mul((x - x_) / sx, (y - y_) / sy))
            return pc
        x_ = np.mean(x)
        y_ = np.mean(y)
        sx = np.sqrt(np.sum(np.square(x - x_)))
        sy = np.sqrt(np.sum(np.square(y - y_)))
        pc = np.sum(np.multiply((x - x_) / sx, (y - y_) / sy))
        return pc
        # return np.corrcoef(y,x)[0][1]


class Configure():
    def __init__(self, model=None):
        '''
        初始化参数
        :param params:参数配置
        :param kv:单个参数配置
        '''
        self.BATCH_SIZE = 16
        self.LEARNING_RATE = 0.001
        self.START_EPOCHS = 1
        self.END_EPOCHS = 2000
        self.DATA_ROOT = 'G:/PersonalList/LearnMaterial/datasets'
        self.TRAIN_FBP500_TXT = self.DATA_ROOT + '/SCUT-FBP数据集/Rating_Collection/Rating_Collection/train.txt'
        self.TEST_FBP500_TXT = self.DATA_ROOT + '/SCUT-FBP数据集/Rating_Collection/Rating_Collection/test.txt'
        self.FBP500_IMG = self.DATA_ROOT + '/SCUT-FBP数据集/Data_Collection/Data_Collection'
        self.FBP500_AUG_IMG=self.DATA_ROOT + '/SCUT-FBP数据集/Data_Collection/DataAug'
        self.TRAIN_FBP5500_TXT = self.DATA_ROOT + '/SCUT-FBP5500_with_Landmarks/SCUT-FBP5500_with_Landmarks/' \
                                                  'train_test_files/split_of_60%training and 40%testing/train.txt'
        self.FBP5500_IMG = self.DATA_ROOT + '/SCUT-FBP5500_with_Landmarks/SCUT-FBP5500_with_Landmarks/Images'
        self.TEST_FBP5500_TXT = self.DATA_ROOT + '/SCUT-FBP5500_with_Landmarks/SCUT-FBP5500_with_Landmarks/' \
                                                 'train_test_files/split_of_60%training and 40%testing/test.txt'

        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.train_aug_transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 通用优化器
        # self.optimizer=torch.optim.Adam(model.parameters(),lr=self.LEARNING_RATE,weight_decay=1e-4)
        # self.optimizer = torch.optim.SGD(model.parameters(), lr=self.LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
        # self.lr_schedu = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=5000, gamma=0.1)
        pass

    def getFBP500TrainLDLTxt(self, times):
        return self.DATA_ROOT + '/SCUT-FBP数据集/Rating_Collection/Rating_Collection/' + 'LDLTrain' + str(times) + '.txt'

    def getFBP500TestLDLTxt(self, times):
        return self.DATA_ROOT + '/SCUT-FBP数据集/Rating_Collection/Rating_Collection/' + 'LDLTest' + str(times) + '.txt'

    def getFBP500TestTxt(self, times):
        return self.DATA_ROOT + '/SCUT-FBP数据集/Rating_Collection/Rating_Collection/' + 'test' + str(times) + '.txt'

    def getFBP500TrainTxt(self, times):
        return self.DATA_ROOT + '/SCUT-FBP数据集/Rating_Collection/Rating_Collection/' + 'train' + str(times) + '.txt'

    def getFBP5500TrainTxt(self, times):
        times += 1
        return self.DATA_ROOT + '/SCUT-FBP5500_with_Landmarks/SCUT-FBP5500_with_Landmarks/train_test_files/5_folders_cross_validations_files/' + 'cross_validation_' + str(
            times) + '/train_' + str(times) + '.txt'

    def getFBP5500TestTxt(self, times):
        times += 1
        return self.DATA_ROOT + '/SCUT-FBP5500_with_Landmarks/SCUT-FBP5500_with_Landmarks/train_test_files/5_folders_cross_validations_files/' + 'cross_validation_' + str(
            times) + '/test_' + str(times) + '.txt'


class Log():

    @staticmethod
    def d(info):
        flag = True
        if flag:
            print('debug:', info)

    @staticmethod
    def i(info):
        flag = True
        if flag:
            print('info:', info)

    @staticmethod
    def e(info):
        flag = True
        if flag:
            print('wrong:', info)

if __name__ == '__main__':
    Log.d([1,2,3])