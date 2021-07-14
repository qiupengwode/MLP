import torchvision
import torch.nn as nn
import resnet
import torchsummary
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.project=nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
    def forward(self,x):
        return self.project(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.project=resnet.resnext50_32x4d(pretrained=False,channel=1)
        self.feature = resnet.resnext50_32x4d(pretrained=False, channel=3)
        self.fc = nn.Linear(512 * 4, 1)
    def forward(self,x1,x2):
        x1=self.project(x1)
        x2=self.feature(x2)

        x3=x1*x2
        x3=self.fc(x3)
        return x3

if __name__ == '__main__':
    model=Net()
    torchsummary.summary(model,input_size=[(1,224,224),(3,224,224)],device='cpu')

