import torch
import torch.nn as nn
import torch.fft as fft
from efficient_kan import KAN
# 设置超参数
k = 4
L = 0.01
drop_out = 0.4
out_channel = 16
activation = nn.ELU()
fn=150


class se_block(nn.Module):
    def __init__(self, channel, ratio=8):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# 定义tCNN网络
class tCNN(nn.Module):
    def __init__(self,win_data):
        super(tCNN, self).__init__()

        # 第一卷积层
        self.conv1 = nn.Conv2d(1, out_channel, kernel_size=(9, 1), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out_channel)
        self.activation1 = activation

        # 第二卷积层
        self.dropout2 = nn.Dropout(drop_out)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, win_data), stride=5, padding=(0,23), bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_channel)
        self.activation2 = activation

        # 第三卷积层
        self.dropout3 = nn.Dropout(drop_out)
        self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), stride=1, padding=(0, 0), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(out_channel)
        self.activation3 = activation

        # 第四卷积层
        self.dropout4 = nn.Dropout(drop_out)
        self.conv4 = nn.Conv2d(out_channel, 32, kernel_size=(1, 6), stride=1, padding=(0, 0), bias=False)
        self.batchnorm4 = nn.BatchNorm2d(32)
        self.activation4 = activation

        # 全连接层
        self.flatten = nn.Flatten()
        self.dropout_fc = nn.Dropout(drop_out)
        self.fc = nn.Linear(32, k)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 前向传播
        x = self.activation1(self.batchnorm1(self.conv1(x)))

        x = self.activation2(self.batchnorm2(self.conv2(self.dropout2(x))))

        x = self.activation3(self.batchnorm3(self.conv3(self.dropout3(x))))
        x = self.activation4(self.batchnorm4(self.conv4(self.dropout4(x))))
        x = self.flatten(x)
        x = self.fc(self.dropout_fc(x))
        #x = self.softmax()
        return x

class net2(nn.Module):
    def __init__(self,win_data):
        super(net2, self).__init__()
        # 第一卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(9, 1), bias=False)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # 前向传播

        x = self.conv1(x)
        x = self.flatten(x)
        x = torch.abs(fft.fft(x))

        return x
class CNNMLP(nn.Module):
    def __init__(self,win_data):
        super(CNNMLP, self).__init__()
        self.se = se_block(9)
        self.tCNN = tCNN(win_data)
        self.net2 = net2(win_data)
        self.fc = nn.Linear(8, 4)
        self.KAN = KAN([win_data*16, win_data, win_data//5,4])

    def forward(self, x):
        x = torch.permute(x,[0, 2 ,1 ,3])
        x = self.se(x)
        x = torch.permute(x,[0, 2 ,1 ,3])
        tcnn_data = self.tCNN(x)
        net2_data = self.net2(x)
        net2_data = self.KAN(net2_data)
        x = torch.cat((tcnn_data, net2_data), 1)  # 横向拼接
        x = self.fc(x)
        return x




