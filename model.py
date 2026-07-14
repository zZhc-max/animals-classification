import torch.nn as nn
import torch

'''
使用nn.Sequential， 将一系列的层结构打包，形成一个整体
'''
class AlexNet(nn.Module):
    def __init__(self, num_classes):#定义初始化方法
        super(AlexNet, self).__init__()
        # 特征提取部分
        self.feature = nn.Sequential(
            # 第一层，图像原始3通道变为96通道
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),)

        # 分类器部分，接受卷积的输出作为输入(需要先flatten)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes) )

    def forward(self, x):
        # print(x.size()) 查看输入数据维度：B,3,224,224
        x = self.feature(x) # ouput:[1, 256, 6, 6]
        # print(x.size())
        x = torch.flatten(x, start_dim=1)  # 展平处理，从channel维度开始展平，（第一个维度为channel）# 256 * 6 * 6
        x = self.classifier(x) # [1, 7]
        return x


if __name__ == '__main__':
    x = torch.randn((1, 3, 224, 224))
    model = AlexNet(num_classes=7)
    out = model(x)
    print(out.shape)
    print(out)