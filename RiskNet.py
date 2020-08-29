from torch import nn, optim


class RiskNet(nn.Module):
    def __init__(self):
        super(RiskNet, self).__init__()

        # 卷积层
        self.conv = nn.Sequential(
            # 第一层卷积
            # 如果使用mnist类的数据集，输入通道为1，如果使用的cifar等彩色图像的数据集，输入通道为3
            nn.Conv2d(3, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

            # 第二层卷积，开始减小卷积窗口
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

            # 第三层卷积
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),

            # 第四层卷积
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),

            # 第五层卷积
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.fc = nn.Sequential(
            # 第一层全连接层
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            # 第二层全连接层
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            # 第三层全连接层（输出层）
            nn.Linear(4096, 5)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x.view(x.shape[0], -1))
        return x
