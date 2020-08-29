import torch
import torchvision
import numpy as np
from PIL import Image


def getLevel(index):
    if index == 0:
        return "A"
    elif index == 1:
        return "B"
    elif index == 2:
        return "C"
    elif index == 3:
        return "D"
    elif index == 4:
        return "E"


if __name__ == '__main__':
    net_path = "riskNet 2020-08-29 17-35-11.pkl"
    img_path = "./rs/train/risk0/001.jpg"

    # 加载网络与参数
    riskNet = torch.load(net_path)
    print(riskNet)

    # 加载影像并进行一些预处理
    img = Image.open(img_path)
    trans = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor()
    ])
    img_pro2 = np.expand_dims(trans(img), 0)  # 扩展成四维张量
    img_tensor = torch.from_numpy(img_pro2).cuda()  # 将数据类型转换为张量

    predict_res = riskNet(img_tensor)  # 调用网络进行推理

    res_numpy = predict_res.cpu().detach().numpy()[0]  # 将预测结果转换为Numpy矩阵
    index = np.where(res_numpy == np.max(res_numpy))[0][0]  # 获取最大值所对应的索引
    riskLevel = getLevel(index)  # 根据索引获取对应的风险等级
    print(index, riskLevel)  # 输出
