import torch
import torchvision
import sys
from matplotlib import pyplot as plt
import time
from RiskNet import RiskNet
import numpy as np


def load_rs_data(batch_size, resize=None, root='./rs'):
    # 用于存放一系列变换的列表
    trans = [torchvision.transforms.RandomResizedCrop(resize),
             torchvision.transforms.RandomHorizontalFlip(),
             torchvision.transforms.ToTensor()]
    # 构造变换
    transform = torchvision.transforms.Compose(trans)

    # 分别加载训练和测试数据并且应用变换
    # 每一类的数据都放在一个文件夹里
    data_train = torchvision.datasets.ImageFolder(root=root + "/train", transform=transform)
    data_test = torchvision.datasets.ImageFolder(root=root + "/test", transform=transform)

    # 加载数据使用的进程数
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4

    # 构造迭代器
    train_iter = torch.utils.data.DataLoader(data_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(data_test,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)

    return train_iter, test_iter


# 装载数据
def load_data(batch_size, resize=None, root='./data/CIFAR10'):
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)

    # torchvision.datasets包内还有许多其他的数据集，只需要修改网络的输入通道数和输出类别数即可
    data_train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    data_test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    print(data_train)
    print(data_train[0][0])
    print(torch.max(data_train[2][0][0]))
    print(torch.min(data_train[2][0][0]))
    print(torch.mean(data_train[2][0][0]))

    print(torch.max(data_train[2][0][1]))
    print(torch.min(data_train[2][0][1]))
    print(torch.mean(data_train[2][0][1]))

    print(torch.max(data_train[2][0][2]))
    print(torch.min(data_train[2][0][2]))
    print(torch.mean(data_train[2][0][2]))
    to_img = torchvision.transforms.ToPILImage()
    a = to_img(data_train[2][0])
    plt.imshow(a)
    plt.show()

    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4

    train_iter = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


# 训练模型
def train(train_iter, test_iter, net, optimizer, device, num_epochs):
    # 将网络部署在gpu设备上
    net = net.to(device)
    print("training on", device)

    # 交叉熵
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0

    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            # 输入的属性
            X = X.to(device)
            # 标签
            y = y.to(device)
            # 预测
            y_hat = net(X)
            # 计算损失
            l = loss(y_hat, y)
            # 梯度下降
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        # 测试集的准确率
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch {}, loss {:.4f}, train acc {:.4f}, test acc{:.4f}, time {:.2f} sec'
              .format(epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


# 训练模型
def trainUntil(train_iter, test_iter, net, optimizer, device, acc_threshold, acc_mean_num, max_epoch):
    logs = []
    times = []
    cur_acc_list = []
    total_acc_list = []
    loss_list = []

    # 将网络部署在gpu设备上
    net = net.to(device)
    print("training on", device)

    # 交叉熵
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0

    epoch_counter = 0
    while True:
        epoch_counter += 1
        if epoch_counter <= max_epoch:
            train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
            for X, y in train_iter:
                # 输入的属性
                X = X.to(device)
                # 标签
                y = y.to(device)
                # 预测
                y_hat = net(X)
                # 计算损失
                l = loss(y_hat, y)
                # 梯度下降
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

                train_l_sum += l.cpu().item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                n += y.shape[0]
                batch_count += 1

            # 测试集的准确率
            test_acc = evaluate_accuracy(test_iter, net)
            cur_acc_list.append(test_acc)
            total_test_acc = np.mean(cur_acc_list[-acc_mean_num:])
            total_acc_list.append(total_test_acc)
            times.append(epoch_counter)
            loss_list.append(train_l_sum / batch_count)

            if total_test_acc < acc_threshold:
                out_str = 'epoch {}, loss {:.8f}, train acc {:.4f}, cur test acc {:.4f}, mean test acc {:.4f} , target acc {:.4f}, time {:.2f} sec'.format(
                    epoch_counter,
                    train_l_sum / batch_count,
                    train_acc_sum / n,
                    test_acc,
                    total_test_acc,
                    acc_threshold,
                    time.time() - start)
                print(out_str)
                logs.append(out_str + '\n')

                # 训练过程影像保存
                if epoch_counter % 5 == 0:
                    plt.cla()
                    plt.figure(figsize=(8, 5))
                    plt.plot(times, cur_acc_list, label='Cur Acc')
                    plt.plot(times, total_acc_list, label='Mean Acc(' + acc_mean_num.__str__() + ' Epoch)')
                    plt.scatter(times[-1], total_acc_list[-1], color='red',
                                label='Acc:' + round(total_test_acc, 4).__str__())
                    # plt.plot(times, loss_list, label='Loss')
                    plt.legend()
                    plt.tight_layout()
                    # plt.pause(0.1) # 打开就会实时可视化，但相对较慢，这里直接保存
                    plt.savefig("./figs/Epoch_" + epoch_counter.__str__().zfill(4) + ".png", dpi=200)
                    plt.close()  # 关闭绘图
            else:
                print('Training Finished!!')
                out_str = 'epoch {}, loss {:.8f}, train acc {:.4f}, cur test acc {:.4f}, mean test acc {:.4f}, time {:.2f} sec'.format(
                    epoch_counter,
                    train_l_sum / batch_count,
                    train_acc_sum / n,
                    test_acc,
                    total_test_acc,
                    time.time() - start)
                print(out_str)
                logs.append(out_str + '\n')
                break
        else:
            print('Reached Max Training Epoch!!')
            out_str = 'epoch {}, loss {:.8f}, train acc {:.4f}, cur test acc {:.4f}, total test acc {:.4f}, target acc {:.4f}, time {:.2f} sec'.format(
                epoch_counter,
                train_l_sum / batch_count,
                train_acc_sum / n,
                test_acc,
                total_test_acc,
                acc_threshold,
                time.time() - start)
            print(out_str)
            logs.append(out_str + '\n')
            break
    return logs


# 评估模型在测试集的表现
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                # 评估模式, 关闭dropout
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                # 改回训练模式
                net.train()
            else:
                if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


if __name__ == '__main__':
    # Hyperparameters
    batchsize = 128  # 可以根据电脑的实际情况进行修改
    target_size = 224  # 网络输入影像大小，这里与imagenet数据集保持一致，为224*224
    lr = 0.001  # 学习率
    num_epoch = 3  # 指定训练的epoch次数
    acc_threshold = 0.9  # 在测试集上的精度，直到大于等于才停止
    acc_mean_num = 30  # 连续n个测试集的精度平均值，作为最终精度
    max_epoch = 20000  # 最大训练次数

    # 训练开始计时
    start_time = time.time()
    str_start_time = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(start_time))

    # 如果有gpu计算设备，选择gpu计算设备，否则选择在cpu上训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 装载网络
    risknet = RiskNet()
    print(risknet)

    # 装载数据
    train_iter, test_iter = load_rs_data(batch_size=batchsize, resize=target_size)

    # 优化算法采用Adam算法，也可以选择torch.optim包下其他的优化算法
    optimizer = torch.optim.Adam(risknet.parameters(), lr=lr)

    # 开始训练
    train_log = trainUntil(train_iter=train_iter,
                           test_iter=test_iter,
                           net=risknet,
                           optimizer=optimizer,
                           device=device,
                           acc_threshold=acc_threshold,
                           acc_mean_num=acc_mean_num,
                           max_epoch=max_epoch)

    # 训练结束计时
    end_time = time.time()
    str_end_time = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(end_time))

    # 保存log文件
    fout = open('train_log ' + str_start_time + '.txt', 'w')
    for i in range(len(train_log)):
        fout.write(train_log[i])
    fout.write('Total cost time: ')
    fout.write((end_time - start_time).__str__() + ' s')
    fout.close()

    # 保存整个神经网络
    torch.save(risknet, "riskNet " + str_start_time + ".pkl")
