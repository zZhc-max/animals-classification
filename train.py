import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import sys
# sys.path.append('./')
from model import AlexNet
import data_input

import os
import json
import time
# from loss import FocalLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = os.path.join(os.path.dirname(__file__), 'data') + '/'

batch_size =  32  #决定了每次喂给模型几张图
lr_gamma = 0.9

train_loader, tra_num = data_input.train_data(data_dir,batch_size)#data_dir指向的文件是alexnet下的data
# validation loader
validate_loader, val_num =  data_input.val_data(data_dir,32)

net = AlexNet(num_classes=6)

pre_train_model =os.path.join(os.path.dirname(__file__),'AlexNet.pth')
save_path = os.path.join(os.path.dirname(__file__),'AlexNet.pth')


net.to(device)  #把网络放到gpu上训练
loss_function = nn.CrossEntropyLoss()

# 写优化器
optimizer = optim.Adam(net.parameters(), lr=0.0001)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.9)


def pre_write_txt(pred, file):  #这就是个打印内容到指定文件的函数
    f = open(file, 'a', encoding='utf-8')
    f.write(str(pred))
    f.write('\n')
    f.close()
    print("-----------------预测结果已经写入文本文件--------------------")

if __name__ == '__main__':
    print(device)

    # 加载已有权重继续训练
    if os.path.exists(pre_train_model):
        net.load_state_dict(torch.load(pre_train_model, map_location=device))
        print(f"已加载预训练模型: {pre_train_model}")

    # 加载之前训练的最佳准确率
    best_acc_path = os.path.join(os.path.dirname(__file__), 'best_acc.json')
    if os.path.exists(best_acc_path):
        with open(best_acc_path, 'r') as f:
            best_acc = json.load(f)['best_acc']
        print(f"已加载历史最佳准确率: {best_acc}")
    else:
        best_acc = 0.0

    for epoch in range(0, 400):  # 从第31轮继续，训练到50轮

        # train
        net.train()  # 在训练过程中调用dropout方法
        running_loss = 0.0
        t1 = time.perf_counter()  # 统计训练一个epoch所需时间
        # print('star')
        tra_acc = 0.0
        for step, data in enumerate(train_loader, start=0):
            # 这个时候的标签还是数字 不是onehot
            images, labels = data
            # print(labels)#结果每次循环打印出来batch=16个数字标签
            optimizer.zero_grad() #清空梯度
            outputs = net(images.to(device))#这个是训练出来的结果
            # soft_outputs = torch.softmax(outputs, dim=1)
            # print(soft_outputs)
            # print(soft_outputs.max(1),torch.argmax(soft_outputs))
            loss = loss_function(outputs, labels.to(device))
            # print(loss)
            loss.backward()
            optimizer.step() #优化器步进

            tra_predict_y = torch.max(outputs, dim=1)[1]
            step_acc = (tra_predict_y == labels.to(device)).sum().item()
            tra_acc += step_acc
            running_loss += loss.item()
            if (step+1)%32 == 0:#就是说step到了15打印信息，每一个step对应一个batch=16包
                print("step:{} train acc:{:.3f} train loss:{:.3f}".format(step,step_acc/len(labels),loss))
        one_epoch_time = time.perf_counter()-t1

    #验证集
        net.eval()  # 在测试过程中关掉dropout方法，不希望在测试过程中使用dropout
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            for data_test in validate_loader:#一个batch=16包，每一包16个标签图片
                test_images, test_labels = data_test
                test_labels_len = len(test_labels)
                outputs = net(test_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == test_labels.to(device)).sum().item()

            accurate_test = acc / val_num
            if accurate_test > best_acc:
                best_acc = accurate_test
                torch.save(net.state_dict(), save_path)
                with open(best_acc_path, 'w') as f:
                    json.dump({'best_acc': best_acc}, f)
            print('\n[epoch %d] trainset_acc:%.3f train_loss: %.3f  testset_accuracy: %.3f best_acc: %.3f one_epoch_time:%.3fs\n' %
                  (epoch + 1, tra_acc/tra_num, running_loss / step, accurate_test,best_acc,one_epoch_time))
            pre_write_txt("epoch:{} trainset_acc:{:.3f} train_loss:{:.3f} testset_accuracy: {:.3f} best_acc: {:.3f}".format(epoch + 1, tra_acc/tra_num, running_loss / step, accurate_test,best_acc), file = os.path.join(os.path.dirname(__file__), 'result.txt'))
        lr_scheduler.step()

    print('Finished Training')