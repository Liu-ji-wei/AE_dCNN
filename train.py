import torch
import numpy as np
from model import CNNMLP
import matplotlib
from dataloader import MyDataset, testDataset
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torch.nn as nn
if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 24})  # 改变所有字体大小，改变其他性质类似
    ####################################################
    #超参数
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    num_epochs =500
    model = CNNMLP(250)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0.01)
    best_acc = 0.0
    ####################################################
    model_accuracies = []
    model_loss = []
    model_loss_test = []
    transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为张量

         ])

    train_data = MyDataset(data_path='E:/独立脑电数据/train/S1/1.0s')
    val_data = testDataset(data_path='E:/独立脑电数据/test/S1/1.0s')  # 1*9*time

    # 创建 DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    predictions = []  # 混淆矩阵列表
    targets = []  # 混淆矩阵列表
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        for features, labels in train_loader:
            features_1 = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(features_1)
            predicted_labels = torch.argmax(outputs, dim=1)  # 获取最大概率的类别索
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)  # 选择最大值作为预测类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        model_loss.append(running_loss/len(train_loader))

        accuracy = correct / total *100
        model_accuracies.append(accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        print(f'Train Accuracy: {accuracy:.4f}')

        # 在测试集上评估模型
        last_hidden_outputs = []
        model.eval()
        targets_tsne = []
        with torch.no_grad():
            correct = 0
            total = 0
            test_loss = 0.0
            for features, labels in test_loader:
                features_1 = features.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(features_1)
                last_hidden_outputs.append(outputs.cpu().numpy())
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)  # 选择最大值作为预测类别
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                predictions.extend(predicted.tolist())
                targets_tsne.extend(labels.tolist())
            model_loss_test.append(test_loss / len(test_loader))
            accuracy = correct / total *100
            print(f'Test Accuracy: {accuracy:.4f}')
        last_hidden_output = np.concatenate(last_hidden_outputs, axis=0)
    #############################################################
