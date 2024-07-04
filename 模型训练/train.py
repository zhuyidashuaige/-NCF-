import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

# 加载预处理后的数据
features = np.load('features.npy')
game_names = pd.read_csv('game_names.csv')['游戏名字'].tolist()

# 定义简单的神经网络模型
class NCFmodel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NCFmodel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练函数
def train_model(features, num_epochs=2000, batch_size=32, learning_rate=0.001, weight_decay=0.0001, step_size=100, gamma=0.1):
    # 转换为Tensor
    features = torch.tensor(features, dtype=torch.float32)

    # 创建数据加载器
    dataset = TensorDataset(features, features)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型初始化
    input_dim = features.shape[1]
    hidden_dim = 128  # 可以调整
    model = NCFmodel(input_dim, hidden_dim)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)

    scheduler =StepLR(optimizer,
             step_size=step_size,
             gamma=gamma)

    # 训练循环
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_features, _ in dataloader:
            # 前向传播
            outputs = model(batch_features)
            loss = criterion(outputs, batch_features)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 更新学习率
        scheduler.step()

        # 打印每个 epoch 的损失
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

    # 保存模型
    torch.save(model.state_dict(), 'game_recommendation_model.pth')
    print('Model training completed and saved.')

# 执行训练
train_model(features)
