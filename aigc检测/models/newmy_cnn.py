import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedCNN(nn.Module):
    def __init__(self, num_classes):
        super(OptimizedCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)
        
        # 计算全连接层的输入特征数
        fc1_input_dim = 128 * 192  # 192是卷积层和池化层后的输出尺寸
        
        # 确保 fc1 的输入维度与实际特征图尺寸匹配
        self.fc1 = nn.Linear(fc1_input_dim, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # 将输入数据的形状调整为 [batch_size, 1, sequence_length]
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        
        x = x.view(x.size(0), -1)  # 展平
        
        
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x







