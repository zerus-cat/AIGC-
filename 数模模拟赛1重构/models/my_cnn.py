import torch.nn as nn
import torch.nn.functional as F

class MyCNN(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool1d(kernel_size=2)
        conv_output_size = embedding_dim - 4
        pool_output_size = (conv_output_size - 2) // 2 + 1
        self.fc1 = nn.Linear(64 * pool_output_size, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x




