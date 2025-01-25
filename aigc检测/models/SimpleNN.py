import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)
