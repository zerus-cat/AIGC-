import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split

def load_data(embedding_path, label_path, batch_size, test_size=0.2):
    embeddings = np.load(embedding_path)
    labels = np.load(label_path)
    
    embedding_dim = embeddings.shape[1]
    
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=test_size, random_state=42)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, embedding_dim



