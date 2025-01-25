# predict.py

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import os

# 导入模型定义，根据需要修改
from models.my_cnn import MyCNN
from models.SimpleNN import SimpleNN
from models.newmy_cnn import OptimizedCNN

def load_data(embeddings_path, labels_path):
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    return embeddings, labels

def predict(model, data_loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for data in data_loader:
            inputs = data[0].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
    return np.array(all_preds)

def write_results(filename, model_name, accuracy):
    with open('result.txt', 'a') as f:
        f.write(f'{filename} - {model_name} - Accuracy: {accuracy * 100:.2f}%\n')

def main(embeddings_path, labels_path, model_path, model_name):
    # 加载数据
    embeddings, labels = load_data(embeddings_path, labels_path)

    # 创建数据集和数据加载器
    tensor_data = TensorDataset(torch.tensor(embeddings, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    data_loader = DataLoader(tensor_data, batch_size=32, shuffle=False)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name=='SimpleNN':
        model = SimpleNN(768,2)  
    elif model_name=='MyCNN':
        model=MyCNN(768,2)
    else:
        model=OptimizedCNN(2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 预测
    predictions = predict(model, data_loader, device)

    # 计算准确率
    accuracy = accuracy_score(labels, predictions)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # 写入结果
    embeddings_filename = os.path.basename(embeddings_path)
    write_results(embeddings_filename, model_name, accuracy)

if __name__ == '__main__':
    # 文件路径，根据需要修改
    embeddings_path = './processed_data/proc_method5/LLM_specific_Testing/ZPQY_embedding_m3e_proc5.npy'
    labels_path = './processed_data/proc_method5/LLM_specific_Testing/ZPQY_label_pred_proc5.npy'
    model_name = 'OptimizedCNN' #可以修改成MyCNN，SimpleNN，OptimizedCNN
    model_path = f'./models/save_models/{model_name}_model.pth'
    main(embeddings_path, labels_path, model_path, model_name)
