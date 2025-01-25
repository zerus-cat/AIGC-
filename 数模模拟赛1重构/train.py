import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import time
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from models.SimpleNN import SimpleNN
from models.my_cnn import MyCNN
from models.newmy_cnn import OptimizedCNN
from utils.early_stopping import EarlyStopping
from utils.data_loader import load_data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve



# 定义超参数
num_classes = 2            # 类别数（二分类问题）
num_epochs = 20            # 训练轮数
batch_size = 32            # 批量大小
sequence_length=100        # 数据沿序列维度的长度
# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
train_loader, test_loader, embedding_dim = load_data('./processed_data/proc_method5/Train_data/text_embedding_m3e_proc5.npy', './processed_data/proc_method5/Train_data/text_label_proc5.npy',batch_size = batch_size)
# # 打印输入数据形状以进行调试
# for inputs, targets in train_loader:
#     print(f"Input shape: {inputs.shape}")
#     break
# 选择模型
model_name = 'OptimizedCNN'  # 可选 'MyCNN' 'SimpleNN' 'OptimizedCNN'
if model_name == 'MyCNN':
    model = MyCNN(embedding_dim=embedding_dim, num_classes=num_classes)
elif model_name=='SimpleNN':
    model=SimpleNN(embedding_dim=embedding_dim,num_classes=num_classes)
elif model_name=='OptimizedCNN':
    model=OptimizedCNN(num_classes=num_classes)
else:
    raise ValueError(f"Unknown model name: {model_name}")

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# 初始化早期停止
early_stopping = EarlyStopping(patience=5, delta=0.01)

# 用于存储每个 epoch 的指标
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# 训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Iter: {batch_idx+1}, Loss: {running_loss/(batch_idx+1):.4f}')
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    val_loss /= len(test_loader)
    val_acc = 100 * correct / total
    
    # 记录每个 epoch 的指标
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Time: {time.time() - start_time:.2f}s')
    
    scheduler.step()
    
    early_stopping(val_loss)
    if early_stopping.should_stop:
        print("Early stopping")
        break

# 完成训练后打印评估指标
print("训练完成，计算评估指标...")

# 保存模型
torch.save(model.state_dict(), f'./models/save_models/{model_name}_model.pth')

# 计算额外的评估参数
precision = precision_score(all_targets, all_predictions, average='weighted')
recall = recall_score(all_targets, all_predictions, average='weighted')
f1 = f1_score(all_targets, all_predictions, average='weighted')
report = classification_report(all_targets, all_predictions)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print('Classification Report:')
print(report)



# 绘制训练和验证损失图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# 绘制训练和验证准确率图
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.tight_layout()
plt.show()

# 混淆矩阵
cm = confusion_matrix(all_targets, all_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ROC 曲线
fpr, tpr, _ = roc_curve(all_targets, all_predictions)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall 曲线
precision, recall, _ = precision_recall_curve(all_targets, all_predictions)
plt.figure()
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()


