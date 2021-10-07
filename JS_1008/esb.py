import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

print("Device", device)

train_data = pd.read_csv("C:/Users/USER/PycharmProjects/A.I/-Project/-Hackerton/csv files/js_train_data.csv")
train_label = pd.read_csv("C:/Users/USER/PycharmProjects/A.I/-Project/-Hackerton/csv files/js_train_label.csv")
test_data = pd.read_csv("C:/Users/USER/Desktop/Hackerton/test_features.csv")

out_list = []
out_list2 = []
for i in range(15624):
    id = train_data.loc[(train_data['id'] == i)].values[:, 2:]
    out_list.append(id)
    out_list2.append(id)

out_list = np.expand_dims(np.array(out_list), axis=1)
data = torch.from_numpy(out_list)
out_list2 = np.array(out_list2).transpose((0,2,1))
data2 = torch.from_numpy(out_list2)

print("Data ok")
print("data shape check", data.shape)  # [15624, 6, 600]

# label data to tensor
out_list = []
for i in range(15624):
    label = train_label.loc[(train_label['index'] == i)].values[0, 2]
    out_list.append(label)

out_list = np.array(out_list)
label = torch.from_numpy(out_list)

print("label ok")
print("lable value check", label[0])
print("label shape check", label.shape)

# test data processing
out_list = []
out_list2 = []
for i in range(782):
    id = test_data.loc[(test_data['id'] == i + 3125)].values[:, 2:]
    out_list.append(id)
    out_list2.append(id)

out_list = np.expand_dims(np.array(out_list), axis = 1)
test_data = torch.from_numpy(out_list)
out_list2 = np.array(out_list2).transpose((0,2,1))
test_data2 = torch.from_numpy(out_list2)

print("test shape check", test_data.shape)

# train - valid dataset split
data, data2, valid_data = data[:14000], data2[:14000], data[14000:]
label, valid_label = label[:14000], label[14000:]

# Create model
model_1 = model1().to(device)
model_2 = model2().to(device)
model_3 = model3().to(device)

# model parameter
batch_size = 500
learning_rate = 0.6  # 다음엔 0.01 정도로
num_epochs = 200  # 다음엔 100 정도로


parameters = list(model_1.parameters()) + list(model_2.parameters()) + list(model_3.parameters())
optimizer = optim.SGD(parameters, lr=learning_rate)
loss = nn.CrossEntropyLoss()

# define data loader
print(data.shape)
print(data2.shape)
print(label.shape)
train_dataset = TensorDataset(data, label)
train_dataset2 = TensorDataset(data2, label)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
train_loader2 = DataLoader(dataset=train_dataset2, batch_size=batch_size, shuffle=True, drop_last=True)
print(" --- Ok before training --- ")

for epoch in range(num_epochs + 1):
    avg_cost = 0
    batch_length = len(train_loader)
    for (x, y), (x2, y2) in zip(train_loader, train_loader2):
        y = y.long().to(device)

        result = torch.stack([model_1(x2.float().to(device)) , model_2(x.float().to(device)), model_3(x.float().to(device))], axis = 1)
        result = F.softmax(result, dim = 1)
        result, index = torch.max(result, dim = 1)
        cost = loss(result, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        corr = torch.argmax(result)
        num_correct = (corr == y).sum().item()
        avg_cost += cost / batch_length
        acc = num_correct / batch_length
        batch_length = batch_length*3

    print(f"Epoch : {epoch} Correct {num_correct}/{batch_length} Avg Cost {avg_cost}")

print(" --- Train finished --- ")
