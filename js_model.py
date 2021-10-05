import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)

print("Device", device)

# 증강 데이터 load
train_data = pd.read_csv("C:/Users/USER/PycharmProjects/A.I/-Project/-Hackerton/csv files/js_train_data.csv")
train_label = pd.read_csv("C:/Users/USER/PycharmProjects/A.I/-Project/-Hackerton/csv files/js_train_label.csv")
test_data = pd.read_csv("C:/Users/USER/Desktop/Hackerton/test_features.csv")

print("Data load ok")

# ------------------ 데이터 600 x 6 shape으로 -------------------- #
# sensor data to tensor
out_list = []
for i in range(15624):
    id = train_data.loc[(train_data['id'] == i)].values[:, 2:]
    out_list.append(id)

out_list = np.expand_dims(np.array(out_list), axis=1)
data = torch.from_numpy(out_list)

print("Data ok")
print("data shape check", data.shape)

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
for i in range(782):
  id = test_data.loc[(test_data['id'] == i + 3125)].values[:,2:]
  out_list.append(id)

out_list = np.expand_dims(np.array(out_list), axis = 1)
test_data = torch.from_numpy(out_list)

print("test shape check", test_data.shape)

# train - valid dataset split
data, valid_data = data[:14000], data[14000:]
label, valid_label = label[:14000], label[14000:]

# --------------------------------------- VGG model ------------------------- #
class VGG_ORG(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.layer1 = nn.Sequential(  # 1 x 600 x 6
            nn.Conv2d(in_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(  # 32 x 300 x 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )

        self.fc1 = nn.Linear(64 * 298, 100)  # 512 자리에 512 x width x heigth 넣어주기
        self.fc2 = nn.Linear(100, 61)

    def forward(self, x):
        # Conv layer
        out = self.layer1(x)  # out shape torch.Size([100, 32, 300, 3])
        out = self.layer2(out)  # out shape torch.Size([100, 64, 298, 1])

        # flatten
        out = out.view(-1, 64 * 298)

        # fc layer
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        return out

# ----------------------- Model --------------------------- #
model = VGG_ORG(in_channel=1).to(device)

# model parameter
batch_size = 100
learning_rate = 0.01 # 다음엔 0.01 정도로
num_epochs = 150     # 다음엔 100 정도로

optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()


# define data loader
train_dataset = TensorDataset(data, label)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

print(" --- Ok before training --- ")

for epoch in range(num_epochs + 1):
  avg_cost = 0
  batch_length = len(train_loader)
  for x, y, in train_loader:

    y = y.long().to(device)

    pred = model(x.float().to(device))  # 100 x 61

    cost = loss(pred, y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    corr = torch.argmax(pred)
    num_correct = (corr == y).sum().item()
    avg_cost += cost / batch_length
    acc = num_correct / batch_length

  print(f"Epoch : {epoch} Correct {num_correct}/{batch_length} Avg Cost {avg_cost}")

print(" --- Train finished --- ")

print(" --- Validate model --- ")
with torch.no_grad():  # Gradient 학습 x

  valid_data = valid_data.float().to(device)
  valid_label = valid_label.long().to(device)

  prediction = model(valid_data)
  print(prediction.shape)
  correct_prediction = torch.argmax(prediction, 1) == valid_label
  accuracy = correct_prediction.float().mean()
  print('Accuracy:', accuracy.item())

print("check 1", valid_label[:30])
print("check 2", torch.argmax(prediction, 1)[:30])

print(" --- Test model --- ")

with torch.no_grad():  # Gradient 학습 x

  test_data = test_data.float().to(device)

  prediction = model(test_data)
  prediction = F.softmax(prediction)
  print(prediction.shape)

prediction = prediction.detach().cpu().numpy()
submission = pd.read_csv('C:/Users/USER/Desktop/Hackerton/sample_submission.csv')

submission.iloc[:, 1:] = prediction
submission.to_csv('js_submission10_05_2.csv', index=False)

print(" --- Save model --- ")
torch.save(model.state_dict(), "saved_model.pt")
