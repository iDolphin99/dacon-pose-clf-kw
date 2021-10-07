'''
 # 1D Convolution : 한 방향으로만 이동하며 Convolution 수행

 conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
 
    dilation : filter 내부에서 얼마만큼 띄어서 filter를 적용할 것인지
    groups : filter의 height 조절
 
 
 input : batch size x feature dimension x time_step
 
 2 차원 이미지의 경우 : channel , height, width  
 1 차원 데이터의 경우 : features , time_step  ---> feature가 in_channle이 된다.

 2D Conv의 경우 filter size를 설정하면 height x width의 사각형 filter가 생성
 1D Conv의 경우 width = input_channle, height = filter size. out_channel 만큼 결과 생성
    
'''

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

'''
# batch size : 10
# feature : 6
# time_step : 100 (time step? height 같다)
input = torch.randn(300,6,600, requires_grad = True)
label = torch.randn(1, 61, requires_grad = True)
print("input shape", input.shape)
'''

train_data = pd.read_csv("C:/Users/USER/PycharmProjects/A.I/-Project/-Hackerton/csv files/js_train_data.csv")
train_label = pd.read_csv("C:/Users/USER/PycharmProjects/A.I/-Project/-Hackerton/csv files/js_train_label.csv")
test_data = pd.read_csv("C:/Users/USER/Desktop/Hackerton/test_features.csv")

out_list = []
for i in range(15624):
    id = train_data.loc[(train_data['id'] == i)].values[:, 2:]
    out_list.append(id)

out_list = np.array(out_list).transpose((0,2,1))
data = torch.from_numpy(out_list)

print("Data ok")
print("data shape check", data.shape) # [15624, 6, 600]

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

out_list = np.array(out_list).transpose((0,2,1))
test_data = torch.from_numpy(out_list)

print("test shape check", test_data.shape)

# train - valid dataset split
data, valid_data = data[:14000], data[14000:]
label, valid_label = label[:14000], label[14000:]

'''
# kernel_size = 내가 time_step을 얼마만큼 볼 것인지 (filter의 height)
# stride = 내가 time_step을 얼만큼 띄워가며 볼 것 인지
m = nn.Conv1d(in_channels=6 , out_channels= 32, kernel_size = 1, stride = 1)
m2 = nn.Conv1d(in_channels=6 , out_channels= 32, kernel_size = 2, stride = 1)
m3 = nn.Conv1d(in_channels=6 , out_channels= 32, kernel_size = 10, stride= 1)

output = m(input)
output2 = m2(input)
output3 = m3(input)
print(output.shape)
print(output2.shape)
print(output3.shape)
'''


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = torch.nn.Conv1d(in_channels = 6, out_channels = 20, kernel_size = 2, stride = 2)
        self.bn1 = torch.nn.BatchNorm1d(20)
        self.layer2 = torch.nn.Conv1d(in_channels = 20, out_channels = 100, kernel_size= 2, stride = 2)
        self.bn2 = torch.nn.BatchNorm1d(100)
        self.layer3 = torch.nn.Conv1d(in_channels = 100, out_channels = 30, kernel_size= 2, stride = 2)
        self.bn3 = torch.nn.BatchNorm1d(30)

        self.activation = torch.nn.ReLU()

        self.fc = nn.Linear(30*75, 61)
        self.bn4= torch.nn.BatchNorm1d(61)


    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.layer3(x)
        x = self.activation(x)
        x = self.bn3(x)

        x = x.view(-1, 30 * 75)
        x = self.fc(x)
        x = self.bn4(x)
        x = self.activation(x)

        return x

# Create model
model = SimpleCNN().to(device)

# model parameter
batch_size = 64
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
submission.to_csv('js_submission10_07_1.csv', index=False)

print(" --- Save model --- ")
torch.save(model.state_dict(), "csv files/saved_model10_07_2.pt")







