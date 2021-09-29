# --------------------- #

# do augumentation here #

# 1. Jittering
# 2. Scaling 
# 3. Rolling 
# 4. Rotation 
# 5. Permutation

# --------------------- #

# link : https://www.slideshare.net/DonghyeonKim7/data-augmentation-for-1d

import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)

print("Device", device)  

# Load data
train_data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Hackerton/Dacon/train_features.csv")
train_label = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Hackerton/Dacon/train_labels.csv")

out_list = []
for i in range(3125):
  id = train_data.loc[(train_data['id'] == i)].values[:,2:]
  out_list.append(id)
  
out_list = np.expand_dims(np.array(out_list), axis = 1)
data = torch.from_numpy(out_list)  

out_list = []
for i in range(3125):
  label = train_label.loc[(train_label['id'] == i)].values[0,1] 
  out_list.append(label)

out_list = np.array(out_list)
label = torch.from_numpy(out_list)  

data , test = data[:2500] , data[2500:]
label, test_label = label[:2500] , label[2500:]

print(data.shape)
print(label.shape)
print(test.shape)
print(test_label.shape)

class Inception(nn.Module):
  def __init__(self, in_channel):
    super().__init__()
    # Block 1
    self.branch1_1 = nn.Conv2d(in_channel, 16, kernel_size = 3, stride = 1, padding = 1) 
    self.branch1_2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1) 

    # Block 2
    self.branch3_1 = nn.Conv2d(in_channel, 16, kernel_size = 1) # 1x1 Conv 
    self.branch3_2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1)
    self.branch3_3 = nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1)

    # Block 3
    self.branch_pool = nn.Conv2d(in_channel, 32, kernel_size = 3, stride = 1,  padding = 1)

  def forward(self, x):

   # print("Input shape : ", np.shape(x)) # torch.Size([100, 8, 301, 4])

    branch1x1 = self.branch1_1(x)
    branch1x1 = self.branch1_2(branch1x1)

    branch3x3 = self.branch3_1(x)
    branch3x3 = self.branch3_2(branch3x3)
    branch3x3 = self.branch3_3(branch3x3)

    branch_pool = F.avg_pool2d(x, kernel_size = 3, stride = 1, padding = 1)
    branch_pool = self.branch_pool(branch_pool)

    '''
            branch1x1   : torch.Size([100, 32, 301, 4])
            branch3x3   : torch.Size([100, 32, 301, 4])
            branch_pool : torch.Size([100, 32, 301, 4])
    '''
    # 3개의 output들을 1개의 list로
    outputs = [branch1x1, branch3x3, branch_pool] # np.shape(outputs)) (3,)
    
    # torch.cat (concatenate)
    cat = torch.cat(outputs,  1) # outputs list의 tensor들을 dim = 1로 이어준다.

    #  cat.shape : torch.Size([300, 32, 301, 4])
    return cat    
  
class Classification(nn.Module):
  def __init__(self):
    super().__init__()

    self.Conv1 = nn.Conv2d(1, 8, kernel_size = 1, stride = 1, padding = 1)
    self.Conv2 = nn.Conv2d(96, 16, kernel_size = 1, stride = 1, padding = 1)
    
    self.Incept1 = Inception(in_channel = 8)
    self.Incept2 = Inception(in_channel = 16)

    self.mp1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
    self.mp2 = nn.MaxPool2d(kernel_size = 3, stride = 1)

    self.fc1 = nn.Linear(96*301*4, 3000)
    self.fc2 = nn.Linear(3000, 100)
    self.fc3 = nn.Linear(100, 61)

  def forward(self, x):
    
    #print("input shape", np.shape(x))

    out = self.Conv1(x) # out_channel = 8
    out = F.relu(self.mp1(out))
    out = self.Incept1(out) # out_channel = 96

    #print("Inception 2 out shape : ", out.shape) # [300, 32, 301, 4]

    out = self.Conv2(out) # out_channel = 16
    out = F.relu(self.mp2(out))
    out = self.Incept2(out) # out_channel = 96

    #print("Inception 2 out shape : ", out.shape) # [900, 32, 301, 4]

    out = out.view(-1, 96*301*4)
    
    #print("out shape  3 : ", out.shape) # [900, 38528]

    out = F.relu(self.fc1(out))

    #print("out shape  4 : ", out.shape)

    out = F.relu(self.fc2(out))

    #print("out shape  5 : ", out.shape)

    out = F.relu(self.fc3(out))

   # print("out shape  6 : ", out.shape)

    return out  
  
model = Classification()  

batch_size = 100
learning_rate = 0.01
num_epochs = 10 # 87 

optimizer = optim.SGD(model.parameters(), lr = learning_rate)
loss = nn.CrossEntropyLoss()

from torch.utils.data import TensorDataset

train_dataset = TensorDataset(data, label)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)

for epoch in range(num_epochs+1):
  avg_cost = 0 
  batch_length = len(train_loader)
  for x, y, in train_loader:    
    
    pred = model(x.float())   # 100 x 61        

    cost = loss(pred, y)
    
    optimizer.zero_grad() 
    cost.backward()       
    optimizer.step()      

    corr = torch.argmax(pred)
    num_correct = (corr == y).sum().item() 
    avg_cost += cost / batch_length     
    acc = num_correct/batch_length 
    
  print("Accuracy", acc)    
  print("Average Cost", avg_cost)
  
with torch.no_grad(): # Gradient 학습 x 

    prediction = model(test.float())
    print(prediction.shape)
    correct_prediction = torch.argmax(prediction, 1) == test_label
    print(correct_prediction)
    accuracy = correct_prediction.float().mean()    
    print('Accuracy:', accuracy.item())  
    
print("check 1", test_label)
print("check 2", torch.argmax(prediction, 1))    
