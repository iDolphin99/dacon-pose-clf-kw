# Classification 

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
  id = np.reshape(id, (-1,600*6))
  out_list.append(id)

out_list = np.array(out_list)
#out_list = np.expand_dims(np.array(out_list), axis = 1)
data = torch.from_numpy(out_list)  
data = data.squeeze(1)

out_list = []
for i in range(3125):
  label = train_label.loc[(train_label['id'] == i)].values[0,1] 
  out_list.append(label)

out_list = np.array(out_list)
#out_list = np.expand_dims(np.array(out_list), axis = 1)
label = torch.from_numpy(out_list)  

class Classify(nn.Module):

  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(3600, 1800)
    self.linear2 = nn.Linear(1800, 900)
    self.linear3 = nn.Linear(900, 300)
    self.linear4 = nn.Linear(300, 100)
    self.linear5 = nn.Linear(100, 61)

    #self.Softmax = nn.Softmax()

  def forward(self,x):

    out = F.relu(self.linear(x))
    out = F.relu(self.linear2(out))
    out = F.relu(self.linear3(out))
    out = F.relu(self.linear4(out))
    out = F.relu(self.linear5(out))
    # out = self.Softmax(out)

    return out
  
model = Classify()  

optimizer = optim.SGD(model.parameters(), lr = 0.01)
loss = nn.CrossEntropyLoss()
batch_size = 100
num_epochs = 10

from torch.utils.data import TensorDataset

train_dataset = TensorDataset(data, label)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)

for epoch in range(num_epochs+1):
  avg_cost = 0 
  batch_length = len(train_loader)
  for x, y in train_loader:  
    
    pred = model(x.float())
    print(pred.shape)
    print(y.shape)
    cost = loss(pred, y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    corr = torch.argmax(pred)
    num_correct = (corr == y).sum().item() 
    avg_cost += cost / batch_length     
    acc = num_correct/batch_length 
    
  print("Accuracy", acc*100)    
  print("Average Cost", avg_cost)

