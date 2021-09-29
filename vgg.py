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
  out_list.append(id)

out_list = np.expand_dims(np.array(out_list), axis = 1)
data = torch.from_numpy(out_list)  

out_list = []
for i in range(3125):
  label = train_label.loc[(train_label['id'] == i)].values[0,1] 
  out_list.append(label)

out_list = np.array(out_list)
#out_list = np.expand_dims(np.array(out_list), axis = 1)
label = torch.from_numpy(out_list)  

# split train , test data
data , test = data[:2500] , data[2500:]
label, test_label = label[:2500] , label[2500:]

class VGG_ORG(nn.Module):
    def __init__(self, in_channel): 
        super().__init__()
        
        self.layer1 = nn.Sequential(           # 1 x 600 x 6 
        nn.Conv2d(in_channel, 16, kernel_size = 3, stride = 1, padding = 1),    
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)            
        )

        self.layer2 = nn.Sequential(            # 32 x 300 x 3
        nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),    
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 3, stride = 1)            
        )
                        
        self.fc1 = nn.Linear(64*298, 100) # 512 자리에 512 x width x heigth 넣어주기 
        self.fc2 = nn.Linear(100, 61)

        #self.last = nn.Softmax()

    def forward(self, x):
        
        # Conv layer
        out = self.layer1(x)           # out shape torch.Size([100, 32, 300, 3])
        out = self.layer2(out)         # out shape torch.Size([100, 64, 298, 1]) 

        # flatten      
        out = out.view(-1, 64*298)

        # fc layer    
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        return out
   
model = VGG_ORG(in_channel=1)

batch_size = 100
learning_rate = 0.01
num_epochs = 30 # 87 

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
    correct_prediction = torch.argmax(prediction, 1) == test_label
    accuracy = correct_prediction.float().mean()    
    print('Accuracy:', accuracy.item())  
    
print("check 1", test_label[:20])
print("check 2", torch.argmax(prediction, 1)[:20])    
