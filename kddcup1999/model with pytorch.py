# -*- coding: utf-8 -*-
"""
PyTorch implementation of simple feed forward neural network for kddCup dataset

Author: Jimmy Li
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
import torch.utils.data
import torch.nn as nn

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"""
General definitions
"""

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#Hyper parameters
num_epochs = 5
cont_input_size = 36
num_classes = 20
batch_size = 100
learning_rate = 0.001
emb_dropout = 0.2

#Defining which columns are categories
cat_features = [1,2,3]
gview = ""

"""
Dataset schema
Represents the kddCup dataset
"""

class kddCupDataset(object):
    
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
            return len(self.data)
        
    def __getitem__(self, index):
        features = self.data.values[index, 0:40]
        label = self.data.values[index, 41]
        
        if self.transform is not None:
            features = self.transform(features)
        
        features = features.astype(np.float32)
        
        return torch.Tensor(features), label

"""
Model schema
Simple feed forward artificial neural network
"""

class FeedForward(nn.Module):
    def __init__(self, emb_dims, num_classes=20):
        super(FeedForward, self).__init__()
        
        #Make a list of embedding layers per cat feature
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
    
        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
    
        self.first_bn = nn.BatchNorm1d(cont_input_size)
        
        self.layer1 = nn.Sequential( #To determine number oif input, always (batch_size, dims)
                nn.Linear(76, 80),
                nn.ReLU()
                )
        self.layer2 = nn.Sequential(
                nn.Linear(80, 160),
                nn.ReLU(),
                nn.Dropout(0.2))
        self.layer3 = nn.Sequential(
                nn.Linear(160, 160),
                nn.ReLU(),
                nn.Dropout(0.2))
        self.layer4 = nn.Sequential(
                nn.Linear(160, 160),
                nn.ReLU(),
                nn.Dropout(0.2))
        self.layer5 = nn.Sequential(
                nn.Linear(160, 80),
                nn.ReLU(),
                nn.Dropout(0.2))
        self.layer6 = nn.Sequential(
                nn.Linear(80, 40),
                nn.ReLU())
        self.layer7 = nn.Sequential(
                nn.Linear(40, num_classes),
                nn.Softmax())
        
    def forward(self, x):
        #Categorial features get their own embed layers which are concat at the end w/ cont features
        embedData = [emb_layer(x[:, cat_features[i]].long()) for i, emb_layer in enumerate(self.emb_layers)]
        embedData = torch.cat(embedData, 1)
        embedData = self.emb_dropout_layer(x)
        
        x = x[:, 4:] #TODO: Lost the leftmost data, next time split data into cont and cat when passing into model        
        out = self.first_bn(x)
        
        out = torch.cat([out, embedData], 1)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        
        return out

"""
Data proprocessing
"""

#Load the data
#Efficient low memory loading
thead = pd.read_csv("C:\\Users\\jimmy\\Desktop\\kddcup.data.corrected", nrows=5) # just read in a few lines to get the column headers
dtypes = dict(zip(thead.columns.values, ['int32', 'float32', 'float64', 'float32', 'float32', 'bool']))   # datatypes as given by the data page
data = pd.read_csv("C:\\Users\\jimmy\\Desktop\\kddcup.data.corrected", header=None, skiprows=1, nrows=5000, dtype=dtypes).dropna()
    
#Encode labels
label_encoders = {}

#Encode the categorized columns
for i in cat_features:
    label_encoders[i] = LabelEncoder()
    data[i] = label_encoders[i].fit_transform(data[i])
    
#Encode the output label
label_encoders[41] = LabelEncoder()
data[41] = label_encoders[41].fit_transform(data[41])

#Storing embedding data
cat_dims = [int(data[col].nunique()) for col in cat_features]
emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

"""
Model and data definitions
"""

model = FeedForward(emb_dims, num_classes).to(device)

train_dataset = kddCupDataset(data, transform=None)
    
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

"""
Loss and optimizer
"""

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

"""
Training the model
"""

total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        
        features = features.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        
        #Forward pass
        outputs = model(features)
        loss = loss_fn(outputs, labels)
        
        #Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 5 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
