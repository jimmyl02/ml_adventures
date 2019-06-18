# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 20:38:33 2019

@author: jimmy
"""

from os import path
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data
import torch.nn as nn

"""
SECTION
Name: Definitions
Description: A set of hyperparameters
"""

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

num_epochs = 500
num_features = 3
num_classes = 6
batch_size = 32
learning_rate = 0.001

column_names = [
    'user',
    'activity',
    'timestamp',
    'x-axis',
    'y-axis',
    'z-axis'
]

labels = [
    'Downstairs',
    'Jogging',
    'Sitting',
    'Standing',
    'Upstairs',
    'Walking'
]

# parameters to adjust based on performance
segment_time_size = 180
time_step = 100
hidden_size = 30
num_lstm_layers = 2

"""
SECTION
Name: Preprocessing
Description: Pre-process the data and load the information
"""

# Remove semicolons from the file and make a new file
    
with open(r'C:\Development\Machine Learning\PyTorch Learning\data\WISDM_ar_v1.1\WISDM_ar_v1.1_raw.txt', 'r') as infile:
    if not path.exists(r"C:\Development\Machine Learning\PyTorch Learning\data\WISDM_ar_v1.1\WISDM_ar_v1.1_prep.csv"):
        with open(r'C:\Development\Machine Learning\PyTorch Learning\data\WISDM_ar_v1.1\WISDM_ar_v1.1_prep.csv', 'w') as outfile:
            data = infile.read()
            data = data.replace(";", "")
            outfile.write(data)

thead = pd.read_csv("C:\\Development\\Machine Learning\\PyTorch Learning\\data\\WISDM_ar_v1.1\\WISDM_ar_v1.1_prep.csv", nrows=5) # just read in a few lines to get the column headers
dtypes = dict(zip(thead.columns.values, ['int32', 'float32', 'float64', 'float32', 'float32', 'bool']))   # datatypes as given by the data page
#NOTE: Limited as my training device doesn't have high amounts of memory
data = pd.read_csv("C:\\Development\\Machine Learning\\PyTorch Learning\\data\\WISDM_ar_v1.1\\WISDM_ar_v1.1_prep.csv", header=None, skiprows=0, nrows=100000, dtype=dtypes, names=column_names).dropna()

data_convoluted = []
labels = []

# Slide a "SEGMENT_TIME_SIZE" wide window with a step size of "TIME_STEP"
# Pre-processing code from https://github.com/bartkowiaktomasz/har-wisdm-lstm-rnns
for i in range(0, len(data) - segment_time_size, time_step):
    x = data['x-axis'].values[i: i + segment_time_size]
    y = data['y-axis'].values[i: i + segment_time_size]
    z = data['z-axis'].values[i: i + segment_time_size]
    data_convoluted.append([x, y, z])

    # Label for a data window is the label that appears most commonly
    label = stats.mode(data['activity'][i: i + segment_time_size])[0][0]
    labels.append(label)

# Convert to numpy
data_convoluted = np.asarray(data_convoluted, dtype=np.float32).transpose(0, 2, 1)

# Encode labels (cross entropy loss takes labels, not one-hot [mathematically same])
encoded_labels = LabelEncoder().fit_transform(labels)

"""
SECTION
Name: Dataloader and Model definition
"""

class wisdmDataset(object):
    
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
            return len(self.data)
        
    def __getitem__(self, index):
        features = self.data[index]
        label = self.labels[index]
        
        if self.transform is not None:
            features = self.transform(features)
        
        features = features.astype(np.float32)
        
        return torch.Tensor(features), label
    
# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers, num_classes):
        
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        
        return out
    
"""
SECTION
Name: Model creation and training
"""

# Model and dataloader creation

model = RNN(num_features, hidden_size, num_lstm_layers, num_classes).to(device)

# Split data into training and validation sets

X_train, X_test, y_train, y_test = train_test_split(data_convoluted, encoded_labels, test_size=0.2, random_state=seed)
print("X train size: ", len(X_train))
print("X test size: ", len(X_test))
print("y train size: ", len(y_train))
print("y test size: ", len(y_test))

train_dataset = wisdmDataset(X_train, y_train, transform=None)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = wisdmDataset(X_test, y_test, transform=None)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Loss and optimizer definition

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model

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
    
"""
SECTION
Name: Verification testing of the model
"""

# Test the model

with torch.no_grad():
    
    correct = 0
    total = 0
    
    for features, labels in test_loader:
        
        features = features.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        
        # Forward pass
        
        outputs = model(features)
        _, predicted = torch.max(outputs.data , 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print('Test Accuracy of the model on the data in the test_loader: {} %'.format(100 * correct / total))
    
    # Save the model weights with information in name
    
    torch.save(model.state_dict(), "./wisdm_weights/verif_" + str(round(100 * correct / total, 3)) + "acc")
