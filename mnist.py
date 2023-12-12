import numpy as np
from tqdm import trange
import othergrad.nn as nn
from othergrad import Tensor
from othergrad.optim import SGD
from time import sleep

# Load data
X_train = np.load("./data/X_train.npy") 
Y_train = np.load("./data/Y_train.npy") 
X_test = np.load("./data/X_test.npy") 
Y_test = np.load("./data/Y_test.npy") 

# Ex: 6
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 10)
        self.fc2 = nn.Linear(10, 30)
        self.fc3 = nn.Linear(30, 40)
        self.fc4 = nn.Linear(40, 10)

    def __call__(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x).relu()
        return self.fc4(x)

model = MLP()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = SGD(
    model.parameters(), 
    lr=0.001, 
    momentum=0.9, 
    nesterov=True
)

EPOCHS = 100
BATCH = 128

batched_data = zip(
    np.array_split(X_train, X_train.shape[0]//BATCH), 
    np.array_split(Y_train, X_train.shape[0]//BATCH)
)

# Train model
model.train()

for _ in (prog_bar := trange(EPOCHS)):
    for X, Y in batched_data:
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    sleep(.001)

# Evaluate performance
model.eval()

train_accuracy = (model(X_train).softmax().data.argmax(axis=1) == Y_train).mean() * 100
test_accuracy = (model(X_test).softmax().data.argmax(axis=1) == Y_test).mean() * 100

print(f"{train_accuracy = } %")
print(f"{test_accuracy = } %")