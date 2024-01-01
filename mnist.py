import numpy as np
from tqdm import trange
import othergrad.nn as nn
from othergrad.optim import SGD
from time import sleep

# Load data
X_train = np.load("./data/X_train.npy") 
Y_train = np.load("./data/Y_train.npy") 
X_test = np.load("./data/X_test.npy") 
Y_test = np.load("./data/Y_test.npy") 


# Ex: 6 TODO
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 1)
        self.fc2 = nn.Linear(1, 10)

    def __call__(self, x):
        x = self.fc1(x).relu()
        return self.fc2(x)


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

accuracy = lambda m, x, y: (m(x).log_softmax().exp().data.argmax(axis=1) == y).mean()
train_accuracy = accuracy(model, X_train, Y_train) * 100
test_accuracy = accuracy(model, X_test, Y_test) * 100 

print(f"train_accuracy = {train_accuracy:.2f} %")
print(f"test_accuracy = {test_accuracy:.2f} %")