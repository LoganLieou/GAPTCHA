from data import X, y, symbols, inverse_map

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProxyModel(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(ProxyModel, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional = True)
        self.embedding = nn.Linear(nHidden * 2, nOut)
    def forward(self, x):
        rec, _ = self.rnn(x)
        T, b, h = rec.size()
        t_rec = rec.view(T*b, h)

        output = self.embedding(t_rec)
        output = output.view(T, 5, -1)
        return output

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(2, 2), padding=(1, 1), stride=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(2, 2), padding=(1, 1), stride=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=(2, 2), padding=(1, 1), stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2))
        )

        self.rnn1 = ProxyModel(200, 20, 20)
        self.rnn2 = ProxyModel(200, 20, 100)
        self.soft = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #outs = [self.fc1(self.conv1(x)) for i in range(5)]
        self.conv1(x)
        out = torch.squeeze(x, 0)
        out = self.rnn1(out)
        out = self.rnn2(out)
        out = self.soft(out)
        # out = self.fc1(out)
        return out

model = Network()
print(model)

# test forward pass
apass = model.forward(torch.Tensor(X[0][None, :, :, :]))
print(apass.shape)

X_train, X_test, y_train, y_test = X[:-10], X[846:], y[:-10], y[846:]
print(len(y_test))

# loss fn
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# train the model
for epoch in range(2):
    running_loss = 0.0
    for i in range(len(X_train) - 1):
        optimizer.zero_grad()
        output = model.forward(torch.Tensor(X_train[i][None, :, :, :]))
        # print((output > 1).sum())
        loss = loss_fn(output, torch.Tensor(y_train[i][None, :, :]))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # every 100 mini batch
        if i % 100 == 99:
            print("current running loss: ", running_loss)
            running_loss = 0.0

def test(model, X_test, y_test):
    # test the model
    with torch.no_grad():
        correct = 0
        length = 0
        for i in range(len(X_test)):
            pred = model.forward(torch.Tensor(X_test[i][None, :, :, :]))
            label = torch.Tensor(y_test[i][None, :, :])

            pred = torch.round(pred)
            correct += (pred == label).sum().item()
            length += len(label)

        score = correct / length

        print(score)
        return score

test(model, X_test, y_test)

with torch.no_grad():
    final = model.forward(torch.Tensor(X[1][None, :, :, :]))
    thresh = nn.Threshold(0.5, 0)
    final = thresh(final)
    final = np.array(torch.round(final))
    print(final.shape)
    print(final)
    print(inverse_map(final[0]))
