import torch
import torch.nn as nn

class OCRNetwork(nn.Module):
   def __init__(self, height, width, n_classes):
      """
      OCRNetwork:
      takes in height, width, n_classes
      """
      super(OCRNetwork, self).__init__()
      # first convolutional block
      self.conv1= nn.Conv2d(1, 32, (3, 3), padding=(1, 1))
      self.max_pool1 = nn.MaxPool2d((2, 2))
      self.activation = nn.ReLU()

      # second convolutional block
      self.conv2= nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
      self.max_pool2 = nn.MaxPool2d((2, 2))

      # need to reshape the downsampled output
      # so that you can pass it to the LSTM
      self.new_shape = ((width//2), (height//2) * 64) # output how many features?
      self.dense1 = nn.Linear((height//2)*64, 64)
      self.dropout1 = nn.Dropout(0.2)

      # rnns
      self.rnn1 = nn.LSTM(64, 128, bidirectional=True)
      self.rnn2 = nn.LSTM(128, 64, bidirectional=True)
      self.dropout2 = nn.Dropout(0.25)

      # output layer
      self.output = nn.Linear(64, n_classes)
      self.softmax = nn.Softmax(1)
   
   def forward(self, x: torch.Tensor) -> torch.Tensor:
      # convolutional block 1
      out = self.conv1(x)
      out = self.activation(out)
      out = self.max_pool1(out)

      # convolutional block 2
      out = self.conv2(x)
      out = self.activation(out)
      out = self.max_pool2(out)

      # reshaping for rnns
      out = torch.reshape(out, self.new_shape)
      out = self.dense1(out)
      out = self.dropout1(out)

      # rnns
      out = self.rnn1(out)
      out = self.dropout2(out)
      out = self.rnn2(out)
      out = self.dropout2(out)

      # output layer
      out = self.output(out)
      out = self.softmax(out)

      return out

# hyperparameters
height = 50
width  = 200
n_classes = 19

model = OCRNetwork(height, width, n_classes)
print(model)
