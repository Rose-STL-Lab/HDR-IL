#Neural Network
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class Net(nn.Module):

  def __init__(self, input_size, output_size, hidden_size):
    super(Net, self).__init__()

    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, output_size)


  #forward propogation function
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


  def train(self, features, labels):

    features = features.float()
    labels = labels.float()


    criterion = nn.MSELoss()
    learning_rate = .05
    optimizer = optim.Adam(self.parameters(), lr = .00005)

    #set gradient to 0
    self.zero_grad()


    output = self(features)

    loss = criterion(output, labels)



    loss.backward()

    optimizer.step()


    return output, loss.item()


  def evaluate(self, features, labels):
    features = features.float()
    labels = labels.float()
    criterion = nn.MSELoss()
    output = self(features)


    loss = criterion(output, labels)

    return loss.item()
