from torch import nn
import torch

class FakeLeNet5(nn.Module):
    def __init__(self): # initialize tha layers that you are going to use
        super(FakeLeNet5, self).__init__()
        self.flatten = nn.Flatten()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 100) # 100 is the number of classes

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 5 * 5)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.log_softmax(x, dim = 1) # dim is the direction along which softmax is computed (row-wise)

        return x
    
class CharLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, output_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        #self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        #output=self.dropout(output) #add dropout for reducing overfitting
        output = self.fc(output[:, -1, :])  # Get the output from the last time step
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))