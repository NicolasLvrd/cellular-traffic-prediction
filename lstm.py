import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import scipy.io
import numpy as np
import datetime

# Dataset class for .mat files
class MatDataset(Dataset):
    def __init__(self, data, targets, seq_length):
        self.data = data
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output

# Load data from .csv
data = np.genfromtxt("datasets/data11Febto13Feb.csv", delimiter=",", dtype=str)

# Extract features: [index, date, ip_source, ip_destination, protocol, packet_size]
data = data[:, 1:]  # Ignore the index column

# Convert date to numerical values (timestamp)
def convert_date_to_timestamp(date_str):
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f")
    return dt.timestamp()

data[:, 0] = np.array([convert_date_to_timestamp(str(date)) for date in data[:, 0]])

# Convert date and packet_size columns to float
data[:, 0] = data[:, 0].astype(float)  # Ensure timestamp is float
data[:, 4] = data[:, 4].astype(float)  # Ensure packet_size is float


# Encode protocol using one-hot encoding
unique_protocols = list(set(data[:, 3]))
protocol_to_index = {protocol: i for i, protocol in enumerate(unique_protocols)}
# replace protocol with index
data[:, 3] = np.array([protocol_to_index[protocol] for protocol in data[:, 3]])

# convert protocol to float
data[:, 3] = data[:, 3].astype(float)

# remove . and : from ip source and ip destination and convert to float
data[:, 1] = np.array([ip.replace(".", "") for ip in data[:, 1]])
data[:, 2] = np.array([ip.replace(".", "") for ip in data[:, 2]])
data[:, 1] = np.array([ip.replace("::", "") for ip in data[:, 1]])
data[:, 2] = np.array([ip.replace("::", "") for ip in data[:, 2]])
data[:, 1] = np.array([ip.replace(":", "") for ip in data[:, 1]])
data[:, 2] = np.array([ip.replace(":", "") for ip in data[:, 2]])


# if one of ip is '', delete the row
data = data[data[:, 1] != '']
data = data[data[:, 2] != '']

# convert letters to decimal
data[:, 1] = np.array([int(ip, 16) for ip in data[:, 1]])
data[:, 2] = np.array([int(ip, 16) for ip in data[:, 2]])



# convert ips to float
data[:, 1] = data[:, 1].astype(float)
data[:, 2] = data[:, 2].astype(float)

# convert data to float
data = data.astype(float)

# Normalize data (each column)
for i in range(data.shape[1]):
    data[:, i] = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])

    

# Define targets (example: use date as target)
targets = data[:, 0]
targets = targets.astype(float)

# Hyperparameters
seq_length = 20
input_size = 5  # Exclude target column
hidden_size = 50
num_layers = 2
output_size = 1
batch_size = 32
num_epochs = 50
learning_rate = 0.001

# Split data into train and test sets
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]
train_targets, test_targets = targets[:train_size], targets[train_size:]

# Create datasets and data loaders
train_dataset = MatDataset(train_data, train_targets, seq_length)
test_dataset = MatDataset(test_data, test_targets, seq_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
test_loss = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "lstm_model.pth")
print("Model saved as lstm_model.pth")

# Load the model
loaded_model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
loaded_model.load_state_dict(torch.load("lstm_model.pth"))
loaded_model.eval()
print("Model loaded successfully.")
