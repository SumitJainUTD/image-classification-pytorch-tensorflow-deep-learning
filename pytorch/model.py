import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class SimpleCNN(nn.Module):
    def __init__(self, data_classes_len):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, data_classes_len)  # Number of classes

        self.model_folder_path = './model'

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 30 * 30)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Input: torch.Size([1, 3, 128, 128])
# After conv1: torch.Size([1, 16, 126, 126])
# After max_pool2d 1: torch.Size([1, 16, 63, 63])
# After conv2: torch.Size([1, 32, 61, 61])
# After max_pool2d 2: torch.Size([1, 32, 30, 30])
# After flattening: torch.Size([1, 28800])
# After fc1: torch.Size([1, 128])
# model = SimpleCNN(len(train_dataset.classes))

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, model, file_name='model.pth'):

        file_path = os.path.join(self.model_folder_path, file_name)
        if os.path.exists(file_path):
            model.load_state_dict(torch.load(file_path))
            print("Model loaded.")