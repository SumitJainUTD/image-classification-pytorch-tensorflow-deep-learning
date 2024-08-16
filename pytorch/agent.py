import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch.model import SimpleCNN
from utils import transform

LR = 0.001


class Agent:
    def __init__(self):

        # Load the datasets with ImageFolder
        self.train_dataset = datasets.ImageFolder(root='../data/train', transform=transform)
        self.val_dataset = datasets.ImageFolder(root='../data/validation', transform=transform)
        self.test_dataset = datasets.ImageFolder(root='../data/test', transform=transform)

        # Define data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

        self.model = SimpleCNN(len(self.train_dataset.classes))

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        # 1. Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.best_val_loss = float('inf')

    def training(self):
        num_epochs = 10  # Define the number of epochs

        for epoch in range(num_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0

            for images, labels in self.train_loader:
                # Move data to the GPU
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()  # Zero the gradients
                outputs = self.model(images)  # Forward pass
                loss = self.criterion(outputs, labels)  # Compute the loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update the weights

                running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(self.train_loader)}')

            self.validation()

    def validation(self):
        self.model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                # Move data to the GPU
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        val_loss /= len(self.val_loader)
        accuracy = 100 * correct / len(self.val_loader.dataset)

        # Check if this is the best model (based on validation loss) and save it
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.model.save()

        print(f'Validation Loss: {val_loss}, Accuracy: {accuracy}%')

    def testing(self):

        self.model.eval()  # Set the model to evaluation mode
        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                # Move data to the GPU
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        test_loss /= len(self.test_loader)
        accuracy = 100 * correct / len(self.test_loader.dataset)

        print(f'Test Loss: {test_loss}, Accuracy: {accuracy}%')


if __name__ == '__main__':
    agent = Agent()
    agent.model.load(agent.model)
    agent.training()
    agent.testing()
