import os

import torch
from PIL import Image
from torchvision import transforms

from pytorch.model import SimpleCNN
from utils import transform
import torch.nn.functional as F


class App:
    def __init__(self):
        self.total_categories = 36
        self.classes = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']
        self.model = SimpleCNN(self.total_categories)
        self.model.load(self.model)

        # 1. Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def load(self, file_name='model.pth'):

        file_path = os.path.join(self.model_folder_path, file_name)
        if os.path.exists(file_path):
            self.model.load_state_dict(torch.load(file_path))
            print("Model loaded.")

    def get_category(self):
        # Load and preprocess the image
        image_path = 'app_img/test.jpg'
        image = Image.open(image_path)
        image = transform(image)
        image = image.unsqueeze(0)  # Add a batch dimension



        self.model.eval()  # Set the model to evaluation mode
        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            # Move data to the GPU
            images = image.to(self.device)
            output = self.model(images)
            # Apply softmax to get the probabilities
            probabilities = F.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)

            index = predicted.item()
            predicted_label = self.classes[index]
            predicted_probability = probabilities[0][index].item()

            print(f'Predicted Label: {predicted_label}')
            print(f'Probability of Correct Prediction: {predicted_probability * 100:.2f}%')
            # correct += (predicted == labels).sum().item()
        #
        # test_loss /= len(self.test_loader)
        # accuracy = 100 * correct / len(self.test_loader.dataset)
        #
        # print(f'Test Loss: {test_loss}, Accuracy: {accuracy}%')

if __name__ == '__main__':
    app = App()
    app.get_category()