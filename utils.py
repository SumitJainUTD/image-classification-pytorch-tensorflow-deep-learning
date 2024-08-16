from torchvision import transforms


class ConvertToRGBA(object):
    """Convert image to RGBA if it's in palette mode."""

    def __call__(self, img):
        if img.mode == 'P':
            img = img.convert('RGBA')
        return img


transform = transforms.Compose([
            ConvertToRGBA(),  # Ensure all images are in RGBA format
            transforms.Resize((128, 128)),  # Resize images to 128x128
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
        ])