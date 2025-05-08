import torch
import torch.nn as nn
import torch.nn.functional as F

from cifar10_models.vgg import vgg19_bn
from cifar10_dataset.data import CIFAR10Data

if __name__ == "__main__":
    model = vgg19_bn(device = torch.device(0), pretrained=True)
    model.to(torch.device(0))
    data = CIFAR10Data(batch_size=128, num_workers=4)
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()

    # Set model to evaluation mode
    model.eval()

    correct = 0
    total = 0

    # Make sure no gradients are computed (faster and uses less memory)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
