# Import necessary libraries 
import torch 
from PIL import Image 
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define a transform to convert PIL  
# image to a Torch tensor 
transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))
])

birds = []
planes = [] 
for i in range(5):
    # Read a PIL image 
    bird_image = Image.open(f'Images/birds/bird{i}.png')
    plane_image = Image.open(f'Images/planes/plane{i}.png')  

    # Convert the PIL image to Torch tensor 
    birds.append((transform(bird_image), 1))
    planes.append((transform(plane_image), 0)) 

cut = 3  
dataset = birds[:cut] + planes[:cut]
dataset_eval = birds[cut:] + planes[cut:]
# print the converted Torch tensor 
#print(birds[0])

label_map = {0: 0, 1: 1}
class_names = ['plane', 'bird']

model = torch.load("model.h5")

val_loader = torch.utils.data.DataLoader(dataset_eval, batch_size=10, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        outputs = model(imgs.view(imgs.shape[0], -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())

print("Accuracy: %.3f" % (correct / total))