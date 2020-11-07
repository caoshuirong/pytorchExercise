import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.utils.data as Data
tran_set = torchvision.datasets.FashionMNIST(root = "./data",train=True,transform=transforms.Compose([transforms.ToTensor()]),download=True)

tran_loader = Data.DataLoader(tran_set,10)

batch = next(iter(tran_loader))

images,labels = batch

grid = torchvision.utils.make_grid(images,nrow=10,padding=0)

plt.figure(figsize=(15,15))

plt.imshow(np.transpose(grid,(1,2,0)))

plt.waitforbuttonpress()

print("labels:",labels)








