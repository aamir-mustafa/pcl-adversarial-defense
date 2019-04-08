"""
Created on Sun Mar 24 17:51:08 2019

@author: aamir-mustafa
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from resnet_model import *  # Imports the ResNet Model
"""
Adversarial Attack Options: fgsm, bim, mim, pgd
"""
num_classes=10

model = resnet(num_classes=num_classes,depth=110)
if True:
    model = nn.DataParallel(model).cuda()
    
#Loading Trained Model
softmax_filename= 'Models_Softmax/CIFAR10_Softmax.pth.tar'    
filename= 'Models_PCL/CIFAR10_PCL.pth.tar' 
robust_model= 'robust_model.pth.tar'
checkpoint = torch.load(robust_model)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Loading Test Data (Un-normalized)
transform_test = transforms.Compose([transforms.ToTensor(),])
    
testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                         download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=256, pin_memory=True,
                                          shuffle=False, num_workers=4)

# Mean and Standard Deiation of the Dataset
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

    return t
def un_normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] * std[0]) + mean[0]
    t[:, 1, :, :] = (t[:, 1, :, :] * std[1]) + mean[1]
    t[:, 2, :, :] = (t[:, 2, :, :] * std[2]) + mean[2]

    return t

# Attacking Images batch-wise
def attack(model, criterion, img, label, eps, attack_type, iters):
    adv = img.detach()
    adv.requires_grad = True

    if attack_type == 'fgsm':
        iterations = 1
    else:
        iterations = iters

    if attack_type == 'pgd':
        step = 2 / 255
    else:
        step = eps / iterations
        
        noise = 0
        
    for j in range(iterations):
        _,_,_,out_adv = model(normalize(adv.clone()))
        loss = criterion(out_adv, label)
        loss.backward()

        if attack_type == 'mim':
            adv_mean= torch.mean(torch.abs(adv.grad), dim=1,  keepdim=True)
            adv_mean= torch.mean(torch.abs(adv_mean), dim=2,  keepdim=True)
            adv_mean= torch.mean(torch.abs(adv_mean), dim=3,  keepdim=True)
            adv.grad = adv.grad / adv_mean
            noise = noise + adv.grad
        else:
            noise = adv.grad

        # Optimization step
        adv.data = adv.data + step * noise.sign()
#        adv.data = adv.data + step * adv.grad.sign()

        if attack_type == 'pgd':
            adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
            adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)

        adv.grad.data.zero_()

    return adv.detach()

# Loss Criteria
criterion = nn.CrossEntropyLoss()
adv_acc = 0
clean_acc = 0
eps =8/255 # Epsilon for Adversarial Attack

for i, (img, label) in enumerate(test_loader):
    img, label = img.to(device), label.to(device)
    
    clean_acc += torch.sum(model(normalize(img.clone().detach()))[3].argmax(dim=-1) == label).item()
    adv= attack(model, criterion, img, label, eps=eps, attack_type= 'bim', iters= 10 )
    adv_acc += torch.sum(model(normalize(adv.clone().detach()))[3].argmax(dim=-1) == label).item()
    print('Batch: {0}'.format(i))
print('Clean accuracy:{0:.3%}\t Adversarial accuracy:{1:.3%}'.format(clean_acc / len(testset), adv_acc / len(testset)))










