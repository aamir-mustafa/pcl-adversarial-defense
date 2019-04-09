"""
author: aamir-mustafa
This is RobustML interface implementation for the paper:
    Adversarial Defense by Restricting the Hidden Space of Deep Neural Networks
"""

import robustml
import torch
import torch.nn as nn
import numpy as np
from resnet_model import *  # Imports the ResNet Model

num_classes=10
model = resnet(num_classes=num_classes,depth=110)
model = nn.DataParallel(model).cuda()
filename= 'robust_model.pth.tar'  
checkpoint = torch.load(filename)
model.load_state_dict(checkpoint['state_dict'])

#Normalize the data as per CIFAR-10 mean and std
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]
    return t


class Model(robustml.model.Model):
  def __init__(self):

    self._dataset = robustml.dataset.CIFAR10()
    self._threat_model = robustml.threat_model.Linf(epsilon=8/255)

  @property
  def dataset(self):
      return self._dataset

  @property
  def threat_model(self):
      return self._threat_model

  def classify(self, x):
      X = torch.Tensor([x]).cuda()
      model.eval()
      out=model(normalize(X.clone().detach()))[-1].argmax(dim=-1) # Our Model outputs intermediate feats as well
      return out

if __name__ == '__main__':
    robust_model = Model()
    #Design a Random Input
    x = np.zeros((3, 32, 32), dtype=np.float32)
    x[1:2 ,5:-5, 12:-12] = 1

    print('Predicted Class', robust_model.classify(x))
    
    
    
    
    
    