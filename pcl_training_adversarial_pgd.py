
"""
Created on Wed Jan 23 10:15:27 2019

@author: aamir-mustafa
Implementation Part 2 of Paper: 
    "Adversarial Defense by Restricting the Hidden Space of Deep Neural Networks"  

Here it is not necessary to save the best performing model (in terms of accuracy). The model with high robustness 
against adversarial attacks is chosen.
This coe implements Adversarial Training using PGD Attack.   
"""

#Essential Imports
import os
import sys
import argparse
import datetime
import time
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from utils import AverageMeter, Logger
from proximity import Proximity
from contrastive_proximity import Con_Proximity
from resnet_model import *  # Imports the ResNet Model


parser = argparse.ArgumentParser("Prototype Conformity Loss Implementation")
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--schedule', type=int, nargs='+', default=[142, 230, 360],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--lr_model', type=float, default=0.01, help="learning rate for model")
parser.add_argument('--lr_prox', type=float, default=0.5, help="learning rate for Proximity Loss") # as per paper
parser.add_argument('--weight-prox', type=float, default=1, help="weight for Proximity Loss") # as per paper
parser.add_argument('--lr_conprox', type=float, default=0.00001, help="learning rate for Con-Proximity Loss") # as per paper
parser.add_argument('--weight-conprox', type=float, default=0.00001, help="weight for Con-Proximity Loss") # as per paper
parser.add_argument('--max-epoch', type=int, default=500)
parser.add_argument('--gamma', type=float, default=0.1, help="learning rate decay")
parser.add_argument('--eval-freq', type=int, default=10)
parser.add_argument('--print-freq', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

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
        _,_,_,out_adv = model(adv.clone())
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
        adv.data = un_normalize(adv.data) + step * noise.sign()
#        adv.data = adv.data + step * adv.grad.sign()

        if attack_type == 'pgd':
            adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
            adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)

        adv.grad.data.zero_()

    return adv.detach()

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    sys.stdout = Logger(osp.join(args.save_dir, 'log_' + 'CIFAR-10_PC_Loss_PGD_AdvTrain' + '.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    # Data Load
    num_classes=10
    print('==> Preparing dataset')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    
    trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                             download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, pin_memory=True,
                                              shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, pin_memory=True,
                                             shuffle=False, num_workers=args.workers)
    
# Loading the Model    
    model = resnet(num_classes=num_classes,depth=110)

    if True:
        model = nn.DataParallel(model).cuda()

    criterion_xent = nn.CrossEntropyLoss()
    criterion_prox_1024 = Proximity(num_classes=num_classes, feat_dim=1024, use_gpu=use_gpu)
    criterion_prox_256 = Proximity(num_classes=num_classes, feat_dim=256, use_gpu=use_gpu)
    
    criterion_conprox_1024 = Con_Proximity(num_classes=num_classes, feat_dim=1024, use_gpu=use_gpu)
    criterion_conprox_256 = Con_Proximity(num_classes=num_classes, feat_dim=256, use_gpu=use_gpu)
    
    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=1e-04, momentum=0.9)
    
    optimizer_prox_1024 = torch.optim.SGD(criterion_prox_1024.parameters(), lr=args.lr_prox)
    optimizer_prox_256 = torch.optim.SGD(criterion_prox_256.parameters(), lr=args.lr_prox)

    optimizer_conprox_1024 = torch.optim.SGD(criterion_conprox_1024.parameters(), lr=args.lr_conprox)
    optimizer_conprox_256 = torch.optim.SGD(criterion_conprox_256.parameters(), lr=args.lr_conprox)
    

    filename= 'Models_Softmax/CIFAR10_Softmax.pth.tar'
    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint['state_dict']) 
    optimizer_model.load_state_dict= checkpoint['optimizer_model']

    start_time = time.time()

    for epoch in range(args.max_epoch):
        
        adjust_learning_rate(optimizer_model, epoch)
        adjust_learning_rate_prox(optimizer_prox_1024, epoch)
        adjust_learning_rate_prox(optimizer_prox_256, epoch)
        
        adjust_learning_rate_conprox(optimizer_conprox_1024, epoch)
        adjust_learning_rate_conprox(optimizer_conprox_256, epoch)
        
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        train(model, criterion_xent, criterion_prox_1024, criterion_prox_256, 
              criterion_conprox_1024, criterion_conprox_256, 
              optimizer_model, optimizer_prox_1024, optimizer_prox_256,
              optimizer_conprox_1024, optimizer_conprox_256,
              trainloader, use_gpu, num_classes, epoch)

        if args.eval_freq > 0 and (epoch+1) % args.eval_freq == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")   #Tests after every 10 epochs
            acc, err = test(model, testloader, use_gpu, num_classes, epoch)
            print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))

            state_ = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                     'optimizer_model': optimizer_model.state_dict(), 'optimizer_prox_1024': optimizer_prox_1024.state_dict(),
                     'optimizer_prox_256': optimizer_prox_256.state_dict(), 'optimizer_conprox_1024': optimizer_conprox_1024.state_dict(),
                     'optimizer_conprox_256': optimizer_conprox_256.state_dict(),}
                     
            torch.save(state_, 'Models_PCL_AdvTrain_PGD/CIFAR10_PCL_AdvTrain_PGD.pth.tar')
            
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

def train(model, criterion_xent, criterion_prox_1024, criterion_prox_256, 
              criterion_conprox_1024, criterion_conprox_256, 
              optimizer_model, optimizer_prox_1024, optimizer_prox_256,
              optimizer_conprox_1024, optimizer_conprox_256,
              trainloader, use_gpu, num_classes, epoch):
    
#    model.train()
    xent_losses = AverageMeter() #Computes and stores the average and current value
    prox_losses_1024 = AverageMeter()
    prox_losses_256= AverageMeter()
    
    conprox_losses_1024 = AverageMeter()
    conprox_losses_256= AverageMeter()
    losses = AverageMeter()
    
    #Batchwise training
    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        model.eval()
        eps= np.random.uniform(0.02,0.05)
        adv = attack(model, criterion_xent, data, labels, eps=eps, attack_type='pgd', iters= 10) # Generates Batch-wise Adv Images
        adv.requires_grad= False
        
        adv= normalize(adv)
        adv= adv.cuda()
        true_labels_adv= labels
        data= torch.cat((data, adv),0)
        labels= torch.cat((labels, true_labels_adv))
        model.train()
        
        feats128, feats256, feats1024, outputs = model(data) 
        loss_xent = criterion_xent(outputs, labels)  
        
        loss_prox_1024 = criterion_prox_1024(feats1024, labels) 
        loss_prox_256= criterion_prox_256(feats256, labels)
        
        loss_conprox_1024 = criterion_conprox_1024(feats1024, labels) 
        loss_conprox_256= criterion_conprox_256(feats256, labels)
        
        loss_prox_1024 *= args.weight_prox 
        loss_prox_256 *= args.weight_prox
        
        loss_conprox_1024 *= args.weight_conprox 
        loss_conprox_256 *= args.weight_conprox
        
        loss = loss_xent + loss_prox_1024 + loss_prox_256  - loss_conprox_1024 - loss_conprox_256 # total loss
        optimizer_model.zero_grad()
        
        optimizer_prox_1024.zero_grad()
        optimizer_prox_256.zero_grad()
        
        optimizer_conprox_1024.zero_grad()
        optimizer_conprox_256.zero_grad()

        loss.backward()
        optimizer_model.step() 
        
        for param in criterion_prox_1024.parameters():
            param.grad.data *= (1. / args.weight_prox)
        optimizer_prox_1024.step() 
        
        for param in criterion_prox_256.parameters():
            param.grad.data *= (1. / args.weight_prox)
        optimizer_prox_256.step()
        

        for param in criterion_conprox_1024.parameters():
            param.grad.data *= (1. / args.weight_conprox)
        optimizer_conprox_1024.step() 
        
        for param in criterion_conprox_256.parameters():
            param.grad.data *= (1. / args.weight_conprox)
        optimizer_conprox_256.step()
        
        losses.update(loss.item(), labels.size(0)) 
        xent_losses.update(loss_xent.item(), labels.size(0))
        prox_losses_1024.update(loss_prox_1024.item(), labels.size(0))
        prox_losses_256.update(loss_prox_256.item(), labels.size(0))

        conprox_losses_1024.update(loss_conprox_1024.item(), labels.size(0))
        conprox_losses_256.update(loss_conprox_256.item(), labels.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})  XentLoss {:.6f} ({:.6f})  ProxLoss_1024 {:.6f} ({:.6f}) ProxLoss_256 {:.6f} ({:.6f}) \n ConProxLoss_1024 {:.6f} ({:.6f}) ConProxLoss_256 {:.6f} ({:.6f}) " \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg, xent_losses.val, xent_losses.avg, 
                          prox_losses_1024.val, prox_losses_1024.avg, prox_losses_256.val, prox_losses_256.avg , 
                          conprox_losses_1024.val, conprox_losses_1024.avg, conprox_losses_256.val,
                          conprox_losses_256.avg  ))


def test(model, testloader, use_gpu, num_classes, epoch):
    model.eval()  
    correct, total = 0, 0

    with torch.no_grad():
        for data, labels in testloader:
            if True:
                data, labels = data.cuda(), labels.cuda()
            feats128, feats256, feats1024, outputs = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()
            

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr_model'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr_model'] = state['lr_model']
            
def adjust_learning_rate_prox(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr_prox'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr_prox'] = state['lr_prox'] 

def adjust_learning_rate_conprox(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr_conprox'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr_conprox'] = state['lr_conprox']             
if __name__ == '__main__':
    main()





