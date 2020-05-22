import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from collections import OrderedDict
import json
from torch.autograd import Variable
import argparse
import os
import sys


#Guide from
#https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
#https://knowledge.udacity.com/questions/23257

def arg_parser():
    
    parser= argparse.ArgumentParser()
    parser.add_argument('--data_dir' , type = str , default = 'flowers' , help = 'dataset directory')
    parser.add_argument('--gpu' , type = bool , default= 'True' , help = 'True: gpu, False:cpu')
    
    parser.add_argument('--lr', type = float , default = 0.001, help = 'learn rate') 
    
    parser.add_argument('--epochs', type = int , default = 1, help = 'Number of Epochs of training')
    parser.add_argument('--arch', type = str , default = 'vgg16', help = 'model' ) 
    parser.add_argument('--hidden_units' , type =  int , default = 600, help = 'hidden units')
    parser.add_argument('--save_dir' , type = str , default = 'checkpoint.pth' , help = 'save directory of the model')
    
    args = parser.parse_args() 
    
    return args

def process_data(train_dir , test_dir, valid_dir ):
    
    #Define Transformations 
    
    
   train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
                                          
   
   test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    
    #Create data sets
   test_dataset= datasets.ImageFolder(test_dir, transform= test_transforms)
   valid_dataset= datasets.ImageFolder(valid_dir, transform = test_transforms)
   train_dataset= datasets.ImageFolder(train_dir, transform =train_transforms)
    
    #Create Data loaders
   trainloader = torch.utils.data.DataLoader(train_dataset , batch_size = 64 , shuffle = True)
   testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = True)
   validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 64 , shuffle = True)
    
   return trainloader, testloader , validloader, train_dataset , test_dataset , valid_dataset


def create_model(arch):
    #I'm only going to test for two models
    #Ask Mentor If I have to test for more
    if arch == 'vg16':
        model = models.vgg16(pretrained=True)
        
        print('Using Model VGG16')
    elif arch == 'denseet121':
        model = models.densenet121(pretrained = True)
        print('Using Model densenet121')
    else:
        print( ' This Program only works with vgg16 or densenet121, defauting to vgg16')
        model = models.vgg16(pretrained= True)
        
        return model

def set_classifier(model , hidden_units):
    print("Creating Classifier: \n")
    input_layers = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_layers, hidden_units, bias=True)),
                                            ('relu1', nn.ReLU()),
                                            ('dropout', nn.Dropout(p=0.5)),
                                            ('fc2', nn.Linear(hidden_units, 128, bias=True)),
                                            ('relu2', nn.ReLU()),
                                            ('dropout', nn.Dropout(p=0.5)),
                                            ('fc3', nn.Linear(128, 102, bias=True)),
                                            ('output', nn.LogSoftmax(dim=1))
                                            ]))
    return classifier

def train_model(epochs, trainloader, validloader, device, model, optimizer, criterion):
    print("Training Model : \n")
    steps=0
    model.to(device)
    running_loss=0
    print_every= 5
    
    for epoch in range(epochs):
        for images , labels in trainloader:
            steps +=1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps% print_every == 0:
                #validates the trained model after every 5 epochs
                #https://stats.stackexchange.com/questions/19048/what-is-the-difference-between-test-set-and-validation-set
                #First runs validation set on this loop 
                #I decided to run the test in a different function so we can test it without training
                
                valid_loss = 0
                accuracy = 0 
                model.eval() #changes to eval mode
                
                
                with torch.no_grad():
                    for images , labels in validloader:
                        images,labels = images.to(device) , labels.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                        
                        #Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equal = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equal.type(torch.FloatTensor)).item()
                        print(f"Epoch {epoch+1}/{epochs}.. "
                              f"Train loss: {running_loss/print_every:.3f}.. "
                              f"Valid loss: {valid_loss/len(validloader):.3f}.."
                              f"Valid accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0
                    #Put the model back in to training mode
                model.train()
    return model

def test_model(model, testloader , device , criterion) :
    test_loss =  0
    accuracy = 0
    model.eval()
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device) , labels.to(device)
            logps = model.forward(images)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            
            #accuracy
            ps = torch.exp(logps)
            top_p , top_class = ps.topk(1,dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            print(f"Test accuracy: {accuracy/len(testloader):.3f}")

def save_checkpoint(model, train_dataset , save_dir , arch):
    model
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {'structure': arch,
                  'classifier': model.classifier,
                  'state_dic': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
    return torch.save(checkpoint, save_dir)


def main():
    in_arg = arg_parser()
    
    
    #check this fucntion again
    if (in_arg.gpu == True):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    
    data_dir = in_arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    trainloader, testloader, validloader, train_dataset, test_dataset, valid_dataset =                        process_data(train_dir, test_dir,valid_dir)
    
    #create model
    
    model = create_model(in_arg.arch)
    
    for param in model.parameters():
        param.requires_gard = False
        
    #load classifier
    model.classifier = set_classifier(model, in_arg.hidden_units)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters() , lr = in_arg.lr)
    
    #Train the model and return it
    model = train_model(in_arg.epochs , trainloader , validloader, device , model,  optimizer, criterion)
    
    print('Model sucessfuly Trained \n \n')
    #run the tests with the test function
    test_model(model , testloader , device , criterion)
    save_checkpoint(model, train_dataset, in_arg.save_dir , in_arg.arch)
    
    print(' Model Sucessfully Saved \n \n')
    
#I've seen someone do this on github ( reference at the top) to test the function so I opted to try the same aproach 
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    

    
    
    
    
    
        
    
            
    

                        

                

    
    

    



    
    
    
    
    
    
    
    

