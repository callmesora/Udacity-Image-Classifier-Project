import json
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image


#New parser for predict 
def args_parser():
    pa = argparse.ArgumentParser(description='predictor')
    pa.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='checkpoint to load from ')
    pa.add_argument('--gpu', type=bool, default='True', help='True: gpu, False: cpu')
    pa.add_argument('--top_k', type=int, default=5, help='top classes')
    pa.add_argument('--img', type=str, required='True',help='Path of image')

    args = pa.parse_args()
    return args

def load_checkpoint(check_path):
    checkpoint = torch.load(check_path)
    #https://stackoverflow.com/questions/4075190/what-is-getattr-exactly-and-how-do-i-use-it
    #getattr allows for meta programming so we can use the models method
    
    #Found this solution thanks to syuukaxiaoy 
    model = getattr(models, checkpoint['structure'])(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False


    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dic'])

    return model

def process_image(image):

     
    #Here I opted for a different version from the one  I did in the notebook, in the notebook I opted to not       use the PIL methods
    #This helped me understand https://knowledge.udacity.com/questions/59039
    
    im = Image.open(image)
    width, height = im.size
    if width > height:
        ratio=width/height
        im.thumbnail((ratio*256,256))
    elif height > width:
        im.thumbnail((256,height/width*256))
    new_width, new_height = im.size # take the size of resized image
    left = (new_width - 224)/2
    top = (new_height - 224)/2
    right = (new_width + 224)/2
    bottom = (new_height+ 224)/2
    im=im.crop((left, top, right, bottom))
    img_array = np.array(im)
    np_img = img_array / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    npimage = (np_img - mean) / std
    #Fix the index
    npimage = npimage.transpose(2, 0, 1)

    return npimage

def predict(image_path, model, device, cat_to_name, topk):
    
   
#Set to cuda if possible
    
    model.to(device)
    model.eval()

    torch_img = torch.from_numpy(np.expand_dims(process_image(image_path), axis=0)).type(torch.FloatTensor).to(device)
    output = torch.exp(model.forward(torch_img))
    probs, classes = output.topk(topk)

    probs = Variable(probs).cpu().numpy()[0]
    probs = [x for x in probs]

    classes = Variable(classes).cpu().numpy()[0]
    classes = [c for c in classes]
    
    #invert dic keys to values , values to keys
    idx_to_classes = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_classes[i] for i in classes]
    #Get the labels from json
    labels = [cat_to_name[l] for l in top_classes]

    return probs, top_classes, labels

def main():
    in_arg = args_parser()
    
    #https://www.programiz.com/python-programming/json
    
    with open('cat_to_name.json','r') as f:
        cat_to_name = json.load(f)
    
    if (in_arg.gpu == True):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    #load model
    model = load_checkpoint(in_arg.checkpoint)
    np_image= process_image(in_arg.img)
    
    top_prob, top_class , top_labels = predict(in_arg.img, model, device , cat_to_name , in_arg.top_k)
    
    print("Predictions")
    #Rubric asks for top Classes , I don't know why 
    print("top classes : " , top_class)
    print("Flowers: " , top_labels)
    print("Probability: ", top_prob)
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)    
