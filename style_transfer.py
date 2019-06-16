"""
This is a basic experiment trying to replicate the results in https://arxiv.org/abs/1508.06576
and using it to minimise the styles of multiple images at the same time.
The idea is that I'll copy the code required to load the VGG model but try to
reproduce the rest by myself.
"""

import time
import os
import argparse

import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms.functional as tvf

def prepare_image(img,img_size):
    img = tvf.resize(img,(img_size,img_size))
    tensor = tvf.to_tensor(img)
    tensor = tvf.normalize(tensor, mean=[0.40760392, 0.45795686, 0.48501961], std=[1,1,1])
    tensor = tensor * 255
    return tensor
#end prepare_image

def prepare_output(tensor):
    tensor = tensor / 255
    tensor = tvf.normalize(tensor, mean=[-0.40760392, -0.45795686, -0.48501961], std=[1,1,1])
    tensor[tensor>1] = 1
    tensor[tensor<0] = 0
    img = tvf.to_pil_image(tensor)
    return img
#end prepare_output

def valid_format(fname):
    return any( fname.endswith("jpeg"), fname.endswith("jpg"), fname.endswith("png") )
#end

class VGG_ST(nn.Module):
    """
    Class that holds the VGG module
    """
    def __init__(self, pool="max", load_dir=None):
        super( VGG_ST, self ).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512 , kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        if pool=="max":
            self.pool1 = nn.MaxPool2d( kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d( kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d( kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d( kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d( kernel_size=2, stride=2)
        elif pool=="avg":
            self.pool1 = nn.AvgPool2d( kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d( kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d( kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d( kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d( kernel_size=2, stride=2)
        else:
            raise ValueError("Invalid pooling type")
        #end if-else
        
        if load_dir is not None:
            self.load_state_dict( torch.load( load_dir ) )
            for param in self.parameters():
                param.requires_grad = False
            #end for
        #end if
    #end __init__
    
    def forward(self, x, out_keys = ["p5"]):
        out = {}
        x = F.relu(self.conv1_1(x))
        out["r11"] = x
        x = F.relu(self.conv1_2(x))
        out["r12"] = x
        x = self.pool1( x )
        out["p1"] = x
        
        x = F.relu(self.conv2_1(x))
        out["r21"] = x
        x = F.relu(self.conv2_2(x))
        out["r22"] = x
        x = self.pool2( x )
        out["p1"] = x
        
        x = F.relu(self.conv3_1(x))
        out["r31"] = x
        x = F.relu(self.conv3_2(x))
        out["r32"] = x
        x = F.relu(self.conv3_3(x))
        out["r33"] = x
        x = F.relu(self.conv3_4(x))
        out["r34"] = x
        x = self.pool3( x )
        out["p1"] = x
        
        x = F.relu(self.conv4_1(x))
        out["r41"] = x
        x = F.relu(self.conv4_2(x))
        out["r42"] = x
        x = F.relu(self.conv4_3(x))
        out["r43"] = x
        x = F.relu(self.conv4_4(x))
        out["r44"] = x
        x = self.pool4( x )
        out["p1"] = x
        
        x = F.relu(self.conv5_1(x))
        out["r51"] = x
        x = F.relu(self.conv5_2(x))
        out["r52"] = x
        x = F.relu(self.conv5_3(x))
        out["r53"] = x
        x = F.relu(self.conv5_4(x))
        out["r54"] = x
        x = self.pool5( x )
        out["p1"] = x
        
        return [out[k] for k in out_keys]
    #end forward
#end VGG_ST

def gram_matrix(x):
    b,c,h,w = x.size()
    F = x.view(b, c, h*w)
    G = torch.bmm(F, F.transpose(1,2))
    G.div_(h*w)
    return G
#end GramMatrix

def gram_mse_loss(x,y):
    return sum( F.mse_loss( gram_matrix(x[i:i+1]), y ) for i in range(x.size(0)) )
#end gram_mse_loss

class StyleTransfer(nn.Module):
    def __init__(self, vgg_path=None):
        super( StyleTransfer, self ).__init__()
        self.vgg = VGG_ST(load_dir=vgg_path)
        self.style_image_layers = ["r11", "r21", "r31", "r41", "r51"]
        self.style_image_layer_weights = [1e3/n**2 for n in [64,128,256,512,512]]
        self.style_image_outputs = []
        self.style_image_weights = []
        
        self.content_image_layers = ["r42"]
        self.content_weights_layers = [1]
        
    #end __init__
    
    def set_style_images(self, imgs, weights=None):
        self.style_image_outputs = self.vgg(imgs, self.style_image_layers)
        self.style_image_weights = weights if weights is not None else [1/imgs.size(0)]*imgs.size(0)
        self.S_tgt = [gram_matrix(out) for out in self.style_image_outputs]
    #end set_style_images
    
    def generate(self, imgs, num_epochs=200, checkpoints=[], savedir="./", fnames=[] ):
        b,c,h,w = imgs.size()
        
        gen_imgs = torch.tensor(imgs, requires_grad = True)
        optimiser = torch.optim.LBFGS([gen_imgs])
        
        C_tgt = self.vgg( imgs, self.content_image_layers )
        
        for epoch in range(num_epochs):
            report_loss = [0]
            def closure():
                optimiser.zero_grad()
                
                out = self.vgg( gen_imgs, self.content_image_layers + self.style_image_layers )
                C_out = out[:len(self.content_image_layers)]
                S_out = out[len(self.content_image_layers):]
                
                C_loss = [ F.mse_loss( out, tgt ) for out, tgt in zip( C_out, C_tgt) ]
                S_loss_ = []
                for s, w in zip( self.S_tgt, self.style_image_weights ):
                    S_loss_.append( [ w * gram_mse_loss( out, tgt ) for out, tgt in zip( S_out, self.S_tgt) ] )
                #end for
                
                num_s_loss = len(self.S_tgt)
                S_loss = [ 0 for _ in range(num_s_loss) ]
                for s_loss in S_loss_:
                    for i in range(num_s_loss):
                        S_loss[i] += s_loss[i]
                    #end for
                #end for
                
                loss = sum(C_loss) + sum(S_loss)
                report_loss[0] = loss
                loss.backward()
                return loss
            #end closure

            optimiser.step(closure)
            
            print("Epoch {epoch}, loss {loss}".format(
                    epoch = epoch,
                    loss = report_loss[0].item(),
            ))
            if epoch in checkpoints:
                print("Saving images")
                self.save_images(
                        gen_imgs.detach().cpu(),
                        list(map(lambda x: savedir+"ckp/"+str(epoch)+"/"+x, fnames))
                )
            #end if
        #end for
        
        return gen_imgs
    #end generate
    
    def save_images(self, gen, paths):
        for i in range(gen.size(0)):
            img = prepare_output(gen[i])
            plt.imsave(paths[i],np.array(img))
            plt.close()
        #end for
    #end while
    #end save_images
#end StyleTransfer
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performs neural style transfer")
    parser.add_argument("--vgg_path", type=str, help="Path to saved VGG model", default="./models/vgg_conv.pth")
    parser.add_argument("--style_path", type=str, help="Path to style folder", default="./styles/van-gogh/")
    parser.add_argument("--target_path", type=str, help="Path to folder containing the images to perform transfer on", default="./targets/")
    parser.add_argument("--gen_path", type=str, help="Path to save the transferred images on", default="./generated/")
    parser.add_argument("--checkpoints", nargs="*", type=int, help="Print the loss and save the intermediary image in these epochs", default=[])
    parser.add_argument("--no_cuda", action="store_const", const=True, default=False, help="Disables cuda")
    parser.add_argument("--num_epochs", type=int, help="Maximum number of epochs to perform gradient descent on the image", default=200)
    parser.add_argument("--img_size", type=int, help="Size to resize input images and output generated images", default=512)
    parser.add_argument("--batch_size", type=int, help="Number of images to generate concurrently", default=1)
    
    args = parser.parse_args()
    
    #args.batch_size = 1
    
    args.style_path += "/" if not args.style_path.endswith("/") else ""
    args.gen_path += "/" if not args.style_path.endswith("/") else ""
    args.target_path += "/" if not args.style_path.endswith("/") else ""
    
    os.makedirs(args.gen_path,exist_ok=True)
    if len(args.checkpoints) > 0:
        args.num_epochs = max(args.num_epochs, max(args.checkpoints))
        os.makedirs(args.gen_path+"ckp/",exist_ok=True)
        for ckp in args.checkpoints:
            os.makedirs(args.gen_path+"ckp/"+str(ckp)+"/",exist_ok=True)
        #end for
    #end if
    
    style_imgs = [Image.open(args.style_path + f) for f in os.listdir(args.style_path) if f.endswith(".jpg") or f.endswith(".png")]
    style_tensors = [prepare_image(img,args.img_size) for img in style_imgs]
    style_tensor_stacked = torch.stack(style_tensors)
    
    fnames = [f for f in os.listdir(args.target_path) if f.endswith(".jpg") or f.endswith(".png")]
    num_imgs = len(fnames)
    gen_fnames = [args.gen_path+f for f in fnames]
    target_imgs = [Image.open(args.target_path+f) for f in fnames]
    target_tensors = [prepare_image(img,args.img_size) for img in target_imgs]
    
    st = StyleTransfer(vgg_path=args.vgg_path)
    
    if not args.no_cuda and torch.cuda.is_available():
        st = st.cuda()
        style_tensor_stacked = style_tensor_stacked.cuda()
    #end if
    
    img_counter = 0
    st.set_style_images(style_tensor_stacked)
    while img_counter < num_imgs:
        target_tensor_stacked = torch.stack(target_tensors[img_counter:img_counter+args.batch_size])
        if not args.no_cuda and torch.cuda.is_available():
            target_tensor_stacked = target_tensor_stacked.cuda()
        #end if
        gnames = gen_fnames[img_counter:img_counter+args.batch_size]
        gen = st.generate(
                target_tensor_stacked,
                num_epochs=args.num_epochs,
                checkpoints=args.checkpoints,
                savedir=args.gen_path,
                fnames=fnames[img_counter:img_counter+args.batch_size],
        ).cpu()
        st.save_images(gen, gnames)
        img_counter += args.batch_size
    #end while
    print("Fin")
