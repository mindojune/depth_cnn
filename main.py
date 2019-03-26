import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image #as img
import os
import os.path

import argparse
import sys
import time
import re

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
from PIL import Image
from torchvision import models
from collections import namedtuple

import utils
from transformer_net import TransformerNet
from vgg import Vgg16
from matplotlib import cm 

from torch.utils.data.dataset import Dataset
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from cnn_finetune import make_model
import loaddata


device = 'cpu' #torch.device('cuda' if use_cuda else 'cpu')


def visualize_trio(image, depth, output):
	cols = 1
	images = [image, output, depth]
	n_images = len(images)
	titles = ["image", "stylized", "depth"]
	if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
	fig = plt.figure()
	for n, (img, title) in enumerate(zip(images, titles)):
		a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
		#if image.ndim == 2:
		#	plt.gray()
		plt.imshow(img)
		a.set_title(title)
	fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
	plt.show()	

	return

def single_stylize(style_model, image):
	content_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Lambda(lambda x: x.mul(255))
		])
	content_image = content_transform(image)
	content_image = content_image.unsqueeze(0).to(device)
	output = style_model(content_image).cpu()
	output = output.squeeze().permute(1,2,0).int().data.numpy()

	return output

def stylize_NYU(frame):
	size = len(frame)


	models = ["candy" , "mosaic", "rain_princess", "udnie"]
	#for model in models:
	for idx in range(size):
		image_name = frame.iloc[idx, 0]
		depth_name = frame.iloc[idx, 1]

		#image = Image.open(image_name)
		#depth = Image.open(depth_name)
		image = matplotlib.image.imread(image_name)
		depth = matplotlib.image.imread(depth_name)

		with torch.no_grad():
			style_model = TransformerNet()
			modelpath = "saved_models/"+models[0]+".pth"
			state_dict = torch.load(modelpath)
			# remove saved deprecated running_* keys in InstanceNorm from the checkpoint
			for k in list(state_dict.keys()):
				if re.search(r'in\d+\.running_(mean|var)$', k):
					del state_dict[k]
			style_model.load_state_dict(state_dict)
			style_model.to(device)

			output = single_stylize(style_model, image)
			# content_image = content_transform(image)
			# content_image = content_image.unsqueeze(0).to(device)
			# output = style_model(content_image).cpu()
			# output = output.squeeze().permute(1,2,0).int().data.numpy()

		visualize_trio(image, depth, output)
	return

def main():
	parser = argparse.ArgumentParser(description='stylized augmentation')
	parser.add_argument('--batch-size', type=int, default=32, metavar='N',
						help='input batch size for training (default: 32)')
	parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
						help='input batch size for testing (default: 64)')
	parser.add_argument('--epochs', type=int, default=20, metavar='N',
						#help='number of epochs to train (default: 100)')
						help='number of epochs to train (default: 20)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
						help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
						help='SGD momentum (default: 0.9)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=100, metavar='N',
						help='how many batches to wait before logging training status')
	parser.add_argument('--model-name', type=str, default='resnet50', metavar='M',
						help='model name (default: resnet50)')
	parser.add_argument('--dropout-p', type=float, default=0.2, metavar='D',
						help='Dropout probability (default: 0.2)')
	parser.add_argument('--ratio', type=int, default=1, help='Orig: Stylized Ratio (default: 1)')
	args = parser.parse_args()
	ratio = args.ratio
	
	print("Using ratio:", ratio)
	print("Using epoch #:", args.epochs)

	# Read NYU Depth Dataset
	train_frame = loaddata.getTrainingDataFrame()
	#train_loader = loaddata.getTrainingData(args.batch_size)
	#test_loader = loaddata.getTestingData(batch_size = 1)
	#print(train_frame)


	stylize_NYU(train_frame)
	exit()
	#################################
	#################################
	#################################
	#################################
	#################################

	return


if __name__ == "__main__":
	main()
