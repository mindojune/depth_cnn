import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
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
#from fast_gradient_sign_untargeted import FastGradientSignUntargeted
#from misc_functions import get_params


classes = (		'plane', 'car', 'bird', 'cat', 'deer',
		'dog', 'frog', 'horse', 'ship', 'truck' )
labels = classes #['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
device = 'cpu' #torch.device('cuda' if use_cuda else 'cpu')

def get_label_string(onehot):
	idx = np.argmax(onehot)

	return labels[idx]

def get_label_idx(onehot):
	idx = np.argmax(onehot)

	return idx	

def load_data(file):
	filepath = os.path.abspath(file)
	dict = unpickle(filepath)
	#for key in dict.keys():
	#   print(key)
	#print("Loading {}".format(dict[b'batch_label']))
	X = np.asarray(dict[b'data'].T).astype("uint8")
	Ypre = np.asarray(dict[b'labels'])
	Y = np.zeros((10,10000))
	for i in range(10000):
		Y[Ypre[i], i] = 1
	names = np.asarray(dict[b'filenames'])
	return X, Y, names

# for CIFAR 10
def unpickle(file):
	# Loaded in this way, each of the batch files contains a dictionary with the following elements:
	# data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. 
	#           The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. 
	#           The image is stored in row-major order, so that the first 32 entries of the array are 
	#           the red channel values of the first row of the image.
	# labels -- a list of 10000 numbers in the range 0-9. 
	#           The number at index i indicates the label of the ith image in the array data.
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def visualize_image(X,Y):
	rgb = X#[:,id]
	img = rgb.reshape(3,32,32).transpose([1, 2, 0])
	plt.imshow(img)
	plt.title(get_label_string(Y))
	plt.show()

def visualize_tensor(X,Y):
	rgb = X#[:,id]
	img = rgb.squeeze().permute(1,2,0).int() #rgb.reshape(3,32,32).transpose([1, 2, 0])
	#print(img)
	plt.imshow(img)
	plt.title(get_label_string(Y))
	plt.show()

def train(epoch, model, train_loader, optimizer, criterion, args):
	total_loss = 0
	total_size = 0
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
		total_loss += loss.item()
		total_size += data.size(0)
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), total_loss / total_size))


def test(model, test_loader, criterion, args):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += criterion(output, target).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

	test_loss /= len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))

def compute_mean_std(dataset):
	"""compute the mean and std of dataset
	Args:
		dataset or test dataset
		witch derived from class torch.utils.data
	
	Returns:
		a tuple contains mean, std value of entire dataset
	"""
	data_r = np.dstack([dataset[i].squeeze()[0, :, :] for i in range(len(dataset))])
	data_g = np.dstack([dataset[i].squeeze()[1, :, :] for i in range(len(dataset))])
	data_b = np.dstack([dataset[i].squeeze()[2, :, :] for i in range(len(dataset))])
	mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
	std = np.std(data_r), np.std(data_g), np.std(data_b)
	return mean, std

# TODOs:
# !!!: Write the signature for the necessary functions first
# 1. Generate Dataset (based on what?) - CIFAR 10 
# 2. Train a CNN on the dataset
# 3. Test the trained network on adversarial examples: https://github.com/MadryLab/cifar10_challenge
# import all depenenciies
# 

class PerturbedCIFAR10(Dataset):
	def __init__(self, X, Y, transform):
		"""
		Args:
			csv_path (string): path to csv file
			img_path (string): path to the folder where images are
			transform: pytorch transforms for transforms and tensor conversion
		"""
		# Transforms
		#self.to_tensor = transforms.ToTensor()
		self.transform = transform

		self.X = X
		self.Y = Y
		assert(len(self.X) == len(self.Y))
		# Calculate len
		self.data_len = len(self.X)

	def __getitem__(self, index):
		if type(self.Y[index]) != int:
			return (self.transform(self.X[index].squeeze()), get_label_idx(self.Y[index]))
		else:
			return (self.transform(self.X[index].squeeze()), self.Y[index])
	def __len__(self):
		return self.data_len


def stylize(model, content_image, output_image = None):
	cuda = False
	device = torch.device("cuda" if cuda else "cpu")

	content_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Lambda(lambda x: x.mul(255))
	])
	content_image = content_transform(content_image)
	content_image = content_image.unsqueeze(0).to(device)

	with torch.no_grad():
		style_model = TransformerNet()
		state_dict = torch.load(model)
		# remove saved deprecated running_* keys in InstanceNorm from the checkpoint
		for k in list(state_dict.keys()):
			if re.search(r'in\d+\.running_(mean|var)$', k):
				del state_dict[k]
		style_model.load_state_dict(state_dict)
		style_model.to(device)
		output = style_model(content_image).cpu()
	#utils.save_image(output_image, output[0])
	return output 

def generate_perturbed_data( X, Y, stylize_option):

	pX, pY = [], []

	models = ["candy" , "mosaic", "rain_princess", "udnie"]

	count = 0
	idx = 0
	total = len(models) * X.shape[0]
	print("total", total)
	for x in X:
		for modelname in models:
			model = "saved_models/"+ modelname+".pth"
			# TODO Make this faster by loading first
			output = stylize(model, x, "ex1_stylized.png")
			#newX0 = torch.tensor(np.transpose(x.reshape(1,target.shape[0], x.shape[1], x.shape[2]), (0,3,1,2)))
			#utils.save_image("ex1_original.png", newX0.squeeze())
			#utils.save_image("ex1_stylized.png", output.squeeze())
			count += 1
			pX.append(output)
			pY.append(Y[idx])
		if count % 10 == 0:
			print("#",count)
			break	
		idx += 1

	return pX, pY

def stylize_batch(model, content_batch, output_image = None):
	cuda = False
	device = torch.device("cuda" if cuda else "cpu")

	content_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Lambda(lambda x: x.mul(255))
	])

	output_batch = []
	idx = 0
	with torch.no_grad():
		style_model = TransformerNet()
		state_dict = torch.load(model)
		# remove saved deprecated running_* keys in InstanceNorm from the checkpoint
		for k in list(state_dict.keys()):
			if re.search(r'in\d+\.running_(mean|var)$', k):
				del state_dict[k]
		style_model.load_state_dict(state_dict)
		style_model.to(device)
		for content_image in content_batch:
			content_image = content_transform(content_image)
			content_image = content_image.unsqueeze(0).to(device)
			output = style_model(content_image).cpu()
			output_batch.append(output)
			if idx % 1000 == 0:
				print("#", idx)
			idx += 1
	return output_batch 

def perturb_data( X, Y, stylize_option):
	n = len(X)

	pX, pY = [  ], [  ]
	models = ["candy" , "mosaic", "rain_princess", "udnie"]
	total = (len(models) + 1) * (len(X))
	print("total", total)
	for modelname in models:
		model = "saved_models/"+ modelname+".pth"
		output_batch = stylize_batch(model, X[0:n])
		pX += output_batch
		pY += Y[0:n].tolist()
		print("One loop done")

	content_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Lambda(lambda x: x.mul(255))
	])
	origX = [ content_transform(x) for x in X[0:n] ]
	pX += origX 
	pY += Y[0:n].tolist()
	return pX, pY


def randomSubset(X, numcopies, ratio):
	if numcopies == 1:
		return X

	assert(len(X) % numcopies == 0)

	newX = []
	
	original = 0.0
	stylized = 0.0
	for i in range(int(len(X)/numcopies)):
		# idx = np.random.randint(numcopies)
		# newX.append(X[idx*50000+i])
		idx = np.random.randint(ratio+1)
		if idx != 0:
			# [1 - ratio ]
			idx = np.random.randint(ratio) + 1 
			newX.append(X[idx*50000+i])
			stylized += 1.0
		else:
			newX.append(X[i])
			original += 1.0
		#ids = {}
		#while len(ids) != ratio - 1:


	print("True ratio:", str(stylized/original))
	return newX


def generate_adversarial(img, label, pretrained_model):
	target_example = 2  # Eel
	(original_image, prep_img) = get_params(img)

	FGS_untargeted = FastGradientSignUntargeted(pretrained_model, 0.01)
	result, flag = FGS_untargeted.generate(original_image, label)

	return result, flag

def main():
	parser = argparse.ArgumentParser(description='stylized defebse')
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

	# filename = "data/perturbed.pkl"
	# if os.path.exists(filename):
	# 	label_meta = unpickle("cifar-10-batches-py/batches.meta")

	# 	X1,Y1,names1 = load_data('cifar-10-batches-py/data_batch_1')
	# 	X2,Y2,names2 = load_data('cifar-10-batches-py/data_batch_2')
	# 	X3,Y3,names3 = load_data('cifar-10-batches-py/data_batch_3')
	# 	X4,Y4,names4 = load_data('cifar-10-batches-py/data_batch_4')
	# 	X5,Y5,names5 = load_data('cifar-10-batches-py/data_batch_5')
		
	# 	Xtest, Ytest, namestest = load_data('cifar-10-batches-py/test_batch')

	# 	X = np.concatenate([X1, X2, X3, X4, X5], axis = 1)
	# 	Y = np.concatenate([Y1, Y2, Y3, Y4, Y5], axis = 1)
	# 	X = np.transpose(X,(1,0))
	# 	Y = np.transpose(Y,(1,0))
	# 	newX = []
	# 	for x in X:
	# 		img = x.reshape(3,32,32).transpose([1, 2, 0])
	# 		newX.append(img)

	# 	print("pkl file found...")
	# 	with open(filename, 'rb') as file:
	# 		data = pickle.load(file)
	# 	print("pkl file loading complete")			
	# 	Xstyle = data[0]
	# 	Ystyle = data[1]

	# 	for i in range(len(X)):
	# 		visualize_image(X[i], Y[i])
	# 		visualize_tensor(Xstyle[i+0*len(X)], Y[i])
	# 		visualize_tensor(Xstyle[i+1*len(X)], Y[i])
	# 		visualize_tensor(Xstyle[i+2*len(X)], Y[i])
	# 		visualize_tensor(Xstyle[i+3*len(X)], Y[i])

	# exit()

	# TODO 1. Generate Dataset (based on what?) - CIFAR 10 
	filename = "data/perturbed.pkl"
	if not os.path.exists(filename):
		print("loading and saving file first...")
		label_meta = unpickle("cifar-10-batches-py/batches.meta")

		X1,Y1,names1 = load_data('cifar-10-batches-py/data_batch_1')
		X2,Y2,names2 = load_data('cifar-10-batches-py/data_batch_2')
		X3,Y3,names3 = load_data('cifar-10-batches-py/data_batch_3')
		X4,Y4,names4 = load_data('cifar-10-batches-py/data_batch_4')
		X5,Y5,names5 = load_data('cifar-10-batches-py/data_batch_5')
		
		Xtest, Ytest, namestest = load_data('cifar-10-batches-py/test_batch')
		
		X = np.concatenate([X1, X2, X3, X4, X5], axis = 1)
		Y = np.concatenate([Y1, Y2, Y3, Y4, Y5], axis = 1)
		X = np.transpose(X,(1,0))
		Y = np.transpose(Y,(1,0))
		newX = []
		for x in X:
			img = x.reshape(3,32,32).transpose([1, 2, 0])
			newX.append(img)

		#Xstyle, Ystyle = generate_perturbed_data( newX, Y, stylize_option="fast")
		Xstyle, Ystyle = perturb_data( newX, Y, stylize_option="fast")		

		with open(filename, 'wb') as file:
			data = [Xstyle, Ystyle]
			pickle.dump(data, file)
	else:
		print("pkl file found...")
		with open(filename, 'rb') as file:
			data = pickle.load(file)
		Xstyle = data[0]
		Ystyle = data[1]

		

	# TODO: don't use the whole set, use
	#       so that each picture is only used once.
	numcopies = 5
	#ratio = 4
	
	X = randomSubset(Xstyle, numcopies, ratio)
	print("Size of Resulting Set:", len(X))
	Y = Ystyle[0:50000]
	assert(len(X) == len(Y))
	mean, std = compute_mean_std(X)
	mean = torch.tensor(mean)
	std = torch.tensor(std)


	# for i in range(10):
	# 	idx = np.random.randint(len(Ystyle))
	# 	visualize_image(Xstyle[idx].numpy().astype(np.uint8),Ystyle[idx])

	# TODO 2. Train a CNN on the dataset
	# model = get_model("model_name")
	# train(model, Xstyle, Ystyle)




	
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	model_name = args.model_name

	if model_name == 'alexnet':
		raise ValueError('The input size of the CIFAR-10 data set (32x32) is too small for AlexNet')

	#model_name = "vgg16"
	model = make_model(
		model_name,
		pretrained=False,#True,
		num_classes=len(classes),
		dropout_p=args.dropout_p,
		input_size=(32, 32) if model_name.startswith(('vgg', 'squeezenet')) else None,
	)
	model = model.to(device)

	
	transform = transforms.Compose([
		#transforms.ToTensor(),
		transforms.Normalize(
			mean= mean, std=  std ) #tensor([137.2619, 121.9672, 108.6618]) tensor([64.2384, 62.3150, 63.4915])
	])
	train_set = PerturbedCIFAR10(X, Y, transform)
	train_loader = torch.utils.data.DataLoader(
		train_set, batch_size=args.batch_size, shuffle=True, num_workers=2
	)

	# test_set = PerturbedCIFAR10(Xadv, Yadv, transform)
	# test_loader = torch.utils.data.DataLoader(
	# 	test_set, batch_size=args.batch_size, shuffle=True, num_workers=2
	# )

	test_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Lambda(lambda x: x.mul(255)),
		transforms.Normalize(
			mean= mean, std=  std ) 
	])	
	test_set = torchvision.datasets.CIFAR10(
		root='./data', train=False, download=True, transform=test_transform
	)
	test_loader = torch.utils.data.DataLoader(
		test_set, args.test_batch_size, shuffle=False, num_workers=2
	)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

	#
	#args.epochs = 0
	for epoch in range(1, args.epochs + 1):
		train(epoch, model, train_loader, optimizer, criterion, args)
		test(model, test_loader, criterion, args)
		torch.save(model.state_dict(), model_name+"_epoch-"+str(epoch)+"_ratio-"+str(ratio)+".pth")
	
	########################################
	########################################
	########################################
	right = 0.0
	Xadv, Yadv = [], []
	Xtest, Ytest, namestest = load_data('cifar-10-batches-py/test_batch')
	Xtest = np.transpose(Xtest,(1,0))
	Ytest = np.transpose(Ytest,(1,0))
	for i in range(len(Xtest)):
		result, flag = generate_adversarial(Xtest[i].reshape(3,32,32).transpose([1, 2, 0]), Ytest[i], model)
		if flag == 1:
			result = torch.tensor(np.transpose(result,(2,0,1)))
			Xadv.append(result)
			Yadv.append(Ytest[i])
		else:
			right += 1.0
	print("result:",  str(right/len(Xtest)))
	########################################
	########################################
	########################################

	# TODO 3. Test the trained network on adversarial examples: https://github.com/MadryLab/cifar10_challenge

	# First generate the dataset

	return


if __name__ == "__main__":
	main()
