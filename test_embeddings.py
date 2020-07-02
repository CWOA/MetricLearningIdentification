import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random

import sys, os
sys.path.append('models')
sys.path.append('.')
sys.path.append('../../')

# Local machine
if os.path.isdir("/home/will"): sys.path.append("/home/will/work/1-RA/src")

# Blue pebble
if os.path.isdir("/home/ca0513"): 
	sys.path.append("home/ca0513/ATI-Pilot-Project/src/")
	os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Blue crystal 4
if os.path.isdir("/mnt/storage/home/ca0513"):
	sys.path.append("/mnt/storage/home/ca0513/ATI-Pilot-Project/src/")
	os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
from torch import optim
import subprocess

from loader import get_loader, get_data_path
from embeddings import embeddings
from utils import *

from Datasets.OpenSetCows2019 import OpenSetCows2019
from Utilities.DataUtils import DataUtils

# Infer on a folder full of images
def inferOnImageFolder():
	# Path to folder full of images
	folder_path = "/home/ca0513/CEiA/other/imperfect-detections"

	# List of images
	image_fps = DataUtils.allFilesAtDirWithExt(folder_path, ".jpg")

	# Path to the weights we'd like to load
	weights_path = "/home/ca0513/CEiA/results/SoftmaxRTL/50-50/fold_0/triplet_cnn_open_cows_best_x1.pkl"

	# Load and define our model
	model = embeddings(pretrained=True, num_classes=46, ckpt_path=weights_path)
	model.cuda()
	model.eval()

	# List of embeddings
	embeddings = []

	# Iterate over each image
	for image in image_fps:
		# Load the image
		img = cv2.imread(image)

		# Convert it to the required format
		img = img.transpose(2,0,1)
		img = torch.from_numpy(img).float()
		img = Variable(img.cuda())

		# Infer on it
		embedding = model(img)

		# Add it to the list
		embeddings.append(embedding)

	# Save the list to file
	save_path = "/home/ca0513/CEiA/other/imperfect-embeddings.npz"
	np.savez(save_path, embeddings=np.array(embeddings))

def test(args):
	# Setup image
	# train/eval
	# Setup Dataloader
	root_dir = os.path.split(args.ckpt_path)[0] + "/" #"/media/alexa/DATA/Miguel/results/" + args.dataset +"/triplet_cnn/" 
	#print(args.ckpt_path)

	# data_loader = get_loader("triplet_resnet_" + args.dataset)
	# data_path = get_data_path(args.dataset)

	# # All, novel or known splits 
	# instances = get_instances(args)

	# t_loader = data_loader(data_path, is_transform=True, 
	#     split=args.split,
	#     img_size=(args.img_rows, args.img_cols), 
	#     augmentations=None, 
	#     instances=instances)
	# n_classes = t_loader.n_classes

	combine=False
	known=True
	if args.instances == "full": combine = True
	elif args.instances == "known": known = True
	elif args.instances == "novel": known = False

	t_loader = OpenSetCows2019(args.current_fold, args.fold_file, suppress_info=False, transform=True, split=args.split, combine=combine, known=known)
	n_classes = t_loader.getNumClasses()

	trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=6, shuffle=False)

	# Define/load our embeddings model
	model = embeddings(pretrained=True,  num_classes=n_classes, ckpt_path=args.ckpt_path, embedding_size=args.embedding_size)
	
	# Put the model on the GPU
	model.cuda()

	# Put the model in evaluation mode
	model.eval()

	output_embedding = np.array([])
	outputs_embedding = np.zeros((1,args.embedding_size))#128
	labels_embedding = np.zeros((1))
	path_imgs = []
	total = 0
	correct = 0

	#for i, (images, labels, path_img) in enumerate(tqdm(trainloader)):
	# for i, (images, labels, path_img) in enumerate(trainloader):
	for i, (images, img_pos, img_neg, labels, _) in enumerate(trainloader):
		images = Variable(images.cuda())
		labels = labels.view(len(labels))
		labels = labels.cpu().numpy()
		#labels = Variable(labels.cuda())
		outputs = model(images)
		output_embedding = outputs.data
		output_embedding = output_embedding.cpu().numpy()

		outputs_embedding = np.concatenate((outputs_embedding,output_embedding), axis=0)
		labels_embedding = np.concatenate((labels_embedding,labels), axis=0)
		#path_imgs.extend(path_img)
	#print(root_dir + args.split + "_" + os.path.split(args.ckpt_path)[1][:-4] + "_" + args.instances)
	
	# print(root_dir)
	save_path = root_dir + args.split + "_" + os.path.split(args.ckpt_path)[1][:-4] + "_" + args.instances
	np.savez(save_path,  embeddings=outputs_embedding, labels=labels_embedding, filenames=path_imgs)
	# print(f"Saved to: {save_path}")
	#print ('Done: ')


	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Params')
	parser.add_argument('--id', nargs='?', type=str, default='', 
						help='Architecture to use [\'fcn8s, unet, segnet etc\']')
	parser.add_argument('--ckpt_path', nargs='?', type=str, default='', 
						help='Path to the saved model')
	parser.add_argument('--test_path', nargs='?', type=str, default='.', 
						help='Path to saving results')
	parser.add_argument('--dataset', nargs='?', type=str, default='cow_id', 
						help='Dataset to use [\'pascal, camvid, ade20k etc\']')
	parser.add_argument('--instances', nargs='?', type=str, default='full',
						help='Dataset split to use [\'full, known, novel\']')
	parser.add_argument('--img_path', nargs='?', type=str, default=None, 
						help='Path of the input image')
	parser.add_argument('--img_rows', nargs='?', type=int, default=224, 
						help='Height of the input image')
	parser.add_argument('--img_cols', nargs='?', type=int, default=224, 
						help='Height of the input image')
	parser.add_argument('--batch_size', nargs='?', type=int, default=5, #7 
						help='Batch Size')
	parser.add_argument('--embedding_size', nargs='?', type=int, default=128, #7 
						help='Size of the dense layer for inference')
	parser.add_argument('--split', nargs='?', type=str, default='test', 
						help='Dataset split to use [\'train, eval\']')
	parser.add_argument('--current_fold', type=int, default=0,
						help="LEAVE: The current fold")
	parser.add_argument('--fold_file', type=str, default="")

	args = parser.parse_args()

	test(args)
