# Core libraries
import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm

# PyTorch
import torch
from torch.utils import data
from torch.autograd import Variable

# Blue pebble
if os.path.isdir("/home/ca0513"): 
	sys.path.append("home/ca0513/ATI-Pilot-Project/src/")
	os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Local libraries
from utilities.utils import *
from models.embeddings import *

# Import our dataset class
from datasets.OpenSetCows2020.OpenSetCows2020 import OpenSetCows2020

"""
File for inferring the embeddings of the test portion of a selected database
"""

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

	dataset = OpenSetCows2020(args.current_fold, args.fold_file, suppress_info=False, transform=True, split=args.split, combine=combine, known=known)
	n_classes = t_loader.getNumClasses()

	# Wrap up the data in a PyTorch dataset loader
	data_loader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=6, shuffle=False)

	# Define our embeddings model
	model = embeddings(pretrained=True,  num_classes=dataset.getNumClasses(), ckpt_path=args.ckpt_path, embedding_size=args.embedding_size)
	
	# Put the model on the GPU and in evaluation mode
	model.cuda()
	model.eval()

	# Embeddings/labels to be stored on the testing set
	output_embedding = np.array([])
	outputs_embedding = np.zeros((1,args.embedding_size))#128
	labels_embedding = np.zeros((1))
	total = 0
	correct = 0

	# Iterate through the testing portion of the dataset and get
	for images, _, _, labels, _ in tqdm(data_loader):
		# Put the images on the GPU and express them as PyTorch variables
		images = Variable(images.cuda())

		# Get the embeddings of this batch of images
		outputs = model(images)

		# Express embeddings in numpy form
		embeddings = outputs.data
		embeddings = embeddings.cpu().numpy()

		# Convert labels to readable numpy form
		labels = labels.view(len(labels))
		labels = labels.cpu().numpy()

		# Store testing data on this batch ready to be evaluated
		outputs_embedding = np.concatenate((outputs_embedding,embeddings), axis=0)
		labels_embedding = np.concatenate((labels_embedding,labels), axis=0)
	
	# Construct the save path
	save_path = root_dir + args.split + "_" + os.path.split(args.ckpt_path)[1][:-4] + "_" + args.instances
	
	# Save the embeddings to a numpy array
	np.savez(save_path,  embeddings=outputs_embedding, labels=labels_embedding)

# Main/entry method
if __name__ == '__main__':
	# Collate command line arguments
	parser = argparse.ArgumentParser(description='Params')
	parser.add_argument('--ckpt_path', nargs='?', type=str, default='', 
						help='Path to the saved model')
	parser.add_argument('--dataset', nargs='?', type=str, default='cow_id', 
						help='Dataset to use [\'pascal, camvid, ade20k etc\']')
	parser.add_argument('--instances', nargs='?', type=str, default='full',
						help='Dataset split to use [\'full, known, novel\']')
	parser.add_argument('--batch_size', nargs='?', type=int, default=5, #7 
						help='Batch Size')
	parser.add_argument('--embedding_size', nargs='?', type=int, default=128, #7 
						help='Size of the dense layer for inference')
	parser.add_argument('--split', nargs='?', type=str, default='test', 
						help='Dataset split to use [\'train, eval\']')
	parser.add_argument('--current_fold', type=int, default=0,
						help="The current fold we'd like to test on")
	parser.add_argument('--fold_file', type=str, default="", required=True,
						help="The file containing known/unknown splits")
	args = parser.parse_args()

	# Let's infer some embeddings
	test(args)
