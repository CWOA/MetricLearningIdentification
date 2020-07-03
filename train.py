# Core libraries
import os
import sys
import argparse
from tqdm import tqdm

# Blue pebble
if os.path.isdir("/home/ca0513"): 
	sys.path.append("/home/ca0513/ATI-Pilot-Project/src/")
	sys.path.append("/home/ca0513/ATI-Pilot-Project/src/Identifier/Supervisted-Triplet-Network")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# PyTorch stuff
import torch
from torch import optim
from torch.utils import data
from torch.autograd import Variable

# Local libraries
from utilities.loss import *
from utilities.utils import *
from utilities.mining_utils import *
from models.triplet_resnet import *
from models.triplet_resnet_softmax import *

# Import our dataset class
from datasets.OpenSetCows2020.OpenSetCows2020 import OpenSetCows2020

"""
File is for training the network via cross fold validation
"""

# Let's cross validate
def crossValidate(args):
	# Loop through each fold for cross validation
	for k in range(args.fold_number, args.num_folds):
		print(f"Beginning training for fold {k+1} of {args.num_folds}")

		# Directory for storing data to do with this fold
		store_dir = os.path.join(args.out_path, f"fold_{k}")

		# Change where to store things
		args.ckpt_path = store_dir

		# Create a folder in the results folder for this fold as well as to store embeddings
		os.makedirs(store_dir, exist_ok=True)

		# Let's train!
		trainFold(args, fold=k)

# Preparations for training for a particular fold
def setup(args, fold):
	# Load the selected dataset
	if args.dataset == "OpenSetCows2020":
		dataset = OpenSetCows2019(fold, args.folds_file, transform=True, suppress_info=False)
	elif args.dataset == "ADD YOUR DATASET HERE":
		pass
	else:
		print(f"Dataset choice not recognised, exiting.")
		sys.exit(1)

	# Print some information about the dataset
	print(f"Found {dataset.getNumTrainingFiles()} training images, {dataset.getNumTestingFiles()} testing images")

	# Wrap up the data in a PyTorch dataset loader
	data_loader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=6, shuffle=True)

	# Setup the selected model
	if args.model == "TripletResnetSoftmax": 
		model = triplet_resnet50_softmax(pretrained=True, num_classes=dataset.getNumClasses())
	if args.model == "TripletResnet": 
		model = triplet_resnet50(pretrained=True, num_classes=dataset.getNumClasses())
	else:
		print(f"Model choice not recognised, exiting.")
		sys.exit(1)

	# Put the model on the GPU and in training mode
	model.cuda()
	model.train()

	# Setup the triplet selection method
	if args.triplet_selection == "HardestNegative":
		triplet_selector = HardestNegativeTripletSelector(margin=args.triplet_margin)
	elif args.triplet_selection == "RandomNegative":
		triplet_selector = RandomNegativeTripletSelector(margin=args.triplet_margin)
	elif args.triplet_selection == "SemihardNegative":
		triplet_selector = SemihardNegativeTripletSelector(margin=args.triplet_margin)
	elif args.triplet_selection == "AllTriplets":
		triplet_selector = AllTripletSelector()
	else:
		print(f"Triplet selection choice not recognised, exiting.")
		sys.exit(1)

	# Setup the selected loss function
	if args.loss_function == "TripletLoss":
		loss_fn = TripletLoss(margin=args.triplet_margin)
	elif args.loss_function == "TripletSoftmaxLoss":
		loss_fn = TripletSoftmaxLoss(margin=args.triplet_margin)
	elif args.loss_function == "OnlineTripletLoss":	
		loss_fn = OnlineTripletLoss(triplet_selector, margin=args.triplet_margin)
	elif args.loss_function == "OnlineTripletSoftmaxLoss":
		loss_fn = OnlineTripletSoftmaxLoss(triplet_selector, margin=args.triplet_margin)
	elif args.loss_function == "OnlineReciprocalTripletLoss":
		loss_fn = OnlineReciprocalTripletLoss(triplet_selector)
	elif args.loss_function == "OnlineReciprocalSoftmaxLoss":
		loss_fn = OnlineReciprocalSoftmaxLoss(triplet_selector)
	else:
		print(f"Loss function choice not recognised, exiting.")
		sys.exit(1)

	# Create our optimiser, if using reciprocal triplet loss, don't have a momentum component
	if "Reciprocal" in args.loss_function:
		optimiser = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=optimiser.weight_decay)
	else:
		optimiser = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=optimiser.weight_decay)

	# Print some details about the setup we've employed
	show_setup(args, dataset.getNumClasses(), optimizer, loss_fn)

	return data_loader, model, loss_fn, optimiser

# Train for a single fold
def trainFold(args, fold=0):
	# Let's prepare the objects we need for training based on command line arguments
	data_loader, model, loss_fn, optimiser = setup(args, fold)

	# Training tracking variables
	global_step = 0 
	accuracy_best = 0

	# Main training loop
	for epoch in tqdm(range(args.n_epoch)):
		# Mini-batch training loop over the training set
		for images, images_pos, images_neg, labels, labels_neg in data_loader:
			# Put the images on the GPU and express them as PyTorch variables
			images = Variable(images.cuda())
			images_pos = Variable(images_pos.cuda())
			images_neg = Variable(images_neg.cuda())
		   
			# Zero the optimiser
			optimiser.zero_grad()

			# Get the embeddings/predictions for each
			if "Softmax" in args.loss_function:
				embed_anch, embed_pos, embed_neg, preds = model(images, images_pos, images_neg)
			else:
				embed_anch, embed_pos, embed_neg = model(images, images_pos, images_neg)

			# Calculate the loss on this minibatch
			if "Softmax" in args.loss_function:
				loss, triplet_loss, loss_softmax = loss_fn(embed_anch, embed_pos, embed_neg, preds, labels, labels_neg)
			else:
				loss = loss_fn(embed_anch, embed_pos, embed_neg, labels)

			# Backprop and optimise
			loss.backward()
			optimiser.step()
			global_step += 1

			# Log the loss if its time to do so
			if global_step % args.logs_freq == 0:
				if "Softmax" in args.loss_function:
					log_loss(	epoch, args.n_epoch, global_step, 
								loss_mean=loss.item(), 
								loss_triplet=triplet_loss.item(), 
								loss_softmax=loss_softmax.item()	)
				else:
					log_loss(epoch, args.n_epoch, global_step, loss_mean=loss.item()) 

		# Every x epochs, let's evaluate on the validation set
		if epoch % args.eval_freq == 0:
			# Temporarily save model weights for the evaluation to use
			save_checkpoint(epoch, model, optimizer, "temp")

			# Test on the validation set
			accuracy_curr = eval_model(fold, args.folds_file, global_step, args.instances_to_eval)

			# Save the model weights as the best if it surpasses the previous best results
			if accuracy_curr > accuracy_best:
				save_checkpoint(epoch, model, optimizer, "best")
				accuracy_best = accuracy_curr

# Main/entry method
if __name__ == '__main__':
	# Collate command line arguments
	parser = argparse.ArgumentParser(description='Parameters for network training')

	# File configuration
	parser.add_argument('--id', nargs='?', type=str, default='open_cows',
						help='Experiment identifier')
	parser.add_argument('--out_path', type=str, default="", required=True,
						help="Path to folder to store results in")
	parser.add_argument('--folds_file', type=str, default="", required=True
						help="Path to json file containing folds")

	# Core settings
	parser.add_argument('--num_folds', type=int, default=1,
						help="Number of folds to cross validate across")
	parser.add_argument('--fold_number', type=int, default=0,
						help="The fold number to START at")
	parser.add_argument('--dataset', type=str, default='OpenSetCows2020',
						help='Which dataset to use')
	parser.add_argument('--model', type=str, default='TripletResnetSoftmax',
						help='Which model to use: [TripletResnetSoftmax, TripletResnet]')
	parser.add_argument('--triplet_selection', type=str, default='HardestNegative',
						help='Which triplet selection method to use: [HardestNegative, RandomNegative,\
						SemihardNegative, AllTriplets]')
	parser.add_argument('--loss_fn', type=str, default='OnlineReciprocalSoftmaxLoss',
						help='Which loss function to use: [TripletLoss, TripletSoftmaxLoss, \
						OnlineTripletLoss, OnlineTripletSoftmaxLoss, OnlineReciprocalTripletLoss, \
						OnlineReciprocalSoftmaxLoss]')

	# ARE THESE NECESSARY?
	parser.add_argument('--instances', nargs='?', type=str, default='known',
						help='Train Dataset split to use [\'full, known, novel\']')
	parser.add_argument('--instances_to_eval', nargs='?', type=str, default='all',
						help='Test Dataset split to use [\'full, known, novel, all\']')

	# Hyperparameters
	parser.add_argument('--img_rows', nargs='?', type=int, default=224, 
						help='Height of the input image')
	parser.add_argument('--img_cols', nargs='?', type=int, default=224, 
						help='Height of the input image')
	parser.add_argument('--embedding_size', nargs='?', type=int, default=128, 
						help='dense layer size for inference')
	parser.add_argument('--n_epoch', nargs='?', type=int, default=500, 
						help='# of the epochs to train for')
	parser.add_argument('--batch_size', nargs='?', type=int, default=16,
						help='Batch Size')
	parser.add_argument('--learning_rate', type=float, default=0.001,
						help="Optimiser learning rate")
	parser.add_argument('--weight_decay', type=float, default=1e-4,
						help="Weight decay")
	parser.add_argument('--triplet_margin', type=float, default=0.5,
						help="Margin parameter for triplet loss")

	# Training settings
	parser.add_argument('--eval_freq', nargs='?', type=int, default=2,
						help='Frequency for evaluating model [epochs num]')
	parser.add_argument('--logs_freq', nargs='?', type=int, default=20,
						help='Frequency for saving logs [steps num]')

	args = parser.parse_args()

	# Let's cross validate!
	crossValidate(args)
