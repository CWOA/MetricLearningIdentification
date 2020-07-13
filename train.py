# Core libraries
import os
import sys
import argparse
from tqdm import tqdm

# PyTorch stuff
import torch
from torch.autograd import Variable

# Local libraries
from utilities.loss import *
from utilities.mining_utils import *
from utilities.utils import Utilities

"""
File is for training the network via cross fold validation
"""

# Let's cross validate
def crossValidate(args):
	# Loop through each fold for cross validation
	for k in range(args.fold_number, args.num_folds):
		print(f"Beginning training for fold {k+1} of {args.num_folds}")

		# Directory for storing data to do with this fold
		args.fold_out_path = os.path.join(args.out_path, f"fold_{k}")

		# Create a folder in the results folder for this fold as well as to store embeddings
		os.makedirs(args.fold_out_path, exist_ok=True)

		# Store the current fold
		args.current_fold = k

		# Let's train!
		trainFold(args)

# Train for a single fold
def trainFold(args):
	# Create a new instance of the utilities class for this fold
	utils = Utilities(args)

	# Let's prepare the objects we need for training based on command line arguments
	data_loader, model, loss_fn, optimiser = utils.setupForTraining(args)

	# Training tracking variables
	global_step = 0 
	accuracy_best = 0

	# Main training loop
	for epoch in tqdm(range(args.num_epochs), desc="Training epochs"):
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
					utils.logTrainInfo(	epoch, global_step, loss.item(), 
										loss_triplet=triplet_loss.item(), 
										loss_softmax=loss_softmax.item()	)
				else:
					utils.logTrainInfo(epoch, global_step, loss.item()) 

		# Every x epochs, let's evaluate on the validation set
		if epoch % args.eval_freq == 0:
			# Temporarily save model weights for the evaluation to use
			utils.saveCheckpoint(epoch, model, optimiser, "current")

			# Test on the validation set
			accuracy_curr = utils.test(global_step)

			# Save the model weights as the best if it surpasses the previous best results
			if accuracy_curr > accuracy_best:
				utils.saveCheckpoint(epoch, model, optimiser, "best")
				accuracy_best = accuracy_curr

# Main/entry method
if __name__ == '__main__':
	# Collate command line arguments
	parser = argparse.ArgumentParser(description='Parameters for network training')

	# File configuration (the only required arguments)
	parser.add_argument('--out_path', type=str, default="", required=True,
						help="Path to folder to store results in")
	parser.add_argument('--folds_file', type=str, default="", required=True,
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
	parser.add_argument('--loss_function', type=str, default='OnlineReciprocalSoftmaxLoss',
						help='Which loss function to use: [TripletLoss, TripletSoftmaxLoss, \
						OnlineTripletLoss, OnlineTripletSoftmaxLoss, OnlineReciprocalTripletLoss, \
						OnlineReciprocalSoftmaxLoss]')

	# Hyperparameters
	parser.add_argument('--img_rows', nargs='?', type=int, default=224, 
						help='Height of the input image')
	parser.add_argument('--img_cols', nargs='?', type=int, default=224, 
						help='Height of the input image')
	parser.add_argument('--embedding_size', nargs='?', type=int, default=128, 
						help='dense layer size for inference')
	parser.add_argument('--num_epochs', nargs='?', type=int, default=500, 
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
