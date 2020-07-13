# Core libraries
import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

# PyTorch
import torch
from torch.utils import data
from torch.autograd import Variable

# Local libraries
from utilities.utils import Utilities
from models.embeddings import resnet50

# Import our dataset class
from datasets.OpenSetCows2020.OpenSetCows2020 import OpenSetCows2020

"""
File for inferring the embeddings of the test portion of a selected database and
evaluating its classification performance using KNN
"""

# For a trained model, let's evaluate it
def evaluateModel(args):
	# Load the relevant datasets
	train_dataset = Utilities.selectDataset(args, True)
	test_dataset = Utilities.selectDataset(args, False)

	# Get the embeddings and labels of the training set and testing set
	train_embeddings, train_labels = inferEmbeddings(args, train_dataset, "train")
	test_embeddings, test_labels = inferEmbeddings(args, test_dataset, "test")

	# Classify them
	accuracy = KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels)

	# Write it out to the console so that subprocess can pick them up and close
	sys.stdout.write(f"Accuracy={str(accuracy)}")
	sys.stdout.flush()
	sys.exit(0)

# Use KNN to classify the embedding space
def KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels, n_neighbors=5):
    # Define the KNN classifier
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-4)

    # Give it the embeddings and labels of the training set
    neigh.fit(train_embeddings, train_labels)

    # Total number of testing instances
    total = len(test_labels-1)

    # Get the predictions from KNN
    predictions = neigh.predict(test_embeddings)

    # How many were correct?
    correct = (predictions == test_labels).sum()

    # Compute accuracy
    accuracy = (float(correct) / total) * 100

    return accuracy

# Infer the embeddings for a given dataset
def inferEmbeddings(args, dataset, split):
	# Wrap up the dataset in a PyTorch dataset loader
	data_loader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=6, shuffle=False)

	# Define our embeddings model
	model = resnet50(pretrained=True, num_classes=dataset.getNumClasses(), ckpt_path=args.model_path, embedding_size=args.embedding_size)
	
	# Put the model on the GPU and in evaluation mode
	model.cuda()
	model.eval()

	# Embeddings/labels to be stored on the testing set
	outputs_embedding = np.zeros((1,args.embedding_size))
	labels_embedding = np.zeros((1))
	total = 0
	correct = 0

	# Iterate through the testing portion of the dataset and get
	for images, _, _, labels, _ in tqdm(data_loader, desc=f"Inferring {split} embeddings"):
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
	
	# If we're supposed to be saving the embeddings and labels to file
	if args.save_embeddings:
		# Construct the save path
		save_path = os.path.join(args.save_path, f"{split}_embeddings.npz")
		
		# Save the embeddings to a numpy array
		np.savez(save_path,  embeddings=outputs_embedding, labels=labels_embedding)

	return outputs_embedding, labels_embedding

# Main/entry method
if __name__ == '__main__':
	# Collate command line arguments
	parser = argparse.ArgumentParser(description='Params')

	# Required arguments
	parser.add_argument('--model_path', nargs='?', type=str, required=True, 
						help='Path to the saved model to load weights from')
	parser.add_argument('--folds_file', type=str, default="", required=True,
						help="The file containing known/unknown splits")
	parser.add_argument('--save_path', type=str, required=True,
						help="Where to store the embeddings")

	parser.add_argument('--dataset', nargs='?', type=str, default='OpenSetCows2020', 
						help='Which dataset to use')
	parser.add_argument('--batch_size', nargs='?', type=int, default=16,
						help='Batch Size')
	parser.add_argument('--embedding_size', nargs='?', type=int, default=128, 
						help='Size of the dense layer for inference')
	parser.add_argument('--current_fold', type=int, default=0,
						help="The current fold we'd like to test on")
	parser.add_argument('--save_embeddings', type=bool, default=True,
						help="Should we save the embeddings to file")
	args = parser.parse_args()

	# Let's infer some embeddings
	evaluateModel(args)
