# Core libraries
import numpy as np

# PyTorch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""
File contains loss functions selectable during training
"""


class TripletLoss(nn.Module):
	def __init__(self, margin=4.0):
		super(TripletLoss, self).__init__()
		self.margin = margin
					
	def forward(self, anchor, positive, negative, labels):
		distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
		distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
		losses = F.relu(distance_positive - distance_negative + self.margin)
	
		return losses.sum()

class TripletSoftmaxLoss(nn.Module):
	def __init__(self, margin=0.0, lambda_factor=0.01):
		super(TripletSoftmaxLoss, self).__init__()
		self.margin = margin
		self.loss_fn = nn.CrossEntropyLoss()
		self.lambda_factor = lambda_factor
					
	def forward(self, anchor, positive, negative, outputs, labels ):
		distance_positive = torch.abs(anchor - positive).sum(1)
		distance_negative = torch.abs(anchor - negative).sum(1)
		losses = F.relu(distance_positive - distance_negative + self.margin)
		loss_softmax = self.loss_fn(input=outputs, target=labels)
		loss_total = self.lambda_factor*losses.sum() + loss_softmax

		return loss_total, losses.sum(), loss_softmax

class OnlineTripletLoss(nn.Module):
	def __init__(self, triplet_selector, margin=0.0):
		super(OnlineTripletLoss, self).__init__()
		self.margin = margin
		self.triplet_selector = triplet_selector

	def forward(self, anchor_embed, pos_embed, neg_embed, labels):
		# Combine the embeddings from each network
		embeddings = torch.cat((anchor_embed, pos_embed, neg_embed), dim=0)

		# Get the (e.g. hardest) triplets in this minibatch
		triplets, num_triplets = self.triplet_selector.get_triplets(embeddings, labels)

		# There might be no triplets selected, if so, just compute the loss over the entire
		# minibatch
		if num_triplets == 0:
			ap_distances = (anchor_embed - pos_embed).pow(2).sum(1)
			an_distances = (anchor_embed - neg_embed).pow(2).sum(1)
		else:
			# Use CUDA if we can
			if anchor_embed.is_cuda: triplets = triplets.cuda()

			# Compute triplet loss over the selected triplets
			ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
			an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)

		# Compute the losses
		losses = F.relu(ap_distances - an_distances + self.margin)

		return losses.mean()

class OnlineTripletSoftmaxLoss(nn.Module):
	def __init__(self, triplet_selector, margin=0.0, lambda_factor=0.01):
		super(OnlineTripletSoftmaxLoss, self).__init__()
		self.margin = margin
		self.loss_fn = nn.CrossEntropyLoss()
		self.lambda_factor = lambda_factor
		self.triplet_selector = triplet_selector
					
	def forward(self, anchor_embed, pos_embed, neg_embed, preds, labels, labels_neg):
		# Combine the embeddings from each network
		embeddings = torch.cat((anchor_embed, pos_embed, neg_embed), dim=0)

		# Define the labels as variables and put on the GPU
		gpu_labels = labels.view(len(labels))
		gpu_labels_neg = labels_neg.view(len(labels_neg))
		gpu_labels = Variable(gpu_labels.cuda())
		gpu_labels_neg = Variable(gpu_labels_neg.cuda())

		# Concatenate labels for softmax/crossentropy targets
		target = torch.cat((gpu_labels, gpu_labels, gpu_labels_neg), dim=0)

		# Get the (e.g. hardest) triplets in this minibatch
		triplets, num_triplets = self.triplet_selector.get_triplets(embeddings, labels)

		# There might be no triplets selected, if so, just compute the loss over the entire
		# minibatch
		if num_triplets == 0:
			ap_distances = (anchor_embed - pos_embed).pow(2).sum(1)
			an_distances = (anchor_embed - neg_embed).pow(2).sum(1)
		else:
			# Use CUDA if we can
			if anchor_embed.is_cuda: triplets = triplets.cuda()

			# Compute triplet loss over the selected triplets
			ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
			an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
		
		# Compute the triplet losses
		triplet_losses = F.relu(ap_distances - an_distances + self.margin)

		# Compute softmax loss		
		loss_softmax = self.loss_fn(input=preds, target=target-1)

		# Compute the total loss
		loss_total = self.lambda_factor*triplet_losses.mean() + loss_softmax

		# Return them all!
		return loss_total, triplet_losses.mean(), loss_softmax

# Reciprocal triplet loss from 
# "Who Goes There? Exploiting Silhouettes and Wearable Signals for Subject Identification
# in Multi-Person Environments"
class OnlineReciprocalTripletLoss(nn.Module):
	def __init__(self, triplet_selector):
		super(OnlineReciprocalTripletLoss, self).__init__()
		self.triplet_selector = triplet_selector

	def forward(self, anchor_embed, pos_embed, neg_embed, labels):
		# Combine the embeddings from each network
		embeddings = torch.cat((anchor_embed, pos_embed, neg_embed), dim=0)

		# Get the (e.g. hardest) triplets in this minibatch
		triplets, num_triplets = self.triplet_selector.get_triplets(embeddings, labels)

		# There might be no triplets selected, if so, just compute the loss over the entire
		# minibatch
		if num_triplets == 0:
			ap_distances = (anchor_embed - pos_embed).pow(2).sum(1)
			an_distances = (anchor_embed - neg_embed).pow(2).sum(1)
		else:
			# Use CUDA if we can
			if anchor_embed.is_cuda: triplets = triplets.cuda()

			# Compute distances over the selected triplets
			ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
			an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)

		# Actually compute reciprocal triplet loss
		losses = ap_distances + (1/an_distances)

		return losses.mean()

# Reciprocal triplet loss from 
# "Who Goes There? Exploiting Silhouettes and Wearable Signals for Subject Identification
# in Multi-Person Environments"
class OnlineReciprocalSoftmaxLoss(nn.Module):
	def __init__(self, triplet_selector, margin=0.0,  lambda_factor=0.01):
		super(OnlineReciprocalSoftmaxLoss, self).__init__()
		self.margin = margin
		self.loss_fn = nn.CrossEntropyLoss()
		self.lambda_factor = lambda_factor
		self.triplet_selector = triplet_selector
					
	def forward(self, anchor_embed, pos_embed, neg_embed, preds, labels, labels_neg):
		# Combine the embeddings from each network
		embeddings = torch.cat((anchor_embed, pos_embed, neg_embed), dim=0)

		# Define the labels as variables and put on the GPU
		gpu_labels = labels.view(len(labels))
		gpu_labels_neg = labels_neg.view(len(labels_neg))
		gpu_labels = Variable(gpu_labels.cuda())
		gpu_labels_neg = Variable(gpu_labels_neg.cuda())

		# Concatenate labels for softmax/crossentropy targets
		target = torch.cat((gpu_labels, gpu_labels, gpu_labels_neg), dim=0)

		# Get the (e.g. hardest) triplets in this minibatch
		triplets, num_triplets = self.triplet_selector.get_triplets(embeddings, labels)

		# There might be no triplets selected, if so, just compute the loss over the entire
		# minibatch
		if num_triplets == 0:
			ap_distances = (anchor_embed - pos_embed).pow(2).sum(1)
			an_distances = (anchor_embed - neg_embed).pow(2).sum(1)
		else:
			# Use CUDA if we can
			if anchor_embed.is_cuda: triplets = triplets.cuda()

			# Compute triplet loss over the selected triplets
			ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
			an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
		
		# Compute the triplet losses
		triplet_losses = ap_distances + (1/an_distances)

		# Compute softmax loss		
		loss_softmax = self.loss_fn(input=preds, target=target-1)

		# Compute the total loss
		loss_total = self.lambda_factor*triplet_losses.mean() + loss_softmax

		# Return them all!
		return loss_total, triplet_losses.mean(), loss_softmax
