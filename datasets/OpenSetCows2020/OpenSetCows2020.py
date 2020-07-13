# Core libraries
import os
import sys
import cv2
import json
import random
import numpy as np

# PyTorch
import torch
from torch.utils import data

# Local libraries
from utilities.ioutils import *

"""
Manages loading the dataset into a PyTorch form
"""

class OpenSetCows2020(data.Dataset):
	# Class constructor
	def __init__(	self,
					fold,
					fold_file,
					split="train",
					combine=False,
					known=True,
					transform=False,
					img_size=(224, 224),
					suppress_info=True	):
		"""
		Class attributes
		"""

		# The root directory for the dataset itself
		self.__root = "datasets/OpenSetCows2020"

		# The fold we're currently considering
		self.__fold = str(fold)

		# The file containing the category splits for this fold
		self.__fold_file = fold_file

		# The split we're after (e.g. train/test)
		self.__split = split

		# Whether we should just load everything
		self.__combine = combine

		# Whether we're after known or unknown categories, irrelevant if combine is true
		self.__known = known

		# Whether to transform images/labels into pyTorch form
		self.__transform = transform

		# The directory containing actual imagery
		self.__train_images_dir = os.path.join(self.__root, "images/train")
		self.__test_images_dir = os.path.join(self.__root, "images/test")

		# Retrieve the number of classes from these
		self.__train_folders = allFoldersAtDir(self.__train_images_dir)
		self.__test_folders = allFoldersAtDir(self.__test_images_dir)
		assert len(self.__train_folders) == len(self.__test_folders)
		self.__num_classes = len(self.__train_folders)

		# Load the folds dictionary containing known and unknown categories for each fold
		if os.path.exists(self.__fold_file):
			with open(self.__fold_file, 'rb') as handle:
				self.__folds_dict = json.load(handle)
		else: 
			print(f"File path doesn't exist: {self.__fold_file}")
			sys.exit(1)

		# A quick check
		assert self.__fold in self.__folds_dict.keys()

		# The image size to resize to
		self.__img_size = img_size

		# A dictionary storing seperately the list of image filepaths per category for 
		# training and testing
		self.__sorted_files = {}

		# A dictionary storing separately the complete lists of filepaths for training and
		# testing
		self.__files = {}

		"""
		Class setup
		"""

		# Create dictionaries of categories: filepaths
		train_files = {os.path.basename(f):allFilesAtDirWithExt(f, ".jpg") for f in self.__train_folders}
		test_files = {os.path.basename(f):allFilesAtDirWithExt(f, ".jpg") for f in self.__test_folders}

		# List of categories to be removed
		remove = []

		# Should we be just returning all the classes (not removing unknown or known classes)
		if not self.__combine:
			# Create a list of categories to be removed according to whether we're after known
			# or unknown classes
			if self.__known: 	remove = self.__folds_dict[self.__fold]['unknown']
			else: 				remove = self.__folds_dict[self.__fold]['known']

		# Remove these from the dictionaries (might not remove anything)
		self.__sorted_files['train'] = {k:v for (k,v) in train_files.items() if k not in remove}
		self.__sorted_files['test'] = {k:v for (k,v) in test_files.items() if k not in remove}

		# Consolidate this into one long list of filepaths for training and testing
		train_list = [v for k,v in self.__sorted_files['train'].items()]
		test_list = [v for k,v in self.__sorted_files['test'].items()]
		self.__files['train'] = [item for sublist in train_list for item in sublist]
		self.__files['test'] = [item for sublist in test_list for item in sublist]

		# Report some things
		if not suppress_info: self.printStats()

	"""
	Superclass overriding methods
	"""

	# Get the number of items for this dataset (depending on the split)
	def __len__(self):
		return len(self.__files[self.__split])

	# Index retrieval method
	def __getitem__(self, index):
		# Get and load the anchor image
		img_path = self.__files[self.__split][index]
		
		# Load the anchor image
		img_anchor = loadResizeImage(img_path, self.__img_size)
		
		# Retrieve the class/label this index refers to
		current_category = self.__retrieveCategoryForFilepath(img_path)

		# Get a positive (another random image from this class)
		img_pos = self.__retrievePositive(current_category, img_path)

		# Get a negative (a random image from a different random class)
		img_neg, label_neg = self.__retrieveNegative(current_category, img_path)

		# Convert all labels into numpy form
		label_anchor = np.array([int(current_category)])
		label_neg = np.array([int(label_neg)])

		# For sanity checking, visualise the triplet
		# self.__visualiseTriplet(img_anchor, img_pos, img_neg, label_anchor)

		# Transform to pyTorch form
		if self.__transform:
			img_anchor, img_pos, img_neg = self.__transformImages(img_anchor, img_pos, img_neg)
			label_anchor, label_neg = self.__transformLabels(label_anchor, label_neg)

		return img_anchor, img_pos, img_neg, label_anchor, label_neg

	"""
	Public methods
	"""	

	# Print stats about the current state of this dataset
	def printStats(self):
		print("Loaded the OpenSetCows2019 dataset_____________________________")
		print(f"Fold = {int(self.__fold)+1}, split = {self.__split}, combine = {self.__combine}, known = {self.__known}")
		print(f"Found {self.__num_classes} categories: {len(self.__folds_dict[self.__fold]['known'])} known, {len(self.__folds_dict[self.__fold]['unknown'])} unknown")
		print(f"With {len(self.__files['train'])} train images, {len(self.__files['test'])} test images")
		print(f"Unknown categories {self.__folds_dict[self.__fold]['unknown']}")
		print("_______________________________________________________________")

	"""
	(Effectively) private methods
	"""

	def __visualiseTriplet(self, image_anchor, image_pos, image_neg, label_anchor):
		print(f"Label={label_anchor}")
		cv2.imshow(f"Label={label_anchor} anchor", image_anchor)
		cv2.imshow(f"Label={label_anchor} positive", image_pos)
		cv2.imshow(f"Label={label_anchor} negative", image_neg)
		cv2.waitKey(0)

	# Transform the numpy images into pyTorch form
	def __transformImages(self, img_anchor, img_pos, img_neg):
		# Firstly, transform from NHWC -> NCWH
		img_anchor = img_anchor.transpose(2, 0, 1)
		img_pos = img_pos.transpose(2, 0, 1)
		img_neg = img_neg.transpose(2, 0, 1)

		# Now convert into pyTorch form
		img_anchor = torch.from_numpy(img_anchor).float()
		img_pos = torch.from_numpy(img_pos).float()
		img_neg = torch.from_numpy(img_neg).float()

		return img_anchor, img_pos, img_neg

	# Transform the numpy labels into pyTorch form
	def __transformLabels(self, label_anchor, label_neg):
		# Convert into pyTorch form
		label_anchor = torch.from_numpy(label_anchor).long()
		label_neg = torch.from_numpy(label_neg).long()

		return label_anchor, label_neg

	# Print some info about the distribution of images per category
	def __printImageDistribution(self):
		for category, filepaths in self.__sorted_files[self.__split].items():
			print(category, len(filepaths))

	# For a given filepath, return the category which contains this filepath
	def __retrieveCategoryForFilepath(self, filepath):
		# Iterate over each category
		for category, filepaths in self.__sorted_files[self.__split].items():
			if filepath in filepaths: return category

	# Get another positive sample from this class
	def __retrievePositive(self, category, filepath):
		# Copy the list of possible positives and remove the anchor
		possible_list = list(self.__sorted_files[self.__split][category])
		assert filepath in possible_list
		possible_list.remove(filepath)

		# Randomly select a filepath
		img_path = random.choice(possible_list)

		# Load and return the image
		img = loadResizeImage(img_path, self.__img_size)
		return img

	def __retrieveNegative(self, category, filepath):
		# Get the list of categories and remove that of the anchor
		possible_categories = list(self.__sorted_files[self.__split].keys())
		assert category in possible_categories
		possible_categories.remove(category)

		# Randomly select a category
		random_category = random.choice(possible_categories)

		# Randomly select a filepath in that category
		img_path = random.choice(self.__sorted_files[self.__split][random_category])

		# Load and return the image along with the selected label
		img = loadResizeImage(img_path, self.__img_size)
		return img, random_category

	"""
	Getters
	"""

	def getNumClasses(self):
		return self.__num_classes

	def getNumTrainingFiles(self):
		return len(self.__files["train"])

	def getNumTestingFiles(self):
		return len(self.__files["test"])

	"""
	Setters
	"""

	"""
	Static methods
	"""

# Entry method/unit testing method
if __name__ == '__main__':
	pass