#!/usr/bin/env python

# Core libraries
import os
import sys
sys.path.append("../")
import cv2
import json
import random
import numpy as np
from PIL import Image

# PyTorch
import torch
from torch.utils import data

# Local libraries
from utilities.DataUtils import DataUtils
from utilities.ImageUtils import ImageUtils

"""
Descriptor
"""

class OpenSetCows2019(data.Dataset):
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
		if os.path.exists("/home/will"): self.__root = "/home/will/work/1-RA/src/Datasets/data/OpenSetCows2019"
		elif os.path.exists("/home/ca0513"): self.__root = "/home/ca0513/ATI-Pilot-Project/src/Datasets/data/OpenSetCows2019"
		elif os.path.exists("/mnt/storage/home/ca0513"): self.__root = "/mnt/storage/home/ca0513/ATI-Pilot-Project/src/Datasets/data/OpenSetCows2019"

		# The fold we're currently considering
		self.__fold = fold

		# The file containing the category splits for this fold
		self.__fold_file = fold_file

		# The split we're after (e.g. train/test)
		self.__split = split

		# Whether we should just load everything
		self.__combine = combine

		# Whether we're after known or unknown categories
		self.__known = known

		# Whether to transform images/labels into pyTorch form
		self.__transform = transform

		# The directory containing actual imagery
		self.__train_images_dir = os.path.join(self.__root, "split/train")
		self.__test_images_dir = os.path.join(self.__root, "split/test")

		# Retrieve the number of classes from these
		self.__train_folders = DataUtils.allFoldersAtDir(self.__train_images_dir)
		self.__test_folders = DataUtils.allFoldersAtDir(self.__test_images_dir)
		assert len(self.__train_folders) == len(self.__test_folders)
		self.__num_classes = len(self.__train_folders)

		# Load the folds dictionary containing known and unknown categories for each fold
		if os.path.exists(os.path.join(self.__root, self.__fold_file)):
			with open(os.path.join(self.__root, self.__fold_file), 'rb') as handle:
				self.__folds_dict = json.load(handle)
		else: 
			print(f"File path doesn't exist: {os.path.join(self.__root, self.__fold_file)}")
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
		Class objects
		"""

		"""
		Class setup
		"""

		# Create dictionaries of categories: filepaths
		train_files = {os.path.basename(f):DataUtils.allFilesAtDirWithExt(f, ".jpg") for f in self.__train_folders}
		test_files = {os.path.basename(f):DataUtils.allFilesAtDirWithExt(f, ".jpg") for f in self.__test_folders}

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

	def __getitem__(self, index):
		# Get and load the anchor image
		img_path = self.__files[self.__split][index]
		
		# Load the anchor image
		img_anchor = self.__loadResizeImage(img_path)
		
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
		# if self.__split == "test":
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
		print(f"Fold = {self.__fold+1}, split = {self.__split}, combine = {self.__combine}, known = {self.__known}")
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
		img = self.__loadResizeImage(img_path)
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
		img = self.__loadResizeImage(img_path)
		return img, random_category

	# Load an image into memory and resize it as required
	def __loadResizeImage(self, img_path):
		# # Load the image
		# img = cv2.imread(img_path)

		# # Resize it proportionally to a maximum of img_size
		# img = ImageUtils.proportionallyResizeImageToMax(img, self.__img_size[0], self.__img_size[1])

		# pos_h = int((self.__img_size[0] - img.shape[0])/2)
		# pos_w = int((self.__img_size[1] - img.shape[1])/2)

		# # Paste it into an zeroed array of img_size
		# new_img = np.zeros((self.__img_size[0], self.__img_size[1], 3), dtype=np.uint8)
		# new_img[pos_h:pos_h+img.shape[0], pos_w:pos_w+img.shape[1], :] = img

		img = Image.open(img_path)

		old_size = img.size

		ratio = float(self.__img_size[0])/max(old_size)
		new_size = tuple([int(x*ratio) for x in old_size])

		img = img.resize(new_size, Image.ANTIALIAS)

		new_img = Image.new("RGB", (self.__img_size[0], self.__img_size[1]))
		new_img.paste(img, ((self.__img_size[0]-new_size[0])//2,
					(self.__img_size[1]-new_size[1])//2))

		new_img = np.array(new_img, dtype=np.uint8)

		return new_img

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

	@staticmethod
	def extractErroneousRegions(dataset_location, visualise=False):
		# Define the errors statically (for now) (image ID: bbox)

		# Definite errors
		# errors = {	590: [1049,742,437,290],
		# 			743: [924,96,558,310],
		# 			2427: [23,422,132,132],
		# 			2538: [415,425,102,62],
		# 			2581: [432,168,98,51],
		# 			2584: [53,169,107,71],
		# 			2607: [0,219,84,41],
		# 			2623: [641,419,88,48],
		# 			2719: [471,0,184,57],
		# 			2730: [604,262,128,74],
		# 			2967: [388,262,119,202],
		# 			3169: [347,350,56,153],
		# 			3200: [284,199,82,146],
		# 			3203: [556,37,65,144],
		# 			3312: [440,4,293,235]		}

		# Possible errors
		errors = {	162: [893,38,575,279],
					1140: [516,167,581,455],
					1538: [245,0,461,333],
					1803: [69,547,534,166],
					2654: [175,347,130,48],
					2654: [161,399,118,57],
					3122: [56,165,95,96],
					3200: [284,199,82,146],
					3200: [352,206,60,135],
					3530: [95,363,172,360]		}

		# Iterate through each key
		for image_id, box in errors.items():
			# Load the image this entry refers to
			image_path = os.path.join(dataset_location, str(image_id).zfill(6)+".jpg")
			image = cv2.imread(image_path)

			# Extract the RoI
			# x1 = box[0]
			# y1 = box[1]
			# x2 = box[2]
			# y2 = box[3]
			x1 = int(box[0] - box[2]/2)
			y1 = int(box[1] - box[3]/2)
			x2 = x1 + int(box[2])
			y2 = y1 + int(box[3])

			# Clamp any negative values at zero
			if x1 < 0: x1 = 0
			if y1 < 0: y1 = 0
			if x2 < 0: x2 = 0
			if y2 < 0: y2 = 0

			# Swap components if necessary so that (x1,y1) is always top left
			# and (x2, y2) is always bottom right
			if x1 > x2: x1, x2 = x2, x1
			if y1 > y2: y1, y2 = y2, y1

			RoI = image[y1:y2,x1:x2]

			# Display it if we're supposed to
			if visualise:
				# cv2.rectangle(image, (x1,y1),(x2,y2), (255,0,0), 3)
				# cv2.imshow("Extracted region", RoI)
				# cv2.waitKey(0)
				pass

			# Construct path and save
			save_path = os.path.join(output_dir, str(image_id).zfill(6)+".jpg")
			cv2.imwrite(save_path, RoI)

# Entry method/unit testing method
if __name__ == '__main__':
	# Create a dataset instance
	# dataset = OpenSetCows2019(0, "2-folds.pkl", split="test")

	# for i in range(dataset.getNumTestingFiles()):
	# 	test = dataset[i]

	# From a list of erroneous detections (e.g. from RetinaNet)
	dataset_location = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\CEADetection\\darknet"
	output_dir = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\CEADetection\\RetinaNet-failures\\possible"
	OpenSetCows2019.extractErroneousRegions(dataset_location, output_dir)
	