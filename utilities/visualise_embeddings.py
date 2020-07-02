import os
import sys
import cv2
import pickle
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.patheffects as PathEffects
from tqdm import tqdm

# Home windows machine
if os.path.isdir("D:\\Work"): sys.path.append("D:\\Work\\ATI-Pilot-Project\\src")

# My libraries
from Utilities.DataUtils import DataUtils
from Utilities.ImageUtils import ImageUtils

# Data paths
# all_embedding_train_path = "../train_triplet_cnn_cow_id_temp_x1_full.npz"
# all_embedding_test_path = "../test_triplet_cnn_cow_id_temp_x1_full.npz"
# known_embedding_train_path = "../train_triplet_cnn_cow_id_temp_x1_known.npz"
# known_embedding_test_path = "../test_triplet_cnn_cow_id_temp_x1_known.npz"
# novel_embedding_train_path = "../train_triplet_cnn_cow_id_temp_x1_novel.npz"
# novel_embedding_test_path = "../test_triplet_cnn_cow_id_temp_x1_novel.npz"

#base_dir = "/home/ca0513/CEiA/results/SoftmaxRTL/50-50/fold_0"
base_dir = "D:\\Work\\results\\SoftmaxRTL\\50-50\\fold_0"
# base_dir = "/home/will/work/CEiA/results/STL/90-10/fold_0"
# base_dir = "/home/will/work/CEiA/results/STL/50-50/fold_0"
# base_dir = "/home/will/work/1-RA/src/Identifier/Supervised-Triplet-Network/results/fold_0"

all_embedding_train_path = os.path.join(base_dir, "train_triplet_cnn_open_cows_temp_x1_full.npz")
all_embedding_test_path = os.path.join(base_dir, "test_triplet_cnn_open_cows_temp_x1_full.npz")
known_embedding_train_path = os.path.join(base_dir, "train_triplet_cnn_open_cows_temp_x1_known.npz")
known_embedding_test_path = os.path.join(base_dir, "test_triplet_cnn_open_cows_temp_x1_known.npz")
novel_embedding_train_path = os.path.join(base_dir, "train_triplet_cnn_open_cows_temp_x1_novel.npz")
novel_embedding_test_path = os.path.join(base_dir, "test_triplet_cnn_open_cows_temp_x1_novel.npz")

# Number of classes
num_classes = 46

# Define our own plot function
def scatter(x, labels, subtitle=None, overlay_class_examples=False, class_examples=None, enable_labels=True):
	# Load the dictionary of folds (which classes are unknown)
	curr_fold = 0
	# folds_fp = "/home/will/work/1-RA/src/Datasets/data/OpenSetCows2019/2-folds.pkl"
	folds_fp = "D:\\Work\\ATI-Pilot-Project\\src\\Datasets\\data\\OpenSetCows2019\\2-folds.pkl"

	# We choose a color palette with seaborn.
	palette = np.array(sns.color_palette("hls", num_classes+1))

	# Make a marker array based on which are the known and unknown classes currently
	known = np.ones(labels.shape)

	if os.path.exists(folds_fp):
		with open(folds_fp, 'rb') as handle:
			folds_dict = pickle.load(handle)

		# Mark which classes are novel/unseen
		for i in range(labels.shape[0]):
			if str(int(labels[i])).zfill(3) in folds_dict[curr_fold]['unknown']: 
				known[i] = 0
	else: folds_dict = {}

	f = plt.figure(figsize=(8, 8))
	ax = plt.subplot(aspect='equal')
	# KNOWN
	sc0 = ax.scatter(	x[known==1,0], x[known==1,1], 
						lw=0, s=40, 
						c=palette[labels[known==1].astype(np.int)], 
						marker="o",
						label="known"	)
	# UNKNOWN
	sc1 = ax.scatter(	x[known==0,0], x[known==0,1],
						lw=0, s=40,
						c=palette[labels[known==0].astype(np.int)],
						marker="^",
						label="unknown"	)
	# plt.legend(loc="lower right")
	plt.xlim(-25, 25)
	plt.ylim(-25, 25)
	ax.axis('off')
	ax.axis('tight')

	# We add the labels for each digit.
	if enable_labels:
		for i in range(1,num_classes):
			if os.path.exists(folds_fp):
				# Colour the text based on whether its a known or unknown class
				if str(i).zfill(3) in folds_dict[curr_fold]['unknown']: text_color = "red"
				else: text_color = "black"
			else: text_color = "black"

			# Position of each label.
			xtext, ytext = np.median(x[labels == i, :], axis=0)
			txt = ax.text(xtext, ytext, str(i), fontsize=24, color=text_color)
			txt.set_path_effects([
				PathEffects.Stroke(linewidth=5, foreground="w"),
				PathEffects.Normal()])
		
	# plt.show()
	plt.tight_layout()
	plt.savefig(subtitle)

def plotEmbeddings():
	# Load them into memory
	all_embedding_train = np.load(all_embedding_train_path)
	all_embedding_test = np.load(all_embedding_test_path)
	# known_embedding_train = np.load(known_embedding_train_path)
	# known_embedding_test = np.load(known_embedding_test_path)
	# novel_embedding_train = np.load(novel_embedding_train_path)
	# novel_embedding_test = np.load(novel_embedding_test_path)

	print("Loaded embeddings")

	perplexity = 25

	# Visualise the learned embedding via TSNE
<<<<<<< HEAD
	visualiser = TSNE(n_components=2, perplexity=30)
=======
	visualiser = TSNE(n_components=2, perplexity=perplexity)
>>>>>>> 00944211ddf9d1418846ee8ddba18bbcd02912b7
	# visualiser = PCA(n_components=2)

	# Perform TSNE magic
	all_tsne_train = visualiser.fit_transform(all_embedding_train['embeddings'])
	all_tsne_test = visualiser.fit_transform(all_embedding_test['embeddings'])
	# known_tsne_train = visualiser.fit_transform(known_embedding_train['embeddings'])
	# known_tsne_test = visualiser.fit_transform(known_embedding_test['embeddings'])
	# novel_tsne_train = visualiser.fit_transform(novel_embedding_train['embeddings'])
	# novel_tsne_test = visualiser.fit_transform(novel_embedding_test['embeddings'])

	print("Visualisation computed")

	# Plot the results and save to file
	scatter(all_tsne_train, all_embedding_train['labels'], f"all-training-perplexity-{perplexity}.pdf")
	scatter(all_tsne_test, all_embedding_test['labels'], f"all-testing-perplexity-{perplexity}.pdf")
	# scatter(known_tsne_train, known_embedding_train['labels'], "Known categories - TRAINING")
	# scatter(known_tsne_test, known_embedding_test['labels'], "Known categories - TESTING")
	# scatter(novel_tsne_train, novel_embedding_train['labels'], "Novel categories - TRAINING")
	# scatter(novel_tsne_test, novel_embedding_test['labels'], "Novel categories - TESTING")

# Plots the embeddings for a particular openness / fold for all different loss functions to evaluate the contrast in
# clusterings
def plotEmbeddingsComparison():
	# Parameters
	openness = "50-50"		# How open the problem should be
	fold = 1				# Which fold to render
	train = 1				# Whether to render the train or test set embeddings
	use_set = 0				# Which set to use (full, known, novel)
	display_class_labels = False # Overlay class labels on the embedding centroids

	# The list of sets
	data_sets = ["full", "known", "novel"]

	# Filenames
	train_file = f"train_triplet_cnn_open_cows_temp_x1_{data_sets[use_set]}.npz"
	test_file = f"test_triplet_cnn_open_cows_temp_x1_{data_sets[use_set]}.npz"

	# On home windows machine
	base_dir = "D:\\Work\\results"

	# Generate the embeddings directories
	STL_dir = os.path.join(base_dir, "TL", openness, f"fold_{fold}")
	RTL_dir = os.path.join(base_dir, "RTL", openness, f"fold_{fold}")
	SoftmaxTL_dir = os.path.join(base_dir, "SoftmaxTL", openness, f"fold_{fold}")
	SoftmaxRTL_dir = os.path.join(base_dir, "SoftmaxRTL", openness, f"fold_{fold}")

	print("Loading the embeddings")

	# Load the embeddings
	STL = {0: np.load(os.path.join(STL_dir, test_file)), 1: np.load(os.path.join(STL_dir, train_file))}
	RTL = {0: np.load(os.path.join(RTL_dir, test_file)), 1: np.load(os.path.join(RTL_dir, train_file))}
	SoftmaxTL = {0: np.load(os.path.join(SoftmaxTL_dir, test_file)), 1: np.load(os.path.join(SoftmaxTL_dir, train_file))}
	SoftmaxRTL = {0: np.load(os.path.join(SoftmaxRTL_dir, test_file)), 1: np.load(os.path.join(SoftmaxRTL_dir, train_file))}

	# Visualise using TSNE
	visualiser = TSNE(n_components=2)

	print("Performing TSNE")

	# Perform TSNE magic
	pbar = tqdm(total=4)
	STL_TSNE = visualiser.fit_transform(STL[train]['embeddings']); pbar.update()
	RTL_TSNE = visualiser.fit_transform(RTL[train]['embeddings']); pbar.update()
	SoftmaxTL_TSNE = visualiser.fit_transform(SoftmaxTL[train]['embeddings']); pbar.update()
	SoftmaxRTL_TSNE = visualiser.fit_transform(SoftmaxRTL[train]['embeddings']); pbar.update()
	pbar.close()

	print("Rendering the embeddings")

	# Actually render the embeddings
	pbar = tqdm(total=4)
	scatter(STL_TSNE, STL[train]['labels'], "TripletLoss.pdf", enable_labels=display_class_labels); pbar.update()
	scatter(RTL_TSNE, RTL[train]['labels'], "ReciprocalTripletLoss.pdf", enable_labels=display_class_labels); pbar.update()
	scatter(SoftmaxTL_TSNE, SoftmaxTL[train]['labels'], "SoftmaxTripletLoss.pdf", enable_labels=display_class_labels); pbar.update()
	scatter(SoftmaxRTL_TSNE, SoftmaxRTL[train]['labels'], "SoftmaxReciprocalTripletLoss.pdf", enable_labels=display_class_labels); pbar.update()
	pbar.close()

# Plot the embedding for some run, where an example for each class is overlaid at the centroid of its embeddings
def plotClassOverlay():
	# Parameters
	openness = "50-50"		# How open the problem should be
	fold = 1				# Which fold to render
	train = 1				# Whether to render the train or test set embeddings
	use_set = 0				# Which set to use (full, known, novel)

	# The list of sets
	data_sets = ["full", "known", "novel"]

	# Filenames
	train_file = f"train_triplet_cnn_open_cows_temp_x1_{data_sets[use_set]}.npz"
	test_file = f"test_triplet_cnn_open_cows_temp_x1_{data_sets[use_set]}.npz"

	# On home windows machine
	base_dir = "D:\\Work\\results"

	# Where to find the splits for the dataset
	splits_dir = "D:\\Work\\ATI-Pilot-Project\\src\\Datasets\\data\\OpenSetCows2019\\2-folds.pkl"
	with open(splits_dir, 'rb') as handle:
		splits_dict = pickle.load(handle)
	splits = splits_dict[fold]

	# Generate the embeddings directories
	embed_dir = os.path.join(base_dir, "SoftmaxRTL", openness, f"fold_{fold}")

	# Load them
	embeddings = {0: np.load(os.path.join(embed_dir, test_file)), 1: np.load(os.path.join(embed_dir, train_file))}

	# Visualise using TSNE
	visualiser = TSNE(n_components=2)
	reduction = visualiser.fit_transform(embeddings[train]['embeddings'])

	# Directory to find dataset
	dataset_dir = "D:\\Work\\Data\\OpenCows2020"

	# Load an example for each class
	class_filepaths = DataUtils.readFolderDatasetFilepathList(dataset_dir)

	# Produce the plot
	f = plt.figure(figsize=(8, 8))
	ax = plt.subplot(aspect='equal')

	xs = []
	ys = []
	margin = 2
	zoom = 0.25

	# Plot the images
	for k, filepaths in class_filepaths.items():
		# Get the labels
		labels = embeddings[train]['labels']

		# Compute the centroid
		x, y = np.median(reduction[labels==int(k), :], axis=0)

		xs.append(x)
		ys.append(y)

		# Load a random image for this class
		image = cv2.imread(random.choice(filepaths))
		image = ImageUtils.proportionallyResizeImageToMax(image, 200, 200)
		
		# Plot the example at the centroid
		imagebox = OffsetImage(image, zoom=zoom)

		if k in splits['unknown']: ab = AnnotationBbox(imagebox, (x, y), bboxprops=dict(color='red'))
		else: ab = AnnotationBbox(imagebox, (x, y))

		ax.add_artist(ab)

	ax.axis('off')
	ax.axis('tight')
	plt.xlim(min(xs)-margin, max(xs)+margin)
	plt.ylim(min(ys)-margin, max(ys)+margin)

	# plt.show()
	# plt.tight_layout()
	plt.savefig("class-overlay.pdf")

# Entry method/unit testing method
if __name__ == '__main__':
	plotEmbeddings()
	# plotEmbeddingsComparison()
	# plotClassOverlay()
