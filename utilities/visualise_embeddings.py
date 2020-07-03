# Core libraries
import os
import sys
import cv2
import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm

# Matplotlib / TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

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
	plt.savefig(subtitle+".pdf")

def plotEmbeddings():
	# Load them into memory
	all_embedding_train = np.load(all_embedding_train_path)
	all_embedding_test = np.load(all_embedding_test_path)
	known_embedding_train = np.load(known_embedding_train_path)
	known_embedding_test = np.load(known_embedding_test_path)
	novel_embedding_train = np.load(novel_embedding_train_path)
	novel_embedding_test = np.load(novel_embedding_test_path)

	print("Loaded embeddings")

	# Visualise the learned embedding via TSNE
	visualiser = TSNE(n_components=2)

	# Perform TSNE magic
	all_tsne_train = visualiser.fit_transform(all_embedding_train['embeddings'])
	all_tsne_test = visualiser.fit_transform(all_embedding_test['embeddings'])
	known_tsne_train = visualiser.fit_transform(known_embedding_train['embeddings'])
	known_tsne_test = visualiser.fit_transform(known_embedding_test['embeddings'])
	novel_tsne_train = visualiser.fit_transform(novel_embedding_train['embeddings'])
	novel_tsne_test = visualiser.fit_transform(novel_embedding_test['embeddings'])

	print("Visualisation computed")

	# Plot the results and save to file
	scatter(all_tsne_train, all_embedding_train['labels'], f"All categories - TRAINING")
	scatter(all_tsne_test, all_embedding_test['labels'], f"ALL categories - TESTING")
	scatter(known_tsne_train, known_embedding_train['labels'], "Known categories - TRAINING")
	scatter(known_tsne_test, known_embedding_test['labels'], "Known categories - TESTING")
	scatter(novel_tsne_train, novel_embedding_train['labels'], "Novel categories - TRAINING")
	scatter(novel_tsne_test, novel_embedding_test['labels'], "Novel categories - TESTING")

# Main/entry method
if __name__ == '__main__':
	# Collate command line arguments
	parser = argparse.ArgumentParser(description='Parameters for visualising the embeddings via TSNE')

	args = parser.parse_args()

	# Let's plot!
	plotEmbeddings()