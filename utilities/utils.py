# Core libraries
import os
import sys
import argparse
import subprocess
import numpy as np
from PIL import Image

# PyTorch stuff
import torch
from torch import optim
from torch.utils import data

# Import our dataset class
from datasets.OpenSetCows2020.OpenSetCows2020 import OpenSetCows2020

"""
File contains a collection of utility functions used for training and evaluation
"""

class Utilities:
    # Class constructor
    def __init__(self, args):
        # Store the arguments
        self.args = args

        # Where to store training logs
        self.log_path = os.path.join(args.fold_out_path, "logs.npz")

        # Prepare arrays to store training information
        self.loss_steps = []
        self.losses_mean = []
        self.losses_softmax = []
        self.losses_triplet = []
        self.accuracy_steps = []
        self.accuracies = []

    # Preparations for training for a particular fold
    def setupForTraining(self, args):
        # Retrieve the correct dataset
        dataset = Utilities.selectDataset(args)

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

        return data_loader, model, loss_fn, optimiser

    # Save a checkpoint as the current state of training
    def saveCheckpoint(self, epoch, model, optimiser, description):
        # Construct a state dictionary for the training's current state
        state = {   'epoch': epoch+1,
                    'model_state': model.state_dict(),
                    'optimizer_state' : optimiser.state_dict()  }

        # Construct the full path for where to save this
        self.checkpoint_path = os.path.join(self.args.fold_out_path, f"{description}_model_state.pkl")

        # And save actually it
        torch.save(state, self.checkpoint_path)

    # Save training logs to file
    def saveLogs(self):
        # Save this data to file for plotting graphs, etc.
        np.savez(   self.log_path, 
                    loss_steps=self.loss_steps,
                    losses_mean=self.losses_mean,
                    losses_softmax=self.losses_softmax,
                    losses_triplet=self.losses_triplet,
                    accuracy_steps=self.accuracy_steps,
                    accuracies=self.accuracies    )

    # Log information 
    def logTrainInfo(self, epoch, global_step, loss_mean, loss_triplet=None, loss_softmax=None):
        # Add to our arrays
        self.loss_steps.append(step)
        self.losses_mean.append(loss_mean)
        if loss_triplet != None: self.losses_triplet.append(loss_triplet)
        if loss_softmax != None: self.losses_softmax.append(loss_softmax)
        
        # Construct a message and print it to the console
        log_message = f"Epoch [{epoch+1}/{self.args.num_epochs}] Global step: {global_step} | loss_mean: {loss_mean}"
        if loss_triplet != None: log_message += ", loss_triplet: {losses_triplet}"
        if loss_softmax != None: log_message += ", loss_softmax: {losses_softmax}"
        print(log_message)

        # Save this new data to file
        self.saveLogs()

    # Evaluate the current model state, calls test.py in a subprocess and saves the results to file
    def test(self, step):
        # Construct subprocess call string
        run_str  = f"python test.py"
        run_str += f" --model_path={self.checkpoint_path}"  # Saved model weights to use
        run_str += f" --dataset={self.args.dataset}"        # Which dataset to use
        run_str += f" --batch_size={self.args.batch_size}"  # Batch size to use when inferring
        run_str += f" --embedding_size={self.args.embedding_size}"  # Embedding dimensionality
        run_str += f" --current_fold={self.args.current_fold}"  # The current fold number
        run_str += f" --folds_file={self.args.folds_file}"  # Where to find info about this fold
        run_str += f" --save_path={self.args.fold_out_path}"    # Where to store the embeddings

        # Let's run the command, decode and save the result
        accuracy = subprocess.check_output([run_str], shell=True)
        accuracy = float(accuracy.decode('utf-8'))
        self.accuracies.append(accuracy)

        # Save this accuracies to file
        self.saveLogs()

        return accuracy

    """
    Static methods
    """

    # Return the selected dataset based on text choice
    @staticmethod
    def selectDataset(args, train):
        # Which split are we after?
        if train: split = "train"
        else: split = "test"

         # Load the selected dataset
        if args.dataset == "OpenSetCows2020":
            dataset = OpenSetCows2020(  args.current_fold, 
                                        args.folds_file, 
                                        split=split, 
                                        transform=True, 
                                        suppress_info=False )
        elif args.dataset == "ADD YOUR DATASET HERE":
            pass
        else:
            print(f"Dataset choice not recognised, exiting.")
            sys.exit(1)

    # Create a sorted list of all files with a given extension at a given directory
    # If full_path is true, it will return the complete path to that file
    @staticmethod
    def allFilesAtDirWithExt(directory, file_extension, full_path=True):
        # Make sure we're looking at a folder
        assert os.path.isdir(directory)

        # Gather the files inside
        if full_path:
            files = [os.path.join(directory, x) for x in sorted(os.listdir(directory)) if x.endswith(file_extension)]
        else:
            files = [x for x in sorted(os.listdir(directory)) if x.endswith(file_extension)]

        return files

    # Similarly, create a sorted list of all folders at a given directory
    @staticmethod
    def allFoldersAtDir(directory, full_path=True):
        # Make sure we're looking at a folder
        if not os.path.isdir(directory): print(directory)
        assert os.path.isdir(directory)

        # Find all the folders
        if full_path:
            folders = [os.path.join(directory, x) for x in sorted(os.listdir(directory)) if os.path.isdir(os.path.join(directory, x))]
        else:
            folders = [x for x in sorted(os.listdir(directory)) if os.path.isdir(os.path.join(directory, x))]

        return folders

    # Load an image into memory, pad it to img size with a black background
    @staticmethod
    def loadResizeImage(img_path, size):      
        # Load the image
        img = Image.open(img_path)

        # Keep the original image size
        old_size = img.size

        # Compute resizing ratio
        ratio = float(size[0])/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        # Actually resize it
        img = img.resize(new_size, Image.ANTIALIAS)

        # Paste into centre of black padded image
        new_img = Image.new("RGB", (img_size[0], size[1]))
        new_img.paste(img, ((size[0]-new_size[0])//2, (size[1]-new_size[1])//2))

        # Convert to numpy
        new_img = np.array(new_img, dtype=np.uint8)

        return new_img
