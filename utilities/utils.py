# Core libraries
import os
import sys
import argparse
import subprocess
import numpy as np
from PIL import Image

# PyTorch stuff
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.append('.')
sys.path.append('..')

# Global variables to keep track of training progress
losses_softmax = []
losses_triplet = []
losses_rec = []
losses_mean = []
accuracies_all = []
accuracies_known = []
accuracies_novel = []

losses = [losses_softmax, losses_triplet, losses_rec, losses_mean]
losses_id = ["loss_softmax", "loss_triplet", "loss_rec", "loss_mean"]

accuracies = [accuracies_all, accuracies_known, accuracies_novel]
accuracies_id = ["accuracies_all", "accuracies_known", "accuracies_novel"]

steps = []
accuracy_step = []

ID = ""

args_local = None
ckpt_full_path =  ""

"""
File contains a collection of utility functions used for training
and evaluation
"""

# Save a checkpoint as the current state of training
def save_checkpoint(epoch, model, optimizer, description):
    global ckpt_full_path

    state = {'epoch': epoch+1,
             'model_state': model.state_dict(),
             'optimizer_state' : optimizer.state_dict(),}

    ckpt_full_path = "{}/{}_{}_{}_{}.pkl".format( args_local.ckpt_path, args_local.arch, args_local.dataset, description, args_local.id)
    torch.save(state, "{}/{}_{}_{}_{}.pkl".format( args_local.ckpt_path, args_local.arch, args_local.dataset, description, args_local.id))

    # print(f"Saved model state to: {ckpt_full_path}")

# Print information about the current training setup
def show_setup(args, n_classes, optimizer, loss_fn):
    global args_local, ID
    args_local=args

    print("Model: {} | Training on: {} | Number of Classes: {}".format(args.arch, args.dataset, n_classes))
    print("Embedding size: {}".format(args.embedding_size))
    print("Epochs: {}".format(args.n_epoch))
    print("Optimizer: {}".format(optimizer))
    print("Loss function: {}".format(loss_fn))

    ID = "Model: {} \n Training on: {} \n Number of Classes: {} \n Epochs: {} \n Optimizer: {} \n Loss function: {} ".format(args.arch, 
        args.dataset, n_classes,args.n_epoch, optimizer, loss_fn)

# Log information 
def log_loss(epoch, total_epochs, step, loss_softmax=None, loss_triplet=None, loss_rec=None, loss_mean=None):
    global steps, losses_softmax, losses_triplet, losses_rec, losses_mean, losses, args_local, ID

    if loss_softmax != None: losses_softmax.append(loss_softmax)
    if loss_triplet != None: losses_triplet.append(loss_triplet)
    if loss_rec != None: losses_rec.append(loss_rec)
    if loss_mean != None: losses_mean.append(loss_mean)

    steps.append(step)

    log_message = "" 

    for idx in range(len(losses)):

        if len(losses[idx]) > 0:
    
            log_message += losses_id[idx] + ": %.4f | " % losses[idx][-1] 

    print("Epoch [{}/{}] | Global step: {} | {} ".format(epoch+1, total_epochs, step, log_message))

    if step % args_local.logs_freq == 0:
        np.savez("{}/{}_{}_train_log_{}".format(args_local.ckpt_path, args_local.dataset, args_local.arch, args_local.id),  steps=steps, losses_softmax=losses_softmax, 
            losses_triplet=losses_triplet, losses_rec=losses_rec, losses_mean=losses_mean, ID=ID)

# Evaluate the current model state
def eval_model(fold, fold_file, step, sets):
    global ckpt_full_path
    global args_local
    global accuracies_all, accuracies_known, accuracies_novel, accuracies, ID, accuracy_step

    sets = sets

    if sets == "all": sets_to_test = ["full", "known", "novel"]
    elif sets == "known": sets_to_test = ["known"]
    elif sets == "novel": sets_to_test = ["novel"]
    elif sets == "full": sets_to_test = ["full"]

    for set_to_test in sets_to_test:
        test_run_str = "python3 test/test_embeddings.py --ckpt_path={} --dataset={} \
                        --instances={} --split=test --embedding_size={} --current_fold={} --fold_file={}"\
                        .format(ckpt_full_path,  args_local.dataset, set_to_test, \
                            args_local.embedding_size, fold, fold_file)
        train_run_str = "python3 test/test_embeddings.py --ckpt_path={} --dataset={} \
                        --instances={} --split=train --embedding_size={} --current_fold={} --fold_file={}"\
                        .format(ckpt_full_path,  args_local.dataset, set_to_test, \
                            args_local.embedding_size, fold, fold_file)

        # print(test_run_str)
        # print(train_run_str)

        subprocess.call([test_run_str], shell=True)
        subprocess.call([train_run_str], shell=True)
        
        #print (ckpt_full_path)
        if set_to_test == "full":
            accuracy_all = subprocess.check_output(["python3 test/nearest_neighbours.py --ckpt_path={} --instances={}".format(ckpt_full_path, set_to_test)], shell=True)
            accuracy_all = float(accuracy_all.decode('utf-8'))
            accuracies_all.append(accuracy_all)
            print ("Accuracy full: {} | step: {} ".format(accuracy_all, step))

        elif set_to_test == "known":
            accuracy_known = subprocess.check_output(["python3 test/nearest_neighbours.py --ckpt_path={} --instances={}".format(ckpt_full_path, set_to_test)], shell=True)
            accuracy_known = float(accuracy_known.decode('utf-8'))
            accuracies_known.append(accuracy_known)

            print ("Accuracy known: {} | step: {} ".format(accuracy_known, step)) 
             
        elif set_to_test == "novel":
            accuracy_novel = subprocess.check_output(["python3 test/nearest_neighbours.py --ckpt_path={} --instances={}".format(ckpt_full_path, set_to_test)], shell=True)
            accuracy_novel = float(accuracy_novel.decode('utf-8'))
            accuracies_novel.append(accuracy_novel) 
            print ("Accuracy novel: {} | step: {} ".format(accuracy_novel, step))


    accuracy_step.append(step)
    #print ("OK")

    np.savez("{}/{}_{}_accuracies_log_{}".format(args_local.ckpt_path, args_local.dataset, args_local.arch, args_local.id),  steps=accuracy_step, accuracies_all=accuracies_all, 
            accuracies_known=accuracies_known, accuracies_novel=accuracies_novel, ID=ID)

    return accuracy_all

# Create a sorted list of all files with a given extension at a given directory
# If full_path is true, it will return the complete path to that file
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