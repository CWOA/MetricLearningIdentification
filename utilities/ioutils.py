# Core libraries
import os
import numpy as np
from PIL import Image

"""
File contains input/output utility functions
"""

# Create a sorted list of all files with a given extension at a given directory
# If full_path is true, it will return the complete path to that file
def allFilesAtDirWithExt(directory, file_extension, full_path=True):
    # Make sure we're looking at a folder
    if not os.path.isdir(directory): print(directory)
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
    new_img = Image.new("RGB", (size[0], size[1]))
    new_img.paste(img, ((size[0]-new_size[0])//2, (size[1]-new_size[1])//2))

    # Convert to numpy
    new_img = np.array(new_img, dtype=np.uint8)

    return new_img
