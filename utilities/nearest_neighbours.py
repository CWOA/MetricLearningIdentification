# Core libraries
import os
import sys
import argparse
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def KNN_accuracy(args, n_neighbors=5):
    train_file = os.path.split(args.ckpt_path)[0]+"/train_" + os.path.split(args.ckpt_path)[1][0:-4] + "_" + args.instances + ".npz"
    test_file  = os.path.split(args.ckpt_path)[0]+"/test_" + os.path.split(args.ckpt_path)[1][0:-4] + "_" + args.instances + ".npz"

    npzfile_train = np.load(train_file)
    npzfile_test = np.load(test_file)

    X_train = npzfile_train['embeddings'][1:]
    y_train = npzfile_train['labels'][1:]
    filenames_train = npzfile_train['filenames'][1:]

    X_test = npzfile_test['embeddings'][1:]
    y_test = npzfile_test['labels'][1:]
    filenames_test = npzfile_test['filenames'][1:]

    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-4)
    neigh.fit(X_train, y_train)

    total = len(y_test-1)
    correct = 0

    # Get the predictions from KNN
    predictions = neigh.predict(X_test)

    # How many were correct?
    correct += (predictions == y_test).sum()

    # Which were incorrect?
    incorrect = (predictions != y_test)

    print(incorrect.sum())
    print(y_test[incorrect == True])

    #print (total)
    #print("Precision KNN : ", 100.*correct/total )
    sys.stdout.write("{}".format(100.*correct/total))
    sys.stdout.flush()
    sys.exit(0)
    #return 100.0 * correct / total
    
    #sys.exit(0)

# Entry method
if __name__ == '__main__':
    # Collate and parse arguments
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--ckpt_path', nargs='?', type=str,
                        help='Path of the input image')
    parser.add_argument('--instances', nargs='?', type=str, default="full",
                        help='Path of the input image')
    args = parser.parse_args()

    # Let's compute the accuracy via KNN
    KNN_accuracy(args)