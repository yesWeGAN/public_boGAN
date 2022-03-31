import os
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans


def cluster_images_in_dir(pickledir_path, pandas_frame, lr_sensitive=False, k=15):
    """Task:  cluster pose in images (image-filenames stored in pandasframe to work seamlessly with ShopDataset class)
    function: clusters all tensors in a directory into k-classes. vectorized using numpy.

    inputs: k (number of desired clusters), path to pickled DensePose vectors, path to pandas Dataframe
    outputs: tensor with class per index
    flags:
    lr_sensitive: defines sensitivity to left/right facing poses. if true, increase k by 40%! """

    clust = pd.read_csv(pandas_frame, ignore_index=True)
    cluster_tensor = np.zeros((25, len(clust)))

    for index in range(len(clust)):
        filename = clust.at[index, "img_path"]
        with open(os.path.join(pickledir_path, (filename.split("/")[-1].replace(".jpg", ".pkl"))), 'rb') as pklfile:
            tensor = pickle.load(pklfile)
            cluster_tensor[:, index] = tensor

    if not lr_sensitive:
        # map left/right sensitive labels on one another. yields 15 labels. see docstring for DensePose labels
        lr_reduced = np.zeros((len(cluster_tensor), 15))
        lr_reduced[:, 0] = cluster_tensor.T[:, 1]
        lr_reduced[:, 1] = cluster_tensor.T[:, 2]
        lr_reduced[:, 2] = cluster_tensor.T[:, 3]
        lr_reduced[:, 3] = cluster_tensor.T[:, 4]
        lr_reduced[:, 4] = cluster_tensor.T[:, 5]
        lr_reduced[:, 5] = cluster_tensor.T[:, 6]
        lr_reduced[:, 6] = cluster_tensor.T[:, 7] + cluster_tensor.T[:, 8]
        lr_reduced[:, 7] = cluster_tensor.T[:, 9] + cluster_tensor.T[:, 10]
        lr_reduced[:, 8] = cluster_tensor.T[:, 11] + cluster_tensor.T[:, 12]
        lr_reduced[:, 9] = cluster_tensor.T[:, 13] + cluster_tensor.T[:, 14]
        lr_reduced[:, 10] = cluster_tensor.T[:, 15] + cluster_tensor.T[:, 16]
        lr_reduced[:, 11] = cluster_tensor.T[:, 17] + cluster_tensor.T[:, 18]
        lr_reduced[:, 12] = cluster_tensor.T[:, 19] + cluster_tensor.T[:, 20]
        lr_reduced[:, 13] = cluster_tensor.T[:, 21] + cluster_tensor.T[:, 22]
        lr_reduced[:, 14] = cluster_tensor.T[:, 23] + cluster_tensor.T[:, 24]

    kmeans = KMeans(n_clusters=k, random_state=0).fit(lr_reduced)
    return kmeans


def print_area_labels(self):
    # call the docstring to see the area labels
    """
    0      = Background
    1, 2   = Torso    1 BACKSIDE 2 FRONTSIDE
    3      = Right Hand MOSTLY USELESS AS ITS FRONT/BACK INSENSITIVE
    4      = Left Hand MOSTLY USELESS AS ITS FRONT/BACK INSENSITIVE
    5      = Right Foot MOSTLY USELESS AS ITS FRONT/BACK INSENSITIVE
    6      = Left Foot MOSTLY USELESS AS ITS FRONT/BACK INSENSITIVE
    7, 9   = Upper Leg Right   7 BACKSIDE  9 FRONTSIDE
    8, 10  = Upper Leg Left    8 BACKSIDE  10 FRONTSIDE
    11, 13 = Lower Leg Right   11 BACKSIDE 13 FRONTSIDE
    12, 14 = Lower Leg Left    12 BACKSIDE 14 FRONTSIDE
    15, 17 = Upper Arm Left
    16, 18 = Upper Arm Right
    19, 21 = Lower Arm Left
    20, 22 = Lower Arm Right
    23, 24 = Head   23 RIGHT SIDE  24 LEFT SIDE ATTENTION ITS INSENSITIVE TO HAIR COVERING
    """
pass