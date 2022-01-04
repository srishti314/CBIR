# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 19:12:38 2018

@author: Srishti
"""

%pylab inline --no-import-all
# OpenCV bindings
import cv2
# To performing path manipulations 
import os
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram 
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
# Utility package -- use pip install cvutils to install
from matplotlib import pyplot as plt
import cvutils
# To read class from file
import csv
import pandas as pd
import os

from numpy.random import RandomState

from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
from skimage import segmentation, color, measure
from skimage.future import graph
from skimage.exposure import rescale_intensity

from skimage.morphology import remove_small_objects, label

from skimage.feature import daisy
from skimage.color import rgb2gray
from datasets import *

path="C:\\Users\\Saksham\\Desktop\\Research AI 2\\Python codes"
os.chdir(path)


train_images = cvutils.imlist("C:/Users/Saksham/Desktop/research-AI 1/python codes/images-corel1K/")

nmb_seg=100
all_features = []


for train_image in train_images:

    image = cv2.imread(train_image)
    #image = cv2.resize(image, (384, 255))
    
    # by default slic computes ~100 superpixels.
    # compactness controls the shape of the superpixels.
    segmentation = slic(image, n_segments=nmb_seg, compactness=20, sigma=0)
    
    fig, axes = plt.subplots(3, 1, figsize=(20, 10))
    axes[0].imshow(image)
    axes[1].imshow(segmentation)
    axes[2].imshow(mark_boundaries(image, segmentation))

    #def merge_small_sp(image, regions, min_size=None):
    #    if min_size is None:
    #        min_size = np.prod(image.shape[:2]) / float(np.max(regions) + 1)
    #    shape = regions.shape
    #    _, regions = np.unique(regions, return_inverse=True)
    #    regions = regions.reshape(shape[:2])
    #    edges = region_graph(regions)
    #    mean_colors = get_mean_colors(image, regions)
    #    mask = np.bincount(regions.ravel()) < min_size
    #    # mapping of old labels to new labels
    #    new_labels = np.arange(len(np.unique(regions)))
    #    for r in np.where(mask)[0]:
    #        # get neighbors:
    #        where_0 = edges[:, 0] == r
    #        where_1 = edges[:, 1] == r
    #        neighbors1 = edges[where_0, 1]
    #        neighbors2 = edges[where_1, 0]
    #        neighbors = np.concatenate([neighbors1, neighbors2])
    #        neighbors = neighbors[neighbors != r]
    #        # get closest in color
    #        distances = np.sum((mean_colors[r] - mean_colors[neighbors]) ** 2,
    #                           axis=-1)
    #        nearest = np.argmin(distances)
    #        # merge
    #        new = neighbors[nearest]
    #        new_labels[new_labels == r] = new
    #        edges[where_0, 0] = new
    #        edges[where_1, 1] = new
    #    regions = new_labels[regions]
    #    _, regions = np.unique(regions, return_inverse=True)
    #    regions = regions.reshape(shape[:2])
    #    grr = np.bincount(regions.ravel()) < min_size
    #    if np.any(grr):
    #        raise ValueError("Something went wrong!")
    #    return regions, new_labels
    #
    #
    #def region_graph(regions):
    #    edges = make_grid_edges(regions)
    #    n_vertices = np.max(regions) + 1
    #
    #    crossings = edges[regions.ravel()[edges[:, 0]]
    #                      != regions.ravel()[edges[:, 1]]]
    #    edges = regions.ravel()[crossings]
    #    edges = np.sort(edges, axis=1)
    #    crossing_hash = (edges[:, 0] + n_vertices * edges[:, 1])
    #    # find unique connections
    #    unique_hash = np.unique(crossing_hash)
    #    # undo hashing
    #    unique_crossings = np.c_[unique_hash % n_vertices,
    #                             unique_hash // n_vertices]
    #    return unique_crossings
    #
    #def make_grid_edges(x, neighborhood=4, return_lists=False):
    #    if neighborhood not in [4, 8]:
    #        raise ValueError("neighborhood can only be '4' or '8', got %s" %
    #                         repr(neighborhood))
    #    inds = np.arange(x.shape[0] * x.shape[1]).reshape(x.shape[:2])
    #    inds = inds.astype(np.int64)
    #    right = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    #    down = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    #    edges = [right, down]
    #    if neighborhood == 8:
    #        upright = np.c_[inds[1:, :-1].ravel(), inds[:-1, 1:].ravel()]
    #        downright = np.c_[inds[:-1, :-1].ravel(), inds[1:, 1:].ravel()]
    #        edges.extend([upright, downright])
    #    if return_lists:
    #        return edges
    #    return np.vstack(edges)
    #
    #def get_mean_colors(image, superpixels):
    #    r = np.bincount(superpixels.ravel(), weights=image[:, :, 0].ravel())
    #    g = np.bincount(superpixels.ravel(), weights=image[:, :, 1].ravel())
    #    b = np.bincount(superpixels.ravel(), weights=image[:, :, 2].ravel())
    #    mean_colors = (np.vstack([r, g, b])
    #                   / np.bincount(superpixels.ravel())).T / 255.
    #    return mean_colors
    #
    #
    #segmentation_clean, _ = merge_small_sp(image, label(segmentation), min_size=200)
    #
    #fig, axes = plt.subplots(3, 1, figsize=(20, 10))
    #axes[0].imshow(image)
    #axes[1].imshow(segmentation_clean)
    #axes[2].imshow(mark_boundaries(image, segmentation_clean))
    
    # daisy feature descriptor
    
    features = daisy(rgb2gray(image))
    print(features.shape)
    
    few_features, desc_image = daisy(rgb2gray(image), visualize=True, step=64)
    print(few_features.shape)
    plt.imshow(desc_image)
    
    # Bag of word approach
    features = daisy(rgb2gray(image), step=30)
    all_features.append(features.reshape(-1, 200))
    
    
    
X = np.vstack(all_features)
X.shape


#by default the descriptor has a radius of 15:
radius = 15
gridx, gridy = np.mgrid[radius:image.shape[0]-radius:4, radius:image.shape[1]-radius:4]
print(gridx.shape)
# for each feature in the features array, superpixels_for_feature contains the superpixel index
superpixels_for_features = segmentation[gridx.ravel(), gridy.ravel()]
print(np.bincount(superpixels_for_features))


# K means clustering
from sklearn.cluster import KMeans
km = KMeans(n_clusters=300)
km.fit(X)
i=1

for train_image in train_images:

    image = cv2.imread(train_images[805])
    features = daisy(rgb2gray(image), step=4)
    words = km.predict(features.reshape(-1, 200))
    print(words.shape)
  
    print(superpixels_for_features.shape)
    
    from scipy.sparse import coo_matrix
    counts = coo_matrix((np.ones(len(words)), (superpixels_for_features, words))).todense()
    print("counts shape: %s max index superpixels: %d max index descriptor: %d" %(counts.shape, segmentation.max(), words.max()))

    counts = coo_matrix((np.ones(len(words)), (superpixels_for_features, words)), shape=(segmentation.max() + 1, 300)).toarray()
    print(counts.shape)
    
    labels = msrc.get_ground_truth(train_image)
    print("labels shape: %s" % str(labels.shape))
    print("segmentation shape: %s" % str(segmentation_clean.shape))
    print("Label counts on pixels:")
    print(np.bincount(labels.ravel()))  # there are three classes present: cow, grass and void.
    votes = coo_matrix((np.ones(labels.size), (segmentation_clean.ravel(), labels.ravel()))).toarray()
    print("votes shape: %s" % str(votes.shape))
    superpixel_labels = np.argmax(votes, axis=1)
    print("labels for superpixels:")
    print(superpixel_labels)
    
    
def get_words_labels(dataset, image_names, vq):
    all_superpixels, all_counts, all_labels = [], [], []
    for image_name in image_names:
        image = dataset.get_image(image_name)
        # compute features and vector-quantize
        features = daisy(rgb2gray(image), step=4)
        words = vq.predict(features.reshape(-1, 200))
        # compute and store segmentation
        segmentation = slic(image, compactness=20, sigma=0)
        segmentation_clean, _ = merge_small_sp(image, label(segmentation), min_size=200)
        all_superpixels.append(segmentation_clean)
        # find feature locations in the image, correspondence to superpixels
        radius = 15
        gridx, gridy = np.mgrid[radius:image.shape[0]-radius:4, radius:image.shape[1]-radius:4]
        superpixels_for_features = segmentation_clean[gridx.ravel(), gridy.ravel()]
        # create bag-of-word histograms
        counts = coo_matrix((np.ones(len(words)), (superpixels_for_features, words)), shape=(segmentation_clean.max() + 1, 300)).toarray()
        all_counts.append(counts)
        # compute superpixel labels
        labels = msrc.get_ground_truth(image_name)
        votes = coo_matrix((np.ones(labels.size), (segmentation_clean.ravel(), labels.ravel()))).toarray()
        print("votes shape: %s" % str(votes.shape))
        superpixel_labels = np.argmax(votes, axis=1)
        all_labels.append(superpixel_labels)
        
        
    return all_counts, all_superpixels, all_labels


counts_train, superpixels_train, y_train = get_words(msrc, train, km)


counts = np.vstack(counts)
counts.shape    
    



