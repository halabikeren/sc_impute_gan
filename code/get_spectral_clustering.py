import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import rand_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import os
from sklearn.manifold import TSNE
import colorcet as cc
import re
import time
import itertools
import umap as umap
import umap.plot 
import argparse
from data_processing_utils import *



def cluster_labels(data_path, outfile, labels_path):
    """
    Input:
    data_path: scRNA-seq dataframe in tsv format
    outfile: path to the clustering labels output
    labels_path: path to the true labels
    """
    df = pd.read_csv(data_path, index_col=0)
    if len(df.columns) == 0:
        df = load_tsv_data(data_path)
    labels = load_label_data(labels_path)
    mat = df.to_numpy()
    #true_labels = load_label_data(labels_data_path)
    # dividing by the max value of each column
    mat_n = mat / np.max(mat, axis=0)
    #mat_n_c = mat_n - np.mean(mat_n, axis=0)
    number_of_labels = len(set(labels))
    #pca = PCA(n_components=2)
    #pcs = pca.fit_transform(mat_n_c)
    #explained_variance_components = pca.explained_variance_ratio_
    #req_explained_variance = 0.8
    #number_of_components = 2
      # to be changed
    # print("Number of required clusters", str(number_of_labels))
    # explained_variance = np.sum(explained_variance_components)
    # while explained_variance < req_explained_variance:
    #     number_of_components += 1
    #     pca = PCA(n_components=2)
    #     pcs = pca.fit_transform(mat_n_c)
    #     explained_variance_components = pca.explained_variance_ratio_
    #     if number_of_components == mat.shape[0]:
    #         break
    number_of_components = 3
    X_embedded = TSNE(n_components=number_of_components, learning_rate="auto", init='random',
                      perplexity=mat.shape[1] // 4, ).fit_transform(mat_n.T)
    # Building the clustering model
    # running spectral clustering on TSNE embeddings as it was done is scIGANs
    spectral_model = SpectralClustering(n_clusters=number_of_labels, affinity='rbf', assign_labels='discretize').fit(
        X_embedded)
    labels_rbf = spectral_model.labels_
    new_df = pd.DataFrame({"labels": labels_rbf})
    new_df.to_csv(outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer ChromEvol models')
    parser.add_argument('--data_path', '-d', help='fata path')
    parser.add_argument('--output_path', '-o', help='out file')
    parser.add_argument('--labels_path', '-l', help='label path')
    # parse arguments
    args = parser.parse_args()
    data_path = args.data_path
    output_path = args.output_path
    labels_path = args.labels_path
    cluster_labels(data_path, output_path, labels_path)
