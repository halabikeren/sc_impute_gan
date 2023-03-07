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
import pickle
import matplotlib

def creates_plots_for_scIGANs_analyses(simulation, original_data_dir, scIGANs_imputed_dir,
                           output_dir, technology_labels, dir_plots_output):
    """
    Creates tables and plots to analyze scIGANs2 improvements
    Input:
    simulation: 1 if simulated data. 0 otherwise
    original_data_dir: path to the generated datasets
    scIGANs_imputed_dir: path to the directory with scIGANs results (imputed matrices)
    output_dir: output directory to save the data analyses results
    technology_labels: 1 if the clustering should be performed with respect to technology labels.
        Otherwise, it is performed with respect to cell type labels
    dir_plots_output: output directory for data analyses
    returns: No return
    """
    dict_df_with_stats = {}
    # noise_as_input: first improvement
    # image_as_input_no_partitioning_w_noise: second improvement with adding noise
    # image_as_input_no_partitioning_wo_noise: second improvement without adding noise
    # image_as_input_partitioning0_no_noise: second improvement without adding noise partitionining 0
    # image_as_input_partitioning1_no_noise: second improvement without adding noise partitionining 1
    # image_as_input_partitioning2_no_noise: second improvement without adding noise partitionining 2
    output_stats_table = os.path.join(dir_plots_output, "summary_stats.csv")
    output_plot_path = os.path.join(dir_plots_output, "umap_plots.png")
    dict_scIGANs_version_names = {"full":"ground truth", "w_dropouts":"data before imputation", "noise_as_input":"scIGANs2_only_BE", "image_as_input_no_partitioning_w_noise":" scIGANs2+BE+DAI_sample+noise",
                                  "image_as_input_no_partitioning_wo_noise":" scIGANs2+BE+DAI_sample", "image_as_input_partitioning0_no_noise":"scIGANs2+BE+DAI_p0",
                                  "image_as_input_partitioning1_no_noise":"scIGANs2+BE+DAI_p1", "image_as_input_partitioning2_no_noise":"scIGANs2+BE+DAI_p2",
                                  "scIGANs_original":"scIGANs"}

    # plot_indices_dict = {"scIGANs1":[0,0], "scIGANs2+noise":[0,1], "scIGANs2":[0,2], "scIGANs2.0":[0,3],
    #                      "scIGANs2.1":[1,0], "scIGANs2.2":[1,1], "dropped out":[1,2], "full":[1,3]}
    # adjusting the figure structure according to the returned scIGANs results
    dict_count_indices_simulated = {0:[0,0], 1:[0,1], 2:[0,2], 3:[1,0], 4:[1,1], 5:[1,2], 6:[2,0], 7:[2,1], 8:[2,2]}
    dict_count_indices_empirical = {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}

    if simulation:
        dropped_x_name = "count_matrix_w_dropouts.tsv"
        plot_indices_dict = dict_count_indices_simulated
        rows = 3
        columns = 3
        plot_size = (15, 15)
    else:
        dropped_x_name = "full_counts.tsv"
        plot_indices_dict = dict_count_indices_empirical
        rows = 2
        columns = 2
        plot_size = (10,10)
    fig, axes = plt.subplots(rows, columns, figsize=plot_size)
    if technology_labels:
        labels_type = "technology_labels.txt"
    else:
        labels_type = "cell_type_labels.txt"
    scIGANs_versions = os.listdir(scIGANs_imputed_dir)
    counter = 0
    for scIGANs_version in scIGANs_versions:
        res_dir_path = os.path.join(output_dir, scIGANs_version)
        if not os.path.isdir(os.path.join(scIGANs_imputed_dir, scIGANs_version)):
            continue
        if simulation:
            prefix = "scIGANs_count_matrix_w_dropouts"
        else:
            prefix = "scIGANs_full_counts"

        file_name = get_file_name(prefix)
        input_data_path = os.path.join(scIGANs_imputed_dir, scIGANs_version, file_name)
        print("##### path is ", input_data_path)
        if not os.path.exists(input_data_path):
            continue
        output_path_clustering = os.path.join(res_dir_path, "clustering.csv")
        umap_path_results = os.path.join(res_dir_path, "umap.pkl")
        if not os.path.exists(output_path_clustering):
            continue
        labels_path = os.path.join(original_data_dir, labels_type)
        version_name = dict_scIGANs_version_names[scIGANs_version]
        run_umap(input_data_path, labels_path, axes, plot_indices_dict[counter], version_name, umap_path_results)
        evaluate_stats(dict_df_with_stats, output_path_clustering, labels_path, version_name)
        counter += 1


    # get dropped and full matrices
    # if empirical there is no full matrix
    if simulation:
        print("##### path is ", os.path.join(original_data_dir,"count_matrix_wo_dropouts.tsv"))
        res_dir_path_full = os.path.join(output_dir, "full")
        input_data_path_full = os.path.join(original_data_dir, "count_matrix_wo_dropouts.tsv")
        edited_table_path = os.path.join(res_dir_path_full, "count_matrix_wo_dropouts.tsv")
        edit_input_scIGANs_tables(input_data_path_full, edited_table_path)
        output_path_clustering = os.path.join(res_dir_path_full, "clustering.csv")
        umap_path_results = os.path.join(res_dir_path_full, "umap.pkl")
        labels_path = os.path.join(original_data_dir, labels_type)
        version_name = dict_scIGANs_version_names["full"]
        run_umap(edited_table_path, labels_path, axes, plot_indices_dict[counter], version_name, umap_path_results)
        evaluate_stats(dict_df_with_stats, output_path_clustering, labels_path, version_name)
        counter += 1
    # for dropped matrices or empirical
    res_dir_dropped = os.path.join(output_dir, "w_dropouts")
    input_data_path_dropped = os.path.join(original_data_dir, dropped_x_name)
    print("##### path is ", input_data_path_dropped)
    edited_table_path = os.path.join(res_dir_dropped, dropped_x_name)
    edit_input_scIGANs_tables(input_data_path_dropped, edited_table_path)
    output_path_clustering = os.path.join(res_dir_dropped, "clustering.csv")
    umap_path_results = os.path.join(res_dir_dropped, "umap.pkl")
    labels_path = os.path.join(original_data_dir, labels_type)
    version_name = dict_scIGANs_version_names["w_dropouts"]
    run_umap(edited_table_path, labels_path, axes, plot_indices_dict[counter], version_name, umap_path_results)
    evaluate_stats(dict_df_with_stats, output_path_clustering, labels_path, version_name)
    df_res = pd.DataFrame(dict_df_with_stats)
    df_res.index = np.array(["MI", "AMI", "NMI", "RI", "ARI", "ACC", "F-score"])
    df_res.to_csv(output_stats_table)
    counter += 1

    plt.tight_layout()
    plt.savefig(output_plot_path)




def evaluate_stats(dict_df_with_stats, output_path_clustering, labels_path, version_name):
    """
    Evaluate clustering results
    Input:
    dict_df_with_stats: part of dictionary that should be filled with the statistics concerning the examined program version
    output_path_clustering: path to the clustering results
    labels_path: path to the true label path
    version_name: the type of data to be clustered
    returns: No return
    """
    true_labels = load_label_data(labels_path)
    labels_clustered_df = pd.read_csv(output_path_clustering, index_col=0)
    labels_clustered = labels_clustered_df["labels"].to_numpy()
    true_labels = adjust_labels(true_labels, labels_clustered, len(set(true_labels)))
    mi = mutual_info_score(true_labels, labels_clustered)
    ami = adjusted_mutual_info_score(true_labels, labels_clustered)
    nmi = normalized_mutual_info_score(true_labels, labels_clustered)
    ri = rand_score(true_labels, labels_clustered)
    ari = adjusted_rand_score(true_labels, labels_clustered)
    acc = accuracy_score(true_labels, labels_clustered)
    f_score = f1_score(true_labels, labels_clustered, average='weighted')
    dict_df_with_stats[version_name] = [mi, ami, nmi, ri, ari, acc, f_score]


def run_umap(X_path, y_path, axes, axes_indices, version_name, umap_path_results):
    """
    Runs umap and creates umap plots.
    Input:
    X_path: scRNA seq data
    y_path: path to the true labels
    axes: axes of plot
    axes_indices: indices for axes
    version_name: type of provided matrix (scIGANs,ground truth, etc.)
    umap_path_results: path to pickle umap results
    """
    true_labels = load_label_data(y_path)
    if not os.path.exists(umap_path_results):
        df_X = pd.read_csv(X_path, index_col=0)
        if len(df_X.columns) == 0:
            df_X = load_tsv_data(X_path)
        print(X_path)
        print("sahpe of X is", df_X.shape)
        mat = df_X.to_numpy()
        mat_n = mat / np.max(mat, axis=0)
        X_embedded = umap.UMAP().fit_transform(mat_n.T)
        pickle.dump(X_embedded, open(umap_path_results, "wb"))
    else:
        with open(umap_path_results, 'rb') as handle:
            X_embedded = pickle.load(handle)

    df = pd.DataFrame({"umap_1": X_embedded[:, 0], "umap_2": X_embedded[:, 1], "object": true_labels})
    colors = sns.color_palette(cc.glasbey, n_colors=len(set(true_labels)))
    sns.scatterplot(ax=axes[axes_indices[0], axes_indices[1]], data=df, x="umap_1", y="umap_2", hue="object", palette=colors)
    axes[axes_indices[0], axes_indices[1]].set_title(version_name)
    if axes_indices[0] !=0 or axes_indices[1] != 0:
        axes[axes_indices[0], axes_indices[1]].get_legend().remove()



def adjust_labels(true_labels, clustered_labels, num_of_clusters):
    """
    Adjusts true labels to integer labeling as it is done in clusterinf algorithms
    input:
    true_labels: np.array of true labels
    clustered_labels: clustered labels
    num_of_clusters: number of clusters
    Returns:
    Adjusted integer labels
    """
    dict_indices = {}
    true_labels_unique = sorted(list(set(true_labels)))
    for i in range(len(true_labels_unique)):
        dict_indices[true_labels_unique[i]] = i
    int_labels = np.arange(num_of_clusters)
    permutations = list(itertools.permutations(int_labels))
    best_permutation = None
    permutation_accuracy = 0
    for i in range(len(permutations)):
        permutation = permutations[i]
        adjusted_labels_lst = []
        for j in range(len(true_labels)):
            adjusted_labels_lst.append(permutation[dict_indices[true_labels[j]]])
        adjusted_labels = np.array(adjusted_labels_lst)
        intersected = np.sum(adjusted_labels == clustered_labels)
        if intersected > permutation_accuracy:
            best_permutation = adjusted_labels
            permutation_accuracy = intersected
    return best_permutation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer ChromEvol models')
    parser.add_argument('--simulation','-s', type=int, help='1 if it is simulated data')
    parser.add_argument('--original_data_dir', '-d', help='original data path')
    parser.add_argument('--scIGANs_imputed_dir', '-i', help="dir of imputed data")
    parser.add_argument('--output_dir', '-o', help='path to clustering results')
    parser.add_argument('--technology_labels', '-t', type=int, help='1 if technology labels, otherwise 0 (cell types)')
    parser.add_argument('--dir_plots_output', '-p', help = "plots output directory")
    # parse arguments
    args = parser.parse_args()
    simulation = args.simulation
    original_data_dir = args.original_data_dir
    scIGANs_imputed_dir = args.scIGANs_imputed_dir
    output_dir = args.output_dir
    technology_labels = args.technology_labels
    dir_plots_output = args.dir_plots_output
    creates_plots_for_scIGANs_analyses(simulation, original_data_dir, scIGANs_imputed_dir, output_dir, technology_labels, dir_plots_output)
