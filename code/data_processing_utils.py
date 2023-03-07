import pandas as pd
import numpy as np


def load_tsv_data(data_path):
    """
    Input:
    scRNA-seq dataframe in tsv format
    returns: a dataframe that represents the scRNA-seq data
    """
    df = pd.read_csv(data_path, sep="\t")
    row_names = df["GENE_ID"]
    df.index = row_names
    df = df.iloc[:, 1:]
    return df


def edit_input_scIGANs_tables(src_data_path, dst_data_path):
    """
    converts tsv to csv
    Input:
    src_data_path: the path to the tsv format file
    dst_data_path: the file path to the resulted table in csv format
    returns: No return
    """
    df = pd.read_csv(src_data_path, sep="\t")
    row_names = df["GENE_ID"]
    df.index = row_names
    df = df.iloc[:, 1:]
    df.to_csv(dst_data_path)


def load_label_data(labels_data_path):
    """
    loads label data (either technology labels or cell types)
    Input:
    labels_data_path: the path to the label data
    returns: np.array of labels (either technoology or cell types)
    """
    true_labels = pd.read_csv(labels_data_path, sep='\t')
    first_col = np.array([true_labels.columns[0]])
    true_labels = true_labels.iloc[:, 0].to_numpy()
    true_labels = np.append(first_col, true_labels)
    return true_labels


def get_file_name(prefix):
    """
    getting the name of scIGANs output imputed matrix
    Input:
    returns: the respective file name
    """
    return prefix+".tsv"+ "_cell_type_labels.txt.txt"



