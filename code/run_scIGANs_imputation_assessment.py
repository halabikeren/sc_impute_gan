import os
from subprocess import Popen
import pandas as pd
import argparse
from data_processing_utils import *



def run_scIGANs_assessment(simulation, original_data_dir, scIGANs_imputed_dir, output_dir, technology_labels):
    """
    Generates spectral clustering for each examined imputed matrix, the ground truth matrix (in case of simulations), and
    the matrix with dropouts.

    """
    if simulation:
        dropped_x_name = "count_matrix_w_dropouts.tsv"
    else:
        dropped_x_name = "full_counts.tsv"
    if technology_labels:
        labels_type = "technology_labels.txt"
    else:
        labels_type = "cell_type_labels.txt"
    scIGANs_versions = os.listdir(scIGANs_imputed_dir)
    for scIGANs_version in scIGANs_versions:
        res_dir_path = os.path.join(output_dir, scIGANs_version)
        if not os.path.isdir(os.path.join(scIGANs_imputed_dir, scIGANs_version)):
            continue
        if not os.path.exists(res_dir_path):
            os.makedirs(res_dir_path)
        job_file_path = os.path.join(res_dir_path, "analysis.sh")
        if simulation:
            prefix = "scIGANs_count_matrix_w_dropouts"
        else:
            prefix = "scIGANs_full_counts"
        print("HEEEYYYY!!!!!!!!!!", scIGANs_imputed_dir)
        file_name = get_file_name(prefix)
        input_data_path = os.path.join(scIGANs_imputed_dir, scIGANs_version, file_name)
        output_path_clustering = os.path.join(res_dir_path, "clustering.csv")
        labels_path = os.path.join(original_data_dir, labels_type)
        if not os.path.exists(output_path_clustering):
            run_job(res_dir_path, input_data_path, output_path_clustering, job_file_path, labels_path)

    # get dropped and full matrices
    # if empirical there is no full matrix
    if simulation:
        res_dir_path_full = os.path.join(output_dir, "full")
        if not os.path.exists(res_dir_path_full):
            os.makedirs(res_dir_path_full)
        input_data_path_full = os.path.join(original_data_dir, "count_matrix_wo_dropouts.tsv")
        edited_table_path = os.path.join(res_dir_path_full, "count_matrix_wo_dropouts.tsv")
        edit_input_scIGANs_tables(input_data_path_full, edited_table_path)
        output_path_clustering = os.path.join(res_dir_path_full, "clustering.csv")
        if not os.path.exists(output_path_clustering):
            job_file_path = os.path.join(res_dir_path_full, "analysis.sh")
            labels_path = os.path.join(original_data_dir, labels_type)
            run_job(res_dir_path_full, edited_table_path, output_path_clustering, job_file_path, labels_path)
    # for dropped matrices or empirical
    res_dir_dropped = os.path.join(output_dir, "w_dropouts")
    if not os.path.exists(res_dir_dropped):
        os.makedirs(res_dir_dropped)
    input_data_path_dropped = os.path.join(original_data_dir, dropped_x_name)
    edited_table_path = os.path.join(res_dir_dropped, dropped_x_name)
    edit_input_scIGANs_tables(input_data_path_dropped, edited_table_path)
    output_path_clustering = os.path.join(res_dir_dropped, "clustering.csv")
    job_file_path = os.path.join(res_dir_dropped, "analysis.sh")
    labels_path = os.path.join(original_data_dir, labels_type)
    if not os.path.exists(output_path_clustering):
        run_job(res_dir_dropped, edited_table_path, output_path_clustering, job_file_path, labels_path)


def run_job(working_dir, input_file_path, out_file_path, job_path, labels_path):
    cmd = "source /groups/itay_mayrose/anatshafir1/miniconda3/etc/profile.d/conda.sh\n"
    cmd += "conda activate googleColab\n"
    python_command = "python /groups/itay_mayrose/anatshafir1/unsupervised_learning/get_spectral_clustering.py -d " \
                     + input_file_path + " -o " + out_file_path + " -l "+labels_path+"\n"
    cmd += python_command
    job_content = create_job_format(working_dir, "assess_scIGANs", "20gb", "itaym", 8, cmd)
    job_file = open(job_path, 'w')
    job_file.write(job_content)
    job_file.close()
    Popen(["qsub", job_path])



def create_job_format(path, job_name, memory, queue, ncpu, cmd):
    text = ""
    text += "#!/bin/bash\n\n"
    text += "#PBS -S /bin/bash\n"
    text += "#PBS -r y\n"
    text += "#PBS -q "+ queue+ "\n"
    text += "#PBS -v PBS_O_SHELL=bash,PBS_ENVIRONMENT=PBS_BATCH\n"
    text += "#PBS -N "+ job_name+"\n"
    text += "#PBS -e " + path + "/"+job_name+".ER" + "\n"
    text += "#PBS -o " + path + "/" + job_name +".OU"+ "\n"
    text += "#PBS -l select=ncpus="+ str(ncpu)+ ":mem="+ memory+"\n"
    text += "cd "+ path+"\n"
    text += cmd
    text+= "\n"
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer ChromEvol models')
    parser.add_argument('--simulation','-s', type=int, help='1 if it is simulated data')
    parser.add_argument('--original_data_dir', '-d', help='original data path')
    parser.add_argument('--scIGANs_imputed_dir', '-i', help="dir of imputed data")
    parser.add_argument('--output_dir', '-o', help='path to clustering results')
    parser.add_argument('--technology_labels', '-t', type=int, help='1 if technology labels, otherwise 0 (cell types)')
    # parse arguments
    args = parser.parse_args()
    simulation = args.simulation
    original_data_dir = args.original_data_dir
    scIGANs_imputed_dir = args.scIGANs_imputed_dir
    output_dir = args.output_dir
    technology_labels = args.technology_labels
    run_scIGANs_assessment(simulation, original_data_dir, scIGANs_imputed_dir, output_dir, technology_labels)
