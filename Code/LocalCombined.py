#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 16:25:18 2018

@author: Tom & AJ

This is used to create temporary functions and run the main source of code
"""
import numpy as np
import pandas as pd
import pymining as pm
import seaborn as sns

from os import listdir
from os.path import isfile, join
import re

from AllPatients import separate_by_recurrence, global_remove_stageT3
from LocalFilter import load_global_patients, radial_mean_sd_for_patients, partition_patient_data_with_outliers, \
    select_atlas
from plot_functions import plot_heat_map_np, plot_histogram, plot_scatter, plot_histogram, \
    plot_heat_map, show_local_fields, plot_sample_mean_and_sd_maps\
import significance_test


sns.set()


def make_average_field(global_df, dataDir=r'../Data/OnlyProstateResults/AllFields'):
    '''
    Produces average radial difference per voxel for all patients within the global dataframe
    :param global_df: global data frame with patients which will contribute to the ave and variance map
    :param dataDir: local field directory
    :return: map of average, variance, std
    '''
    df = pd.DataFrame(global_df["patientList"])
    dfFiles = df.assign(file_path=lambda df: df["patientList"].map(lambda x: r'%s/%s.csv' % (dataDir, x)))
    masterDF = pd.DataFrame.empty
    fieldMaps = []
    x = 0
    for f in dfFiles.file_path:
        fieldMaps.append(pd.read_csv(f, header=None))
        x = x + 1

    masterDF = pd.concat(fieldMaps)
    by_row_indexRec = masterDF.groupby(masterDF.index)
    mean_field = by_row_indexRec.mean()
    variance_field = by_row_indexRec.var()
    std_field = by_row_indexRec.std()

    return mean_field, variance_field, std_field


def stack_local_fields(global_df, recurrence_label, dataDir=r'../Data/OnlyProstateResults/AllFields'):
    """
    :param global_df: either recurring non-recurring global data field
    :param recurrence_label:  =0 for non-recurring or =1 for recurring
    :param dataDir: Data directory containing local field data file
    :return: 3d np array of local field stacked 120x60xnumber of recurrence/non-recurrence i.e [theta x phi x patient_index] and label
    """
    df = pd.DataFrame(global_df["patientList"])
    dfFiles = df.assign(file_path=lambda df: df["patientList"].map(lambda x: r'%s/%s.csv' % (dataDir, x)))
    numberOfPatients = len(dfFiles)
    fieldMaps = np.zeros((60, 120, numberOfPatients))

    if recurrence_label == 1:
        label_array = np.ones(numberOfPatients)
    else:
        label_array = np.zeros(numberOfPatients)

    i = 0
    for f in dfFiles.file_path:
        fieldMaps[:, :, i] = pd.read_csv(f, header=None).as_matrix()[:, :]
        i += 1

    return fieldMaps, label_array


def print_volume_difference_details(patientsDF):
    '''
    :param patientsDF: is a dataframe that contains the patients global variables
    :return: prints volume difference statistics for collectiver patient, then patients with recurrence and patients /
    without
    '''
    print(patientsDF["volumeContourDifference"].describe())
    patients_who_recur, patients_who_dont_recur = separate_by_recurrence(patientsDF)
    print(patients_who_recur["volumeContourDifference"].describe())
    print(patients_who_dont_recur["volumeContourDifference"].describe())


def get_corrupt_patients(all_patients_df, data_directory):
    # Create list of filepaths
    file_list = [f for f in listdir(data_directory) if isfile(join(data_directory, f))]
    # parse filename and extract id
    patient_id = list(map(lambda x: re.split('[_ .]', x)[1], file_list))
    # print(patient_id)
    clean_patients_ct_scans = all_patients_df[~all_patients_df['patientList'].isin(patient_id)]
    return clean_patients_ct_scans


def test_cuts(enhancedDF, dataDirectory):
    # Plotting histograms before any cuts
    plot_histogram(enhancedDF['sd'], 'red', 50, name="Standard Deviations of patients map")
    plot_histogram(enhancedDF['maxval'], 'lime', 50, name="Maximum value of patients map")
    plot_histogram(enhancedDF['DSC'], 'green', 50, name="Dice of patients map")
    plot_histogram(enhancedDF['volumeContourDifference'], 'blue', 50, name="Volume difference of patients map")

    # Look for poor triangulation of patient maps
    # SD of patient map
    _, _, upper_bound = partition_patient_data_with_outliers(enhancedDF, 0, 98, "sd")
    show_local_fields(upper_bound, dataDirectory, 'upper_sd_bound_combined')
    # Delete the upperbound of std. dev. maps
    clean_patients = enhancedDF[~enhancedDF['patientList'].isin(upper_bound['patientList'])]
    # Histogram of std. dev. after cuts
    plot_histogram(clean_patients['sd'], 'red', 50, name="Standard Deviations of patients map")

    # Max value of Patient
    _, lower_bound, upper_bound = partition_patient_data_with_outliers(enhancedDF, 2, 98, "maxval")
    show_local_fields(upper_bound, dataDirectory, 'upper_max_R_bound_combined')
    show_local_fields(lower_bound, dataDirectory, 'lower_max_R_bound_combined')
    clean_patients = clean_patients[~clean_patients['patientList'].isin(upper_bound['patientList'])]
    clean_patients = clean_patients[~clean_patients['patientList'].isin(lower_bound['patientList'])]

    # Histogram of removed sd and max radial difference
    plot_histogram(clean_patients['maxval'], 'lime', 50, "Maximum value of patients map")

    print("After statistical scan cuts")
    print(clean_patients.shape)
    return clean_patients


def test_function():
    dataDirectory = r"../Data/OnlyProstateResults/AllFields"
    outputDirectory = r"../outputResults"
    rawPatientData = load_global_patients()
    enhancedDF = radial_mean_sd_for_patients(dataDirectory, rawPatientData.allPatients)
    print("Before any cuts")
    print(rawPatientData.originalAllPatients.shape)

    # Statistical cuts
    enhancedDF = test_cuts(enhancedDF, dataDirectory)

    # removes patients with corrupt scans
    rogue_ct_scans_dir_16 = r"../Corrupt_CT_Scans/16Fractions/"
    rogue_ct_scans_dir_19 = r"../Corrupt_CT_Scans/19Fractions/"
    rouge_ct_scans_dir_16_old = r"../Corrupt_CT_Scans/16Fractions_old/"
    rouge_ct_scans_dir_comb = r"../Corrupt_CT_Scans/Combined_Fractions/"

    # Removes the selected corrupt ct scan patients from enhancedDF
    selected_patients = get_corrupt_patients(enhancedDF, rouge_ct_scans_dir_comb)

    # t-statistics
    (global_neg_pvalue, global_pos_pvalue, neg_tthresh, pos_tthresh, t_value_map) = significance_test.pymining_t_test(
        enhancedDF)
    print('Global negative p: %.6f Global positive p: %.6f' % (global_neg_pvalue, global_pos_pvalue))
    plot_heat_map_np(t_value_map[0], 'maximum t-value map')
    significance_test.t_map_with_thresholds(t_value_map[0])
    plot_histogram(t_value_map[0].flatten(),'magenta', 50, 't-distrubtion of map')
    plot_scatter(enhancedDF, 'lime')

    # wilcoxon statistics
    w_stat, p_map = significance_test.wilcoxon_test_statistics(enhancedDF)
    plot_heat_map_np(w_stat, 'wilcoxon map')
    plot_heat_map_np(p_map, 'p map')

if __name__ == '__main__':
    # method_of_refining_data()
    # test_cuts()
    test_function()
