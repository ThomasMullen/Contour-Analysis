#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 16:25:18 2018

@author: Tom & AJ

This is used to create temporary functions and run the main source of code
"""
import re
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import seaborn as sns

from AllPatients import separate_by_recurrence, separate_by_risk
from LocalFilter import load_global_patients, radial_mean_sd_for_patients, partition_patient_data_with_outliers
from plot_functions import plot_heat_map_np, plot_scatter, plot_histogram, plot_heat_map, show_local_fields, \
    test_on_single_map, triangulation_qa
from significance_test import wilcoxon_test_statistics, pymining_t_test, t_map_with_thresholds, test_superimpose, \
    global_statistical_analysis

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


def plot_sample_mean_and_sd_maps(selected_patients):
    dataDirectory = r"../Data/OnlyProstateResults/AllFields"
    patients_who_recur, patients_who_dont_recur = separate_by_recurrence(selected_patients)
    # (meanMap1, varMap, stdMap) = load_local_field_recurrence(selected_patients, dataDirectory)

    (meanMap1, varMap1, stdMap1) = make_average_field(patients_who_recur, dataDirectory)

    meanMap1.to_csv("../outputResults/recurrence_mean_map.csv", header=None, index=False)
    stdMap1.to_csv("../outputResults/recurrence_std_map.csv", header=None, index=False)

    plot_heat_map(meanMap1, -1, 1, 'mean map - patients_who_recur')
    plot_heat_map(varMap1, 0, 1, 'variance map - patients_who_recur')
    plot_heat_map(stdMap1, 0, 1, 'standard deviation map - patients_who_recur')

    (meanMap2, varMap2, stdMap2) = make_average_field(patients_who_dont_recur, dataDirectory)
    plot_heat_map(meanMap2, -1, 1, 'mean map - patients_who_dont_recur')
    plot_heat_map(varMap2, 0, 1, 'variance map - patients_who_dont_recur')
    plot_heat_map(stdMap2, 0, 1, 'standard deviation map - patients_who_dont_recur')

    meanMap2.to_csv("../outputResults/no_recurrence_mean_map.csv", header=None, index=False)
    stdMap2.to_csv("../outputResults/no_recurrence_std_map.csv", header=None, index=False)

    plot_heat_map(meanMap1 - meanMap2, -0.3, 0.3, 'Difference in mean map')
    # Var[X-Y] = Var[X]+Var[Y]
    # Standard deviation is the square root of the variance
    plot_heat_map(np.sqrt(varMap1 + varMap2), 0, 1.5, 'Difference in std map')
    (meanMap1 - meanMap2).to_csv("../outputResults/mean_difference_map.csv", header=None, index=False)
    np.sqrt(varMap1 + varMap2).to_csv("../outputResults/std_difference_map.csv", header=None, index=False)



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
    '''
    Reads in filenames from data directory, exstracts the patient ids and removes them from the dataset
    :param all_patients_df: global dataframe
    :param data_directory: data directory that has the rogue patient names
    :return: returns clean dataset
    '''
    # Create list of filepaths
    file_list = [f for f in listdir(data_directory) if isfile(join(data_directory, f))]
    # parse filename and extract id
    patient_id = list(map(lambda x: re.split('[_ .]', x)[1], file_list))
    # print(patient_id)
    clean_patients_ct_scans = all_patients_df[~all_patients_df['patientList'].isin(patient_id)]
    return clean_patients_ct_scans


def statistical_cuts(enhancedDF, dataDirectory=r"../Data/OnlyProstateResults/AllFields"):
    '''
    Produce histograme plots and within has partition cuts base on patients max delta r, std, dsc, and vol diff
    :param enhancedDF: global patient df
    :param dataDirectory: Directory of the patient radial maps
    :return: clean dataset
    '''
    # Plotting histograms before any cuts
    plot_histogram(enhancedDF['sd'], 'red', 50, name="Standard Deviations of patients map")
    plot_histogram(enhancedDF['maxval'], 'lime', 50, name="Maximum value of patients map")
    plot_histogram(enhancedDF['DSC'], 'green', 50, name="Dice of patients map")
    plot_histogram(enhancedDF['volumeContourDifference'], 'blue', 50, name="Volume difference of patients map")

    # Look for poor triangulation of patient maps
    # SD of patient map
    _, _, upper_bound = partition_patient_data_with_outliers(enhancedDF, 0, 5, "sd")
    clean_patients = enhancedDF[~enhancedDF['patientList'].isin(upper_bound['patientList'])]
    print(clean_patients.shape)

    # Max value of Patient
    _, lower_bound, upper_bound = partition_patient_data_with_outliers(enhancedDF, -5, 5, "maxval")
    clean_patients = clean_patients[~clean_patients['patientList'].isin(upper_bound['patientList'])]
    clean_patients = clean_patients[~clean_patients['patientList'].isin(lower_bound['patientList'])]
    print(clean_patients.shape)

    # Dice Coefficient
    _, lower_bound, _ = partition_patient_data_with_outliers(enhancedDF, 0.7, 0.999, "DSC")
    clean_patients = clean_patients[~clean_patients['patientList'].isin(lower_bound['patientList'])]
    print(clean_patients.shape)
    return clean_patients


def read_and_return_patient_stats(calculated_csv_name="test", dataDirectory=r"../Data/OnlyProstateResults/AllFields"):
    '''
    Loads up and concatenates table then writes a single data frame with mean, st and max r value for each patient
    map added
    :param dataDirectory: The directory that stores all the patient fields
    :param calculated_csv_name: name of the new csv
    :return:
    '''
    rawPatientData = load_global_patients()
    enhancedDF = radial_mean_sd_for_patients(dataDirectory, rawPatientData.allPatients)
    enhancedDF.to_csv('../Data/OnlyProstateResults/' + calculated_csv_name + '.csv')


def cuts_from_ct_scans(global_df):
    # removes patients with corrupt scans
    rogue_ct_scans_dir_16 = r"../Corrupt_CT_Scans/16Fractions/"
    rogue_ct_scans_dir_19 = r"../Corrupt_CT_Scans/19Fractions/"
    rouge_ct_scans_dir_16_old = r"../Corrupt_CT_Scans/16Fractions_old/"
    rouge_ct_scans_dir_comb = r"../Corrupt_CT_Scans/Combined_Fractions/"

    # Removes the selected corrupt ct scan patients from enhancedDF
    global_df = get_corrupt_patients(global_df, rouge_ct_scans_dir_comb)
    return global_df


def test_analysis_function():
    dataDirectory = r"../Data/OnlyProstateResults/AllFields"
    enhancedDF = pd.read_csv(r'../Data/OnlyProstateResults/All_patient_data.csv')

    # Statistical cuts
    # enhancedDF = statistical_cuts(enhancedDF)

    # t-statistics
    # global_neg_p_value, global_pos_p_value, neg_t_thresh, pos_t_thresh, t_value_map = pymining_t_test(enhancedDF)
    # print('Global negative p: %.6f Global positive p: %.6f' % (global_neg_p_value, global_pos_p_value))
    # # plot_heat_map_np(t_value_map[0], 'maximum t-value map')
    # # t_map_with_thresholds(t_value_map[0])
    # plot_histogram(t_value_map[0].flatten(), 'magenta', 50, 't-distrubtion of map')
    # plot_scatter(enhancedDF, 'lime')
    # test_superimpose(t_value_map[0], pos_t_thresh, neg_t_thresh)


    global_neg_p_value, global_pos_p_value, neg_t_thresh, pos_t_thresh, t_value_map = pymining_t_test(enhancedDF)
    print('Global negative p: %.6f Global positive p: %.6f' % (global_neg_p_value, global_pos_p_value))
    # plot_heat_map_np(t_value_map[0], 'maximum t-value map')
    # t_map_with_thresholds(t_value_map[0])
    plot_histogram(t_value_map[0].flatten(), 'magenta', 50, 't-distrubtion of map')
    plot_scatter(enhancedDF, 'lime')
    test_superimpose(t_value_map[0], pos_t_thresh, neg_t_thresh)
    global_statistical_analysis(enhancedDF)

    global_neg_p_value, global_pos_p_value, neg_t_thresh, pos_t_thresh, t_value_map = pymining_t_test(separate_by_risk(enhancedDF)[0])
    print('Global negative p: %.6f Global positive p: %.6f' % (global_neg_p_value, global_pos_p_value))
    # plot_heat_map_np(t_value_map[0], 'maximum t-value map')
    # t_map_with_thresholds(t_value_map[0])
    plot_histogram(t_value_map[0].flatten(), 'magenta', 50, 't-distrubtion of map')
    plot_scatter(separate_by_risk(enhancedDF)[0], 'lime')
    test_superimpose(t_value_map[0], pos_t_thresh, neg_t_thresh)
    global_statistical_analysis(separate_by_risk(enhancedDF)[0])

    global_neg_p_value, global_pos_p_value, neg_t_thresh, pos_t_thresh, t_value_map = pymining_t_test(separate_by_risk(enhancedDF)[1])
    print('Global negative p: %.6f Global positive p: %.6f' % (global_neg_p_value, global_pos_p_value))
    # plot_heat_map_np(t_value_map[0], 'maximum t-value map')
    # t_map_with_thresholds(t_value_map[0])
    plot_histogram(t_value_map[0].flatten(), 'magenta', 50, 't-distrubtion of map')
    plot_scatter(separate_by_risk(enhancedDF)[1], 'lime')
    test_superimpose(t_value_map[0], pos_t_thresh, neg_t_thresh)
    global_statistical_analysis(separate_by_risk(enhancedDF)[1])

    global_neg_p_value, global_pos_p_value, neg_t_thresh, pos_t_thresh, t_value_map = pymining_t_test(separate_by_risk(enhancedDF)[2])
    print('Global negative p: %.6f Global positive p: %.6f' % (global_neg_p_value, global_pos_p_value))
    # plot_heat_map_np(t_value_map[0], 'maximum t-value map')
    # t_map_with_thresholds(t_value_map[0])
    plot_histogram(t_value_map[0].flatten(), 'magenta', 50, 't-distrubtion of map')
    plot_scatter(separate_by_risk(enhancedDF)[2], 'lime')
    test_superimpose(t_value_map[0], pos_t_thresh, neg_t_thresh)
    global_statistical_analysis(separate_by_risk(enhancedDF)[2])



    # wilcoxon statistics
    # w_stat, p_map = wilcoxon_test_statistics(enhancedDF)
    # plot_heat_map_np(w_stat, 'wilcoxon map')
    # plot_heat_map_np(p_map, 'p map')


if __name__ == '__main__':
    # method_of_refining_data()
    # test_cuts()
    # read_and_return_patient_stats("All_patient_data")
    test_analysis_function()
    # triangulation_qa()
    # test_on_single_map()
