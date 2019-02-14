#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 16:25:18 2018

@author: Tom & AJ
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymining as pm
import seaborn as sns

from os import listdir
from os.path import isfile, join
import re

from AllPatients import separate_by_recurrence, global_remove_stageT3
from LocalFilter import load_global_patients, radial_mean_sd_for_patients, partition_patient_data_with_outliers
from plot_functions import plot_heat_map_np, plot_histogram, plot_scatter, plot_histogram, \
    plot_heat_map, save_heat_map, \
    create_polar_axis

sns.set()


# List the patient ID's of those who are contained in our ATLAS and have corrupted local maps & prothesis
# atlas = {'200806930','201010804', '201304169', '201100014', '201205737','201106120', '201204091', '200803943', '200901231', '200805565', '201101453', '200910818', '200811563','201014420'}
# corrupt = {'196708754','200801658','201201119','200911702','200701370','200700427','200610929','200606193','200600383','200511824'}
# corrupt16frac = {'200701370','200700427','200610929','200606193','200600383','200511824'}


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


def show_local_fields(global_df, dataDir=r'../Data/OnlyProstateResults/AllFields', file_name = 'untitled'):
    '''
    This loads the patients radial maps from the global data frame and labels them by their ID number. Should only
    by used for small global dataframes i.e. finding outliers from extreme bounds
    :param global_df: Data frame that contains patient list number
    :param dataDir: Directory which contains local field map
    :return: a radial plot of map title with patient ID
    '''
    df = pd.DataFrame(global_df["patientList"])
    dfFiles = df.assign(file_path=lambda df: df["patientList"].map(lambda x: r'%s/%s.csv' % (dataDir, x)))
    x = 0
    print(dfFiles.patientList)
    dfFiles.patientList.to_csv('../patient_outliers/'+file_name+'.csv')
    for f in dfFiles.file_path:
        print(dfFiles.iloc[x].patientList)
        # plot_heat_map(pd.read_csv(f, header=None), -5, 5, dfFiles.iloc[x].patientList)
        x = x + 1


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


def pyminingLocalField(selected_patients):
    # Tag patients with recurrence:1 and non-recurrence:0
    patients_who_recur, patients_who_dont_recur = separate_by_recurrence(selected_patients)
    (rec_fieldMaps, rec_label_array) = stack_local_fields(patients_who_recur, 1)
    (nonrec_fieldMaps, nonrec_label_array) = stack_local_fields(patients_who_dont_recur, 0)

    # Concatenate the two
    totalPatients = np.concatenate((rec_fieldMaps, nonrec_fieldMaps), axis=-1)
    labels = np.concatenate((rec_label_array, nonrec_label_array))

    # Now use pymining to get DSC cuts global p value. It should be similar to that from scipy
    global_neg_pvalue, global_pos_pvalue, neg_tthresh, pos_tthresh = pm.permutationTest(totalPatients, labels, 1000)
    t_value_map = pm.imagesTTest(totalPatients, labels)  # no longer.[0] element

    return global_neg_pvalue, global_pos_pvalue, neg_tthresh, pos_tthresh, t_value_map


def plot_tTest_data(neg_globalp, pos_globalp, negative_tthresh, positive_tthresh, t_value_map):
    # Print Global p value
    print('Global negative p: %.6f Global positive p: %.6f' % (neg_globalp, pos_globalp))

    # Plot Threshold histogram
    plot_histogram(negative_tthresh, 'red', 20, "Negative T-value")
    plot_histogram(positive_tthresh, 'red', 20, "Positive T-value")

    # Plot Threshhold Map
    plot_heat_map_np(t_value_map, 'maximum t-value map')
    # tThresh = sns.heatmap(max_tvalue_map, center=0, cmap='RdBu')
    # tThresh.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$')
    # plt.show()

    # # Plot Local P-values
    # p_map_upper = pValueMap(t_value_map, positive_tthresh)
    # p_map_lower = pValueMap_neg_t(t_value_map, negative_tthresh)

    # p_value_contour_plot(p_map_upper)
    # p_value_contour_plot(p_map_lower)


def t_map_with_thresholds(t_map):
    '''
    A function which will apply contours on the t-map, at values of the 5th and 95th percentiles of the
    distribution of the t-map.
    :param t_map: A 2D array of t-values
    :return: A plot of the t-map with p-contours
    '''

    critical_t_values = np.percentile(t_map.flatten(), [5, 95])
    # clrs = ['magenta', 'lime', 'orange', 'red']
    plt.contour(t_map, levels=critical_t_values, colors='magenta')
    plt.gca()
    plt.show()


def pValueMap(t_to_p_map):
    '''
    A function which will create a map of p-values from a map of t-values and thresholds
    Start at the 0th percentile, and iterate up to the 100th percentile in increments of 1.
    Upon each iteration, obtain the tthresh-value at each percentile
    Find the number of points in tthresh above the tthresh-value, to obtain a non-normalised p-value
    Normalise the p-value by dividing by the number of map elements, i.e. the size of t_to_p_map
    :param t_to_p_map: A 2D array of t-values
    :return: A 2D array of p-values
    '''

    # Make a deep copy of the t_to_p_map
    p_map = t_to_p_map.copy()

    # Define and set an iterator to initially zero, this will iterate through percentiles
    # I.e. start from percentile 0
    variableThreshold = 0

    # Loop over percentiles of the t-map, to convert the t_map->p_map
    while variableThreshold < 100:
        # Count and sum the number of points less that the variable percentile of the t-map
        pValue = sum(i < np.percentile(p_map.flatten(), variableThreshold) for i in p_map.flatten())
        pValue = pValue / 7200  # Normalise the p-values by dividing by the number of map elements
        p_map[p_map > np.percentile(p_map.flatten(), variableThreshold)] = pValue
        variableThreshold = variableThreshold + 1  # Iterate bottom up,  i.e. -ve -> +ve t

    # Returns a p-map on the scale of 0:1. Values closest to 0 represent the greatest significance for -t
    # and values closest to 1 represent the same for +t.
    return p_map


def p_value_contour_plot(t_map, t_thresh, percentile_array):
    '''
    Take in t-map, t-threshold distribution, and upper/lower tail. Will produce a map with significant contours 0.002,
    0.005, 0.01, 0.05. This uses Andrews method.
    :param t_map: it a map of t-statistic
    :param t_thresh: t threshold distribution
    :param percentile_array: list of percentiles to be contoured
    :return: t-map with p-value contours of a specific tail
    '''

    # get t at percentiles of t_thresh
    critical_t_values = np.percentile(t_thresh, percentile_array)
    # contour labels of p-values
    # p_value_names = percentile_array/100
    clrs = ['magenta', 'lime', 'orange']  # , 'red']
    plt.contour(t_map, levels=critical_t_values, colors=clrs)
    plt.gca()
    plt.show()

    # custom label names
    # strs = ['p=0.002', 'p=0.005', 'p=0.01', 'p=0.05']
    # fmt = {}
    # for l, s in zip(CS.levels, strs):
    #     fmt[l] = s
    # plt.clabel(CS, fontsize=10, fmt=fmt)
    # plt.show()


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

def semester_one_cuts(df):
    selected_patients, _, _ = partition_patient_data_with_outliers(df, 0, 99,
                                                                           discriminator_fieldname="sd")
    selected_patients, _, _ = partition_patient_data_with_outliers(selected_patients, 0, 98.5,
                                                                           discriminator_fieldname="maxval")
    selected_patients, _, _ = partition_patient_data_with_outliers(selected_patients, 5, 100,
                                                                           discriminator_fieldname="DSC")
    selected_patients, _, _ = partition_patient_data_with_outliers(selected_patients, 4, 96,
                                                                           discriminator_fieldname="volumeContourDifference")
    return selected_patients


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


def test_pymining():
    dataDirectory = r"../Data/OnlyProstateResults/AllFields"
    outputDirectory = r"../outputResults"
    rawPatientData = load_global_patients()
    enhancedDF = radial_mean_sd_for_patients(dataDirectory, rawPatientData.allPatients)
    print("Before any cuts")
    print(rawPatientData.originalAllPatients.shape)
    plot_scatter(enhancedDF, 'red')

    enhancedDF = test_cuts(enhancedDF, dataDirectory)

    # removes patients with corrupt scans
    rogue_ct_scans_dir_16 = r"../Corrupt_CT_Scans/16Fractions/"
    rogue_ct_scans_dir_19 = r"../Corrupt_CT_Scans/19Fractions/"
    rouge_ct_scans_dir_16_old = r"../Corrupt_CT_Scans/16Fractions_old/"
    rouge_ct_scans_dir_comb = r"../Corrupt_CT_Scans/Combined_Fractions/"

    # links all directory in one function
    # file_names = ['16Fractions/', '16Fractions_old/', '19Fractions/']
    # file_list = [(r'../Corrupt_CT_Scans/%s' % x) for x in file_names]

    # Removes the selected corrupt ct scan patients from enhancedDF
    selected_patients = get_corrupt_patients(enhancedDF, rouge_ct_scans_dir_comb)
    print("After ct scan cuts")
    print(selected_patients.shape)
    # t-statistics
    (global_neg_pvalue, global_pos_pvalue, neg_tthresh, pos_tthresh, t_value_map) = pyminingLocalField(
        enhancedDF)
    print('Global negative p: %.6f Global positive p: %.6f' % (global_neg_pvalue, global_pos_pvalue))
    plot_heat_map_np(t_value_map[0], 'maximum t-value map')
    t_map_with_thresholds(t_value_map[0])
    plot_histogram(t_value_map[0].flatten(),'magenta', 50, 't-distrubtion of map')
    plot_scatter(enhancedDF, 'lime')
    # plot_sample_mean_and_sd_maps(selected_patients)

if __name__ == '__main__':
    # method_of_refining_data()
    # test_cuts()
    test_pymining()
