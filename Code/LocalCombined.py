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

from AllPatients import separate_by_recurrence
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
    plot_heat_map(meanMap1, -1, 1, 'mean map - patients_who_recur')
    plot_heat_map(varMap1, 0, 1, 'variance map - patients_who_recur')
    plot_heat_map(stdMap1, 0, 1, 'standard deviation map - patients_who_recur')

    (meanMap2, varMap2, stdMap2) = make_average_field(patients_who_dont_recur, dataDirectory)
    plot_heat_map(meanMap2, -1, 1, 'mean map - patients_who_dont_recur')
    plot_heat_map(varMap2, 0, 1, 'variance map - patients_who_dont_recur')
    plot_heat_map(stdMap2, 0, 1, 'standard deviation map - patients_who_dont_recur')

    plot_heat_map(meanMap1 - meanMap2, -0.3, 0.3, 'Difference in mean map')
    # Var[X-Y] = Var[X]+Var[Y]
    # Standard deviation is the square root of the variance
    plot_heat_map(np.sqrt(varMap1 + varMap2), 0, 1.5, 'Difference in std map')


def show_local_fields(global_df, dataDir=r'../Data/OnlyProstateResults/AllFields'):
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
    for f in dfFiles.file_path:
        print(dfFiles.iloc[x].patientList)
        plot_heat_map(pd.read_csv(f, header=None), -3, 3, dfFiles.iloc[x].patientList)
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
    global_neg_pvalue, global_pos_pvalue, neg_tthresh, pos_tthresh = pm.permutationTest(totalPatients, labels, 100)
    t_value_map = pm.imagesTTest(totalPatients, labels)  # no longer.[0] element

    return global_neg_pvalue, global_pos_pvalue, neg_tthresh, pos_tthresh, t_value_map


def plot_tTest_data(neg_globalp, pos_globalp, negative_tthresh, positive_tthresh, t_value_map):
    # Print Global p value
    # print('Global negative p: %.6f Global positive p: %.6f' % neg_globalp, pos_globalp)

    # Plot Threshold histogram
    plot_histogram(negative_tthresh, 'red', 20, "Negative T-value")
    plot_histogram(positive_tthresh, 'red', 20, "Positive T-value")

    # Plot Threshhold Map
    plot_heat_map_np(t_value_map, 'maximum t-value map')
    # tThresh = sns.heatmap(max_tvalue_map, center=0, cmap='RdBu')
    # tThresh.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$')
    # plt.show()

    # Plot Local P-values
    p_map_upper = pValueMap_pos_t(t_value_map, positive_tthresh)
    p_map_lower = pValueMap_neg_t(t_value_map, negative_tthresh)
    
    p_value_contour_plot(p_map_upper)
    p_value_contour_plot(p_map_lower)


def pValueMap_pos_t(t_to_p_map, t_thresh):
    '''
    A function which will create a map of p-values from a map of t-values and thresholds
    Start at the 100th percentile, and iterate down to the 0th percentile in increments of 1
    Upon each iteration, obtain the tthresh-value at each percentile
    Find the number of points in tthresh above the tthresh-value, to obtain a non-normalised p-value
    Normalise the p-value by dividing by the number of map elements, i.e. the size of t_to_p_map
    :param t_to_p_map: A 2D array of t-values
    :param t_thresh: A 1D array of t-values used to threshold, obtained from permutation test
    :return: A 2D array of p-values
    '''

    p_map = t_to_p_map.copy()
    ''' Calculate the p-values for the upper tail '''
    p_map[p_map < p_map.mean()] = np.NaN  # Set the lower tail to np.NaN
    variableThreshold = 100

    while variableThreshold > 0:
        # Upper tail p-values calculation
        pValue = sum(i > np.percentile(t_thresh, variableThreshold) for i in t_thresh)
        pValue = pValue / 7200  # Normalise
        p_map[p_map > np.percentile(t_thresh, variableThreshold)] = pValue
        variableThreshold = variableThreshold - 1  # Iterate top down

    return p_map


def pValueMap_neg_t(t_to_p_map, t_thresh):
    '''
    A function which will create a map of p-values from a map of t-values and thresholds
    :param t_to_p_map: A 2D array of t-values
    :param t_thresh: A 1D array of t-values used to threshold, obtained from permutation test
    :return: A 2D array of p-values
    '''

    # Start at the 100th percentile, and iterate down to the 0th percentile in increments of 1
    # Upon each iteration, obtain the tthresh-value at each percentile
    # Find the number of points in tthresh above the tthresh-value, to obtain a non-normalised p-value
    # Normalise the p-value by dividing by the number of map elements, i.e. the size of t_to_p_map

    ''' Calculate the p-values for the upper tail '''
    t_to_p_map[t_to_p_map > t_to_p_map.mean()] = np.NaN  # Set the lower tail to np.NaN
    variableThreshold = 0

    while variableThreshold < 100:
        # Upper tail p-values calculation
        pValue = sum(i < np.percentile(t_thresh, variableThreshold) for i in t_thresh)
        pValue = pValue / 7200  # Normalise
        t_to_p_map[t_to_p_map < np.percentile(t_thresh, variableThreshold)] = pValue
        variableThreshold = variableThreshold + 1  # Iterate top down

    return t_to_p_map


def p_value_contour_plot(t_map, t_thresh, percentile_array):
    '''
    Take in t-map, t-threshold distribution, and upper/lower tail. Will produce a map with significant contours 0.002,
    0.005, 0.01, 0.05.
    :param t_map: it a map of t-statistic
    :param t_thresh: t threshold distribution
    :param percentile_array: list of percentiles to be contoured
    :return: t-map with p-value contours of a specific tail
    '''

    # get t at percentiles of t_thresh
    critical_t_values = np.percentile(t_thresh, percentile_array)
    # contour labels of p-values
    #p_value_names = percentile_array/100
    clrs = ['magenta', 'lime', 'orange', 'red']
    CS = plt.contour(t_map[0], levels=critical_t_values, colors=clrs)
    ax = plt.gca()

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


def test_pymining():
    dataDirectory = r"../Data/OnlyProstateResults/AllFields"
    outputDirectory = r"../outputResults"
    # (meanVals, sdVals) = extractPatientSDVals(dataDirectory, allPatients.allPatients)
    rawPatientData = load_global_patients()
    enhancedDF = radial_mean_sd_for_patients(dataDirectory, rawPatientData.allPatients)

    print_volume_difference_details(enhancedDF)
    selected_patients, _, _ = partition_patient_data_with_outliers(enhancedDF, 0, 99,
                                                                   discriminator_fieldname="sd")  # 0-99.6 grabs 4 at large std dev # 99.73 std
    print_volume_difference_details(selected_patients)
    selected_patients, _, _ = partition_patient_data_with_outliers(enhancedDF, 0, 98.5,
                                                                   discriminator_fieldname="maxval")  # 0-99.6 grabs 4 at large std dev # 99.73 std
    print_volume_difference_details(selected_patients)
    selected_patients, _, _ = partition_patient_data_with_outliers(enhancedDF, 5, 100,
                                                                   discriminator_fieldname="DSC")  # 0-99.6 grabs 4 at large std dev # 99.73 std
    print_volume_difference_details(selected_patients)
    selected_patients, _, upper = partition_patient_data_with_outliers(enhancedDF, 2.5, 97.5,
                                                                       discriminator_fieldname="volumeContourDifference")  # 0-99.6 grabs 4 at large std dev # 99.73 std

    (global_neg_pvalue, global_pos_pvalue, neg_tthresh, pos_tthresh, t_value_map) = pyminingLocalField(
        selected_patients)

    # p_value_contour_plot(t_value_map, neg_tthresh, [0.2, 0.5, 50, 99])
    # plot_heat_map_np(t_value_map[0], 'maximum t-value map')
    p_value_contour_plot(t_value_map, pos_tthresh, [10,])


    # plot_sample_mean_and_sd_maps(selected_patients)

    # plot_tTest_data(global_neg_pvalue, global_pos_pvalue, neg_tthresh, pos_tthresh, t_value_map[0])


if __name__ == '__main__':
    # method_of_refining_data()
    test_pymining()
