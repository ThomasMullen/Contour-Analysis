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
from LocalFilter import load_global_patients, radial_mean_sd_for_patients, partition_patient_data_with_outliers, \
    plot_heat_map, plot_histogram_with_two_data_sets, plot_scatter, plot_histogram, plot_heat_map_np

sns.set()


# List the patient ID's of those who are contained in our ATLAS and have corrupted local maps & prothesis
# atlas = {'200806930','201010804', '201304169', '201100014', '201205737','201106120', '201204091', '200803943', '200901231', '200805565', '201101453', '200910818', '200811563','201014420'}
# corrupt = {'196708754','200801658','201201119','200911702','200701370','200700427','200610929','200606193','200600383','200511824'}
# corrupt16frac = {'200701370','200700427','200610929','200606193','200600383','200511824'}


def load_local_field_recurrence(global_df, dataDir=r'../Data/OnlyProstateResults/AllFields', corrupt_maps=[]):
    df = pd.DataFrame(global_df["patientList"])
    dfFiles = df.assign(file_path=lambda df: df["patientList"].map(lambda x: r'%s/%s.csv' % (dataDir, x)))

    masterDF = pd.DataFrame.empty
    fieldMaps = []
    # 140=200710358,55=200705181 corrupt patient ID an xi
    x = 0
    for f in dfFiles.file_path:
        if x not in corrupt_maps:
            fieldMaps.append(pd.read_csv(f, header=None))
        else:
            print(f)
        x = x + 1

    masterDF = pd.concat(fieldMaps)
    by_row_indexRec = masterDF.groupby(masterDF.index)
    meanRecurrence = by_row_indexRec.mean()
    varRecurrence = by_row_indexRec.var()
    stdRecurrence = by_row_indexRec.std()

    return (meanRecurrence, varRecurrence, stdRecurrence)


def show_local_fields(global_df, dataDir=r'../Data/OnlyProstateResults/AllFields'):
    """ A function to print the each patients radial field for inspection """

    df = pd.DataFrame(global_df["patientList"])
    dfFiles = df.assign(file_path=lambda df: df["patientList"].map(lambda x: r'%s/%s.csv' % (dataDir, x)))

    fieldMaps = []

    x = 0
    for f in dfFiles.file_path:
        fieldMaps.append(pd.read_csv(f, header=None))
        plot_heat_map(fieldMaps[x], -1, 1, x)
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

    # Now use pymining to get a global p value. It should be similar to that from scipy
    globalp, tthresh = pm.permutationTest(totalPatients, labels, 10)
    max_t_value_map = pm.imagesTTest(totalPatients, labels)[0]

    return globalp, tthresh, max_t_value_map


def plot_tTest_data(globalp, tthresh, max_tvalue_map):
    # Print Global p value
    print('Global p: %s' % globalp)

    # Plot Threshold histogram
    plot_histogram(tthresh, 'red', 20, "T-value")

    # Plot Threshhold Map
    plot_heat_map_np(max_tvalue_map, 'maximum t-value map')
    # tThresh = sns.heatmap(max_tvalue_map, center=0, cmap='RdBu')
    # tThresh.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$')
    # plt.show()

    # Plot Local P-values
    p_value_contour_plot(max_tvalue_map, tthresh)


def plot_sample_mean_and_sd_maps(selected_patients):
    dataDirectory = r"../Data/OnlyProstateResults/AllFields"
    patients_who_recur, patients_who_dont_recur = separate_by_recurrence(selected_patients)
    # (meanMap1, varMap, stdMap) = load_local_field_recurrence(selected_patients, dataDirectory)

    (meanMap1, varMap1, stdMap1) = load_local_field_recurrence(patients_who_recur, dataDirectory)
    plot_heat_map(meanMap1, -1, 1, 'mean map - patients_who_recur')
    plot_heat_map(varMap1, 0, 1, 'variance map - patients_who_recur')
    plot_heat_map(stdMap1, 0, 1, 'standard deviation map - patients_who_recur')

    (meanMap2, varMap2, stdMap2) = load_local_field_recurrence(patients_who_dont_recur, dataDirectory)
    plot_heat_map(meanMap2, -1, 1, 'mean map - patients_who_dont_recur')
    plot_heat_map(varMap2, 0, 1, 'variance map - patients_who_dont_recur')
    plot_heat_map(stdMap2, 0, 1, 'standard deviation map - patients_who_dont_recur')

    plot_heat_map(meanMap1 - meanMap2, -0.3, 0.3, 'Difference in mean map')
    # Var[X-Y] = Var[X]+Var[Y]
    # Standard deviation is the square root of the variance
    plot_heat_map(np.sqrt(varMap1 + varMap2), 0, 1.5, 'Difference in std map')


def pValueMap(tMaxMap, tthresh):
    variableThreshold = 100

    # Set map values
    tMaxMap[tMaxMap < tMaxMap.mean()] = np.NaN
    # lowTMap[lowTMap > lowTMap.mean()] = 0
    # while p-value above a certain threshold is < 0.05
    # while (sum(i > np.percentile(tthresh, variableThreshold) for i in tthresh)/7200) < 0.05:
    while variableThreshold > 0:
        # Set values less than this threshold to the p value
        pValue = sum(i > np.percentile(tthresh, variableThreshold) for i in tthresh) / 7200
        tMaxMap[tMaxMap > np.percentile(tthresh, variableThreshold)] = pValue
        variableThreshold = variableThreshold - 1

    return tMaxMap


def p_value_contour_plot(max_tvalue_map, tthresh):
    clrs = ['r', 'g', 'b']
    CS = plt.contour(pValueMap(max_tvalue_map, tthresh), levels=[0.01, 0.05, 0.1], colors=clrs)
    ax = plt.gca()
    ax.set_facecolor('white')
    # custom label names
    strs = ['p=0.01', 'p=0.05', 'p=0.1']
    fmt = {}
    for l, s in zip(CS.levels, strs):
        fmt[l] = s

    # plt.clabel(CS, fontsize=10, fmt=fmt)

    plt.show()


def test_pymining():
    dataDirectory = r"../Data/OnlyProstateResults/AllFields"
    outputDirectory = r"../outputResults"
    # (meanVals, sdVals) = extractPatientSDVals(dataDirectory, allPatients.allPatients)
    rawPatientData = load_global_patients()
    enhancedDF = radial_mean_sd_for_patients(dataDirectory, rawPatientData.allPatients)

    selected_patients, _, _ = partition_patient_data_with_outliers(enhancedDF, 0, 99,
                                                                   discriminator_fieldname="sd")  # 0-99.6 grabs 4 at large std dev # 99.73 std
    selected_patients, _, _ = partition_patient_data_with_outliers(selected_patients, 0, 98.5,
                                                                   discriminator_fieldname="maxval")  # 0-99.6 grabs 4 at large std dev # 99.73 std
    selected_patients, _, _ = partition_patient_data_with_outliers(selected_patients, 5, 100,
                                                                   discriminator_fieldname="DSC")  # 0-99.6 grabs 4 at large std dev # 99.73 std
    selected_patients, _, _ = partition_patient_data_with_outliers(selected_patients, 5, 95,
                                                                   discriminator_fieldname="volumeContourDifference")  # 0-99.6 grabs 4 at large std dev # 99.73 std
    (globalp, tthresh, max_t_value_map) = pyminingLocalField(selected_patients)
    plot_sample_mean_and_sd_maps(selected_patients)
    plot_tTest_data(globalp, tthresh, max_t_value_map)


def method_of_refining_data():
    dataDirectory = r"../Data/OnlyProstateResults/AllFields"
    outputDirectory = r"../outputResults"
    # (meanVals, sdVals) = extractPatientSDVals(dataDirectory, allPatients.allPatients)
    rawPatientData = load_global_patients()
    enhancedDF = radial_mean_sd_for_patients(dataDirectory, rawPatientData.allPatients)
    patients_who_recur, patients_who_dont_recur = separate_by_recurrence(enhancedDF)

    DSCbins = 75  # [0,0.5,0.6,0.7,0.8,0.9,1]
    VolBins = 75  # [-40,-16,-10,-2.5,2.5,10,16,40]
    # =============================================================================
    #     Cut based on standard deviation of each map, eliminates global anomalies on maps
    # =============================================================================

    # sd plot before cuts
    plot_histogram_with_two_data_sets(patients_who_recur['sd'], 'red', 75, patients_who_dont_recur['sd'], 'green', 75,
                                      "Standard Deviation of radial difference, $\sigma_{\Delta R}$")
    # scatter plot of volume contour to auto-contour
    plot_scatter(enhancedDF, 'r', 'upper left')

    # Apply SD cut
    selected_patients, lower_patients_outliers, upper_patients_outliers = partition_patient_data_with_outliers(
        enhancedDF, 0, 99)  # 0-99.6 grabs 4 at large std dev # 99.73 std
    # lower_patients_outliers.to_csv('%s/lower_patients_outliers_localMaps.csv' % outputDirectory)
    # upper_patients_outliers.to_csv('%s/upper_patients_outliers_localMaps.csv' % outputDirectory)

    # Separate By Recurrence
    patients_who_recur, patients_who_dont_recur = separate_by_recurrence(selected_patients)

    # Plot Histograms
    plot_histogram_with_two_data_sets(patients_who_recur['volumeContourDifference'], 'r', VolBins,
                                      patients_who_dont_recur['volumeContourDifference'], 'g', VolBins,
                                      name="Volume difference between "
                                           "contour and auto-contour, "
                                           "$\Delta V$", legendPos="upper "
                                                                   "right")
    plot_histogram_with_two_data_sets(patients_who_recur['DSC'], 'r', DSCbins, patients_who_dont_recur['DSC'], 'g',
                                      DSCbins,
                                      name="Dice coefficient", legendPos="upper left")

    # sd plot after cuts
    plot_histogram_with_two_data_sets(patients_who_recur['sd'], 'red', 75, patients_who_dont_recur['sd'], 'green', 75,
                                      "Standard Deviation of radial difference, $\sigma_{\Delta R}$")

    # =============================================================================
    #     Cut based on the maximum value of each map, eliminates local map anomalies
    # =============================================================================

    # max value of map histogram before cuts
    plot_histogram_with_two_data_sets(patients_who_recur['maxval'], 'red', 75, patients_who_dont_recur['maxval'],
                                      'green', 75,
                                      "Maximum value of radial difference, $max(\Delta R)$")

    # Apply Max Value Cut
    selected_patients, lower_patients_outliers, upper_patients_outliers = partition_patient_data_with_outliers(
        selected_patients, 0, 98.5, discriminator_fieldname="maxval")  # 0-99.6 grabs 4 at large std dev # 99.73 std
    # lower_patients_outliers.to_csv('%s/lower_patients_outliers_maxval.csv' % outputDirectory)
    # upper_patients_outliers.to_csv('%s/upper_patients_outliers_maxval.csv' % outputDirectory)

    # Separate by Recurrence
    patients_who_recur, patients_who_dont_recur = separate_by_recurrence(selected_patients)

    # Plot Histograms
    plot_histogram_with_two_data_sets(patients_who_recur['volumeContourDifference'], 'r', VolBins,
                                      patients_who_dont_recur['volumeContourDifference'], 'g', VolBins,
                                      legendPos="upper right", name="Volume difference between contour and "
                                                                    "auto-contour, $\Delta V$")
    plot_histogram_with_two_data_sets(patients_who_recur['DSC'], 'r', DSCbins, patients_who_dont_recur['DSC'], 'g',
                                      DSCbins, name="Dice coefficient", legendPos="upper left")

    # max value map after cuts
    plot_histogram_with_two_data_sets(patients_who_recur['maxval'], 'red', 75, patients_who_dont_recur['maxval'],
                                      'green', 75,
                                      "Maximum value of radial difference, $max(\Delta R)$")

    # =============================================================================
    #     Remove patients based on DSC and Vdiff globally
    # =============================================================================

    # DSC before cuts
    plot_histogram_with_two_data_sets(patients_who_recur['DSC'], 'red', DSCbins, patients_who_dont_recur['DSC'],
                                      'green', DSCbins,
                                      name="Dice coefficient", legendPos="upper left")

    # DSC cut & local maps
    selected_patients, lower_patients_outliers, upper_patients_outliers = partition_patient_data_with_outliers(
        selected_patients, 5, 100, discriminator_fieldname="DSC")  # 0-99.6 grabs 4 at large std dev # 99.73 std
    # lower_patients_outliers.to_csv('%s/lower_patients_outliers_DSC.csv' % outputDirectory)
    # upper_patients_outliers.to_csv('%s/upper_patients_outliers_DSC.csv' % outputDirectory)

    # Separate by Recurrence
    patients_who_recur, patients_who_dont_recur = separate_by_recurrence(selected_patients)

    # Plot Histograms
    plot_histogram_with_two_data_sets(patients_who_recur['volumeContourDifference'], 'r', VolBins,
                                      patients_who_dont_recur['volumeContourDifference'], 'g', VolBins,
                                      name="Volume difference between contour and auto-contour, $\Delta V$",
                                      legendPos="upper right")

    # DSC after cuts
    plot_histogram_with_two_data_sets(patients_who_recur['DSC'], 'red', DSCbins, patients_who_dont_recur['DSC'],
                                      'green', DSCbins,
                                      name="Dice coefficient", legendPos="upper left")

    # Vdiff before cuts
    plot_histogram_with_two_data_sets(patients_who_recur['volumeContourDifference'], 'red', VolBins,
                                      patients_who_dont_recur['volumeContourDifference'], 'green', VolBins,
                                      name="Volume difference between contour and auto-contour, $\Delta V$",
                                      legendPos="upper right")

    # Vdiff, DSC & local maps cut
    selected_patients, lower_patients_outliers, upper_patients_outliers = partition_patient_data_with_outliers(
        selected_patients, 5, 95,
        discriminator_fieldname="volumeContourDifference")  # 0-99.6 grabs 4 at large std dev # 99.73 std
    lower_patients_outliers.to_csv('%s/lower_patients_outliers_Vdiff.csv' % outputDirectory)
    upper_patients_outliers.to_csv('%s/upper_patients_outliers_Vdiff.csv' % outputDirectory)
    patients_who_recur, patients_who_dont_recur = separate_by_recurrence(selected_patients)

    # Vdiff after cuts
    plot_histogram_with_two_data_sets(patients_who_recur['volumeContourDifference'], 'red', VolBins,
                                      patients_who_dont_recur['volumeContourDifference'], 'green', VolBins,
                                      name="Volume difference between contour and auto-contour, $\Delta V$",
                                      legendPos="upper right")
    # scatter plot of volume contour to auto-contour
    plot_scatter(selected_patients, 'r', 'upper left')

    # =============================================================================
    # Maps with data removed
    # =============================================================================

    # Maps with corrupt patients fully cut out
    (meanMap1, varMap, stdMap) = load_local_field_recurrence(patients_who_recur, dataDirectory)
    plot_heat_map(meanMap1, -1, 1, 'mean map - patients_who_recur')
    plot_heat_map(varMap, 0, 1, 'variance map - patients_who_recur')
    plot_heat_map(stdMap, 0, 1, 'standard deviation map - patients_who_recur')

    (meanMap2, varMap, stdMap) = load_local_field_recurrence(patients_who_dont_recur, dataDirectory)
    plot_heat_map(meanMap2, -1, 1, 'mean map - patients_who_dont_recur')
    plot_heat_map(varMap, 0, 1, 'variance map - patients_who_dont_recur')
    plot_heat_map(stdMap, 0, 1, 'standard deviation map - patients_who_dont_recur')

    plot_heat_map(meanMap1 - meanMap2, -0.1, 0.1, 'mean map - patients_who_dont_recur')

    # printing local data fields
    #    show_local_fields(patients_who_dont_recur,dataDirectory)
    return


if __name__ == '__main__':
    # method_of_refining_data()
    test_pymining()
