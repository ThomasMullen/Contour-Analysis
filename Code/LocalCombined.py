#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 16:25:18 2018

@author: Tom & AJ
"""
import numpy as np
import pandas as pd
import seaborn as sns
import os
import pymining as pm
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


from AllPatients import recurrenceGroups
from LocalFilter import load_global_patients, radial_mean_sd_for_patients, partition_patient_data_with_outliers, \
    plot_heat_map, plotHist2, return_patient_sample_range, plot_scatter
sns.set()

# List the patient ID's of those who are contained in our ATLAS and have corrupted local maps & prothesis
#atlas = {'200806930','201010804', '201304169', '201100014', '201205737','201106120', '201204091', '200803943', '200901231', '200805565', '201101453', '200910818', '200811563','201014420'}
#corrupt = {'196708754','200801658','201201119','200911702','200701370','200700427','200610929','200606193','200600383','200511824'}
#corrupt16frac = {'200701370','200700427','200610929','200606193','200600383','200511824'}


def load_local_field_recurrence(global_df, dataDir = r'../Data/OnlyProstateResults/AllFields', corruptMaps = []):
    df = pd.DataFrame(global_df["patientList"])
    dfFiles = df.assign(file_path=lambda df: df["patientList"].map(lambda x: r'%s/%s.csv' % (dataDir, x)))

    masterDF = pd.DataFrame.empty
    fieldMaps = []
    # 140=200710358,55=200705181 corrupt patient ID an xi
    x = 0
    for f in dfFiles.file_path:
        if x not in corruptMaps:
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



def show_local_fields(global_df, dataDir = r'../Data/OnlyProstateResults/AllFields'):
    """ A function to print the each patients radial field for inspection """

    df = pd.DataFrame(global_df["patientList"])
    dfFiles = df.assign(file_path=lambda df: df["patientList"].map(lambda x: r'%s/%s.csv' % (dataDir, x)))

    fieldMaps = []

    x=0
    for f in dfFiles.file_path:
        fieldMaps.append(pd.read_csv(f, header=None))
        plot_heat_map(fieldMaps[x],-1,1,x)
        x = x + 1



def stack_local_fields(global_df, recurrence_label,  dataDir = r'../Data/OnlyProstateResults/AllFields'):
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
    patients_who_recur, patients_who_dont_recur = recurrenceGroups(selected_patients)
    (rec_fieldMaps, rec_label_array) = stack_local_fields(patients_who_recur, 1)
    (nonrec_fieldMaps, nonrec_label_array) = stack_local_fields(patients_who_dont_recur, 0)

    # Concatenate the two
    totalPatients = np.concatenate((rec_fieldMaps, nonrec_fieldMaps), axis=-1)

    # Label first 31 recurring as 1
    labels = np.concatenate((rec_label_array, nonrec_label_array))

    # ## Use scipy ttest, we should get the same result later
    # print(ttest_ind(patientMapRecurrenceContainer, patientMapNonRecurrenceContainer, equal_var=False, axis=-1))

    # ## Now use pymining to get a global p value. It should be similar to that from scipy
    globalp, tthresh = pm.permutationTest(totalPatients, labels)
    print(globalp)

    # Plot Threshold histogram

    # plt.hist(tthresh, 20, alpha=0.5, label='t-Test Threshold', normed=True, color='green')
    # plt.xlabel('t value')
    # plt.ylabel('Frequency')
    # plt.legend(loc='upper left')
    # # fig.savefig('/Users/Tom/Documents/University/ProstateCode/LocalAnalysis/T-testHist.png')
    # plt.show()

    # Plot Threshhold Map
    tThresh = sns.heatmap(pm.imagesTTest(totalPatients, labels)[0], center=0)
    tThresh.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$')

    plt.show()

def test_pymining():
    dataDirectory = r"../Data/OnlyProstateResults/AllFields"
    outputDirectory = r"../outputResults"
    # (meanVals, sdVals) = extractPatientSDVals(dataDirectory, allPatients.allPatients)
    rawPatientData = load_global_patients()
    enhancedDF = radial_mean_sd_for_patients(dataDirectory, rawPatientData.allPatients)
    patients_who_recur, patients_who_dont_recur = recurrenceGroups(enhancedDF)

    selected_patients, _, _ = partition_patient_data_with_outliers(enhancedDF, 0, 99) # 0-99.6 grabs 4 at large std dev # 99.73 std
    selected_patients, _, _ = partition_patient_data_with_outliers(selected_patients, 0, 98.5, discriminator_fieldname="maxval") # 0-99.6 grabs 4 at large std dev # 99.73 std
    selected_patients, _, _ = partition_patient_data_with_outliers(selected_patients, 5, 100, discriminator_fieldname="DSC") # 0-99.6 grabs 4 at large std dev # 99.73 std
    selected_patients, _, _ = partition_patient_data_with_outliers(selected_patients, 5, 95, discriminator_fieldname="volumeContourDifference") # 0-99.6 grabs 4 at large std dev # 99.73 std

    pyminingLocalField(selected_patients)


def test_local_field():
    dataDirectory = r"../Data/OnlyProstateResults/AllFields"
    outputDirectory = r"../outputResults"
    # (meanVals, sdVals) = extractPatientSDVals(dataDirectory, allPatients.allPatients)
    rawPatientData = load_global_patients()
    enhancedDF = radial_mean_sd_for_patients(dataDirectory, rawPatientData.allPatients)
    patients_who_recur, patients_who_dont_recur = recurrenceGroups(enhancedDF)

    DSCbins = 75 #[0,0.5,0.6,0.7,0.8,0.9,1]
    VolBins = 75 #[-40,-16,-10,-2.5,2.5,10,16,40]
    # =============================================================================
    #     Cut based on standard deviation of each map, eliminates global anomalies on maps
    # =============================================================================

            # sd plot before cuts
    plotHist2(patients_who_recur['sd'], 'red', 75,patients_who_dont_recur['sd'],'green',75, "Standard Deviation of radial difference, $\sigma_{\Delta R}$")
            # scatter plot of volume contour to auto-contour
    plot_scatter(enhancedDF,'r','upper left')

    selected_patients, lower_patients_outliers, upper_patients_outliers = partition_patient_data_with_outliers(enhancedDF, 0, 99) # 0-99.6 grabs 4 at large std dev # 99.73 std
    lower_patients_outliers.to_csv('%s/lower_patients_outliers_localMaps.csv' % outputDirectory)
    upper_patients_outliers.to_csv('%s/upper_patients_outliers_localMaps.csv' % outputDirectory)
    patients_who_recur, patients_who_dont_recur = recurrenceGroups(selected_patients)

#    plotHist2(patients_who_recur['volumeContourDifference'], 'r', VolBins, patients_who_dont_recur['volumeContourDifference'], 'g', VolBins, name="Volume difference between contour and auto-contour, $\Delta V$",legendPos="upper right")
#    plotHist2(patients_who_recur['DSC'], 'r', DSCbins, patients_who_dont_recur['DSC'], 'g', DSCbins, name="Dice coefficient",legendPos ="upper left")
#
        # sd plot after cuts
    plotHist2(patients_who_recur['sd'], 'red', 75,patients_who_dont_recur['sd'],'green',75, "Standard Deviation of radial difference, $\sigma_{\Delta R}$")

    # =============================================================================
    #     Cut based on the maximum value of each map, eliminates local map anomalies
    # =============================================================================
        # max value of map histogram before cuts
    plotHist2(patients_who_recur['maxval'], 'red', 75,patients_who_dont_recur['maxval'],'green',75, "Maximum value of radial difference, $max(\Delta R)$")

    selected_patients, lower_patients_outliers, upper_patients_outliers = partition_patient_data_with_outliers(selected_patients, 0, 98.5, discriminator_fieldname="maxval") # 0-99.6 grabs 4 at large std dev # 99.73 std
    lower_patients_outliers.to_csv('%s/lower_patients_outliers_maxval.csv' % outputDirectory)
    upper_patients_outliers.to_csv('%s/upper_patients_outliers_maxval.csv' % outputDirectory)
    patients_who_recur, patients_who_dont_recur = recurrenceGroups(selected_patients)
#    plotHist2(patients_who_recur['volumeContourDifference'], 'r', VolBins, patients_who_dont_recur['volumeContourDifference'], 'g', VolBins, name="Volume difference between contour and auto-contour, $\Delta V$",legendPos="upper right")
#    plotHist2(patients_who_recur['DSC'], 'r', DSCbins, patients_who_dont_recur['DSC'], 'g', DSCbins, name="Dice coefficient",legendPos ="upper left")

        # max value map after cuts
    plotHist2(patients_who_recur['maxval'], 'red', 75,patients_who_dont_recur['maxval'],'green',75, "Maximum value of radial difference, $max(\Delta R)$")

    # =============================================================================
    #     Remove patients based on DSC and Vdiff globally
    # =============================================================================

            # DSC before cuts
    plotHist2(patients_who_recur['DSC'], 'red', DSCbins, patients_who_dont_recur['DSC'], 'green', DSCbins, name="Dice coefficient",legendPos ="upper left")

    # DSC cut & local maps
    selected_patients, lower_patients_outliers, upper_patients_outliers = partition_patient_data_with_outliers(selected_patients, 5, 100, discriminator_fieldname="DSC") # 0-99.6 grabs 4 at large std dev # 99.73 std
    lower_patients_outliers.to_csv('%s/lower_patients_outliers_DSC.csv' % outputDirectory)
    upper_patients_outliers.to_csv('%s/upper_patients_outliers_DSC.csv' % outputDirectory)
    patients_who_recur, patients_who_dont_recur = recurrenceGroups(selected_patients)
#    plotHist2(patients_who_recur['volumeContourDifference'], 'r', VolBins, patients_who_dont_recur['volumeContourDifference'], 'g', VolBins, name="Volume difference between contour and auto-contour, $\Delta V$",legendPos="upper right")

        # DSC after cuts
    plotHist2(patients_who_recur['DSC'], 'red', DSCbins, patients_who_dont_recur['DSC'], 'green', DSCbins, name="Dice coefficient",legendPos ="upper left")


      # Vdiff before cuts
    plotHist2(patients_who_recur['volumeContourDifference'], 'red', VolBins, patients_who_dont_recur['volumeContourDifference'], 'green', VolBins, name="Volume difference between contour and auto-contour, $\Delta V$",legendPos="upper right")

    # Vdiff, DSC & local maps cut
    selected_patients, lower_patients_outliers, upper_patients_outliers = partition_patient_data_with_outliers(selected_patients, 5, 95, discriminator_fieldname="volumeContourDifference") # 0-99.6 grabs 4 at large std dev # 99.73 std
    lower_patients_outliers.to_csv('%s/lower_patients_outliers_Vdiff.csv' % outputDirectory)
    upper_patients_outliers.to_csv('%s/upper_patients_outliers_Vdiff.csv' % outputDirectory)
    patients_who_recur, patients_who_dont_recur = recurrenceGroups(selected_patients)

       # Vdiff after cuts
    plotHist2(patients_who_recur['volumeContourDifference'], 'red', VolBins, patients_who_dont_recur['volumeContourDifference'], 'green', VolBins, name="Volume difference between contour and auto-contour, $\Delta V$",legendPos="upper right")
        # scatter plot of volume contour to auto-contour
    plot_scatter(selected_patients,'r','upper left')

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

    plot_heat_map(meanMap1-meanMap2, -0.1, 0.1, 'mean map - patients_who_dont_recur')

    # printing local data fields
#    show_local_fields(patients_who_dont_recur,dataDirectory)
    return selected_patients


if __name__ == '__main__':
    # selected_patients = test_local_field()
    test_pymining()