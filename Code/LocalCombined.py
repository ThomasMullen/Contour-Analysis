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
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from AllPatients import separate_by_recurrence, separate_by_risk
from LocalFilter import load_global_patients, radial_mean_sd_for_patients, partition_patient_data_with_outliers
from plot_functions import plot_heat_map_np, plot_scatter, plot_histogram, plot_heat_map, show_local_fields, \
    test_on_single_map, triangulation_qa
from significance_test import wilcoxon_test_statistic, mann_whitney_test_statistic, pymining_t_test, \
    t_map_with_thresholds, test_superimpose, \
    global_statistical_analysis, map_with_thresholds, non_parametric_permutation_test

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


def print_volume_difference_details(patient_database):
    """
    :param patient_database: is a dataframe that contains the patients global variables
    :return: prints volume difference statistics for collectiver patient, then patients with recurrence and patients /
    without
    """
    print(patient_database["volumeContourDifference"].describe())
    patients_who_recur, patients_who_dont_recur = separate_by_recurrence(patient_database)
    print(patients_who_recur["volumeContourDifference"].describe())
    print(patients_who_dont_recur["volumeContourDifference"].describe())


def get_corrupt_patients(all_patients_df, data_directory):
    """
    Reads in filenames from data directory, exstracts the patient ids and removes them from the dataset
    :param all_patients_df: global dataframe
    :param data_directory: data directory that has the rogue patient names
    :return: returns clean dataset
    """
    # Create list of file path
    file_list = [f for f in listdir(data_directory) if isfile(join(data_directory, f))]
    # parse filename and extract id
    patient_id = list(map(lambda x: re.split('[_ .]', x)[1], file_list))
    # print(patient_id)
    clean_patients_ct_scans = all_patients_df[~all_patients_df['patientList'].isin(patient_id)]
    return clean_patients_ct_scans


def statistical_cuts(patient_database, data_directory=r"../Data/OnlyProstateResults/AllFields"):
    """
    Produce histogram plots and within has partition cuts base on patients max delta r, std, dsc, and vol diff
    :param patient_database: global patient df
    :param dataDirectory: Directory of the patient radial maps
    :return: clean dataset
    """
    # Plotting histograms before any cuts
    plot_histogram(patient_database['sd'], 'red', 50, name="Standard Deviations of patients map")
    plot_histogram(patient_database['maxval'], 'lime', 50, name="Maximum value of patients map")
    plot_histogram(patient_database['DSC'], 'green', 50, name="Dice of patients map")
    plot_histogram(patient_database['volumeContourDifference'], 'blue', 50, name="Volume difference of patients map")

    # Look for poor triangulation of patient maps
    # SD of patient map
    _, _, upper_bound = partition_patient_data_with_outliers(patient_database, 0, 5, "sd")
    clean_patients = patient_database[~patient_database['patientList'].isin(upper_bound['patientList'])]
    print(clean_patients.shape)

    # Max value of Patient
    _, lower_bound, upper_bound = partition_patient_data_with_outliers(patient_database, -5, 5, "maxval")
    clean_patients = clean_patients[~clean_patients['patientList'].isin(upper_bound['patientList'])]
    clean_patients = clean_patients[~clean_patients['patientList'].isin(lower_bound['patientList'])]
    print(clean_patients.shape)

    # Dice Coefficient
    _, lower_bound, _ = partition_patient_data_with_outliers(patient_database, 0.7, 0.999, "DSC")
    clean_patients = clean_patients[~clean_patients['patientList'].isin(lower_bound['patientList'])]
    print(clean_patients.shape)
    return clean_patients


def read_and_return_patient_stats(calculated_csv_name="All_patient_data_no_NA",
                                  dataDirectory=r"../Data/Deep_learning_results/deltaRMaps"):
    '''
    Loads up and concatenates table then writes a single data frame with mean, st and max r value for each patient
    map added
    :param dataDirectory: The directory that stores all the patient fields
    :param calculated_csv_name: name of the new csv
    :return:
    '''
    rawPatientData = load_global_patients()
    enhancedDF = radial_mean_sd_for_patients(dataDirectory, rawPatientData.allPatients)
    enhancedDF.to_csv('../Data/Deep_learning_results/' + calculated_csv_name + '.csv')


def cuts_from_ct_scans(global_df):
    # removes patients with corrupt scans
    # rogue_ct_scans_dir_16 = r"../Corrupt_CT_Scans/16Fractions/"
    # rogue_ct_scans_dir_19 = r"../Corrupt_CT_Scans/19Fractions/"
    # rouge_ct_scans_dir_16_old = r"../Corrupt_CT_Scans/16Fractions_old/"
    rouge_ct_scans_dir_comb = r"../Corrupt_CT_Scans/Corrupt/"

    # Removes the selected corrupt ct scan patients from enhancedDF
    global_df = get_corrupt_patients(global_df, rouge_ct_scans_dir_comb)
    return global_df


def test_analysis_function(enhancedDF):
    # Low and intermediate risk patients
    low_and_intermediate_risk_patients = enhancedDF[~enhancedDF['risk'].isin(['High', 'high', 'int/high'])]
    _, p_map_mwu = mann_whitney_test_statistic(low_and_intermediate_risk_patients)
    map_with_thresholds(p_map_mwu)
    global_statistical_analysis(low_and_intermediate_risk_patients)

    # High risk patients
    high_risk_patients = enhancedDF[enhancedDF['risk'].isin(['High', 'high', 'int/high'])]
    _, p_map_mwu = mann_whitney_test_statistic(high_risk_patients)
    map_with_thresholds(p_map_mwu)
    global_statistical_analysis(high_risk_patients)


def test_survival_analysis(patient_data_base):
    # Remove patient that have not time event
    # patient_data_base = patient_data_base[patient_data_base.recurrenceTime != '']
    #
    # T = patient_data_base["timeToEvent"]
    # E = patient_data_base["recurrence"]
    #
    kmf = KaplanMeierFitter()
    # kmf.fit(T, event_observed=E)
    # kmf.plot()

    dice_median = patient_data_base["DSC"].median()

    # dsc = patient_data_base['DSC']
    # ix = (dsc <= dice_median)
    #
    # kmf.fit(T[~ix], E[~ix], label='Lower Half Dice Patients')
    # kmf.fit(T[ix], E[ix], label='Upper Half Dice Patients')
    # kmf.plot()

    upper_group = patient_data_base[patient_data_base.DSC >= dice_median]
    lower_group = patient_data_base[patient_data_base.DSC <= dice_median]
    T1 = upper_group["timeToEvent"]
    E1 = upper_group["recurrence"]
    T2 = lower_group["timeToEvent"]
    E2 = lower_group["recurrence"]

    ax = plt.subplot(111)

    kmf.fit(T1, event_observed=E1, label=['Upper'], timeline=4)
    kmf.survival_function_
    kmf.confidence_interval_
    kmf.median_
    kmf.survival_function_.plot(ax=ax)
    kmf.fit(T2, event_observed=E2, label=['Lower'], timeline=4)
    kmf.survival_function_
    kmf.confidence_interval_
    kmf.median_
    kmf.survival_function_.plot(ax=ax)
    plt.title('Lifespans of different tumor DNA profile')
    plt.show()
    return


if __name__ == '__main__':
    # read_and_return_patient_stats()
    dataDirectory = r"../Data/Deep_learning_results/deltaRMaps"
    enhancedDF = pd.read_csv(r'../Data/Deep_learning_results/All_patient_data_no_time_event_NA.csv')
    enhancedDF = cuts_from_ct_scans(enhancedDF)
    # test_survival_analysis(enhancedDF)
    test_analysis_function(enhancedDF)
    # triangulation_qa()
    # test_on_single_map()
