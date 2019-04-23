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
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf')


from lifelines import KaplanMeierFitter, NelsonAalenFitter, CoxPHFitter
from AllPatients import separate_by_recurrence, separate_by_risk
from LocalFilter import load_global_patients, radial_mean_sd_for_patients, partition_patient_data_with_outliers
from plot_functions import plot_heat_map_np, plot_scatter, plot_histogram, plot_heat_map, show_local_fields, \
    test_on_single_map, triangulation_qa, load_map, create_polar_axis
from significance_test import cph_global_test, mann_whitney_test_statistic, pymining_t_test, \
    t_map_with_thresholds, test_superimpose, sample_normality_test, \
    global_statistical_analysis, map_with_thresholds, non_parametric_permutation_test, stack_local_fields, normality_map
from DataFormatting import data_frame_to_XDR

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

    print("\nGlobal analysis: low & intermediate risk patients")
    low_and_intermediate_risk_patients = enhancedDF[~enhancedDF['risk'].isin(['High', 'high', 'int/high'])]
    _, p_map_mwu = mann_whitney_test_statistic(low_and_intermediate_risk_patients)
    map_with_thresholds(p_map_mwu)
    global_statistical_analysis(low_and_intermediate_risk_patients)

    # High risk patients
    print("\nGlobal analysis: high risk patients")
    high_risk_patients = enhancedDF[enhancedDF['risk'].isin(['High', 'high', 'int/high'])]
    _, p_map_mwu = mann_whitney_test_statistic(high_risk_patients)
    map_with_thresholds(p_map_mwu)
    global_statistical_analysis(high_risk_patients)


def survival_analysis_dsc(patient_data_base, category='DSC'):
    # Remove patient that have not time event
    # patient_data_base = patient_data_base[patient_data_base.recurrenceTime != '']

    # Survival function groupby DSC
    category_groups = patient_data_base[category].quantile([.25, .5, .75, 1.0])
    dsc = patient_data_base[category]

    ix_1 = (dsc <= category_groups[0.25])
    ix_2 = (dsc > category_groups[0.25]) & (dsc < category_groups[0.5])
    ix_3 = (dsc > category_groups[0.5]) & (dsc < category_groups[0.75])
    ix_4 = (dsc > category_groups[0.75])

    # fit the model for 1st cohort
    kmf = KaplanMeierFitter()
    T = patient_data_base["timeToEvent"]
    E = patient_data_base["recurrence"]
    kmf.fit(T[ix_1], E[ix_1], label="First Quartile")
    a1 = kmf.plot()
    # fit the model for 2nd cohort
    kmf.fit(T[ix_2], E[ix_2], label="Second Quartile")
    kmf.plot(ax=a1)
    # fit the model for 3rd cohort
    kmf.fit(T[ix_3], E[ix_3], label='Third Quartile')
    kmf.plot(ax=a1)
    # fit the model for 4th cohort
    kmf.fit(T[ix_4], E[ix_4], label='Forth Quartile')
    kmf.plot(ax=a1)
    plt.show()
    return


def survival_analysis_fractions(patient_df):
    T = patient_df["timeToEvent"]
    E = patient_df["recurrence"]
    kmf = KaplanMeierFitter()
    fraction_19 = patient_df["fractions"]
    ix_19 = fraction_19 == 19
    #
    kmf.fit(T[ix_19], E[ix_19], label="19 Fractions")
    a2 = kmf.plot()
    # fit the model for 2nd cohort
    kmf.fit(T[~ix_19], E[~ix_19], label="16 Fractions")
    kmf.plot(ax=a2)
    plt.show()


def file_conversion_test(patients):
    _, _ = stack_local_fields(patients, 0)


def numerate_categorical_data(clean_patient_data):
    """
    A function to numerate the categorical data ready for our Cox regression

    :param clean_patient_data: A data frame of patients
    :return: a data frame of patients with parameters in categorical format
    """

    # Numerate categorical data
    clean_patient_data['fractions'] = clean_patient_data['fractions'].apply(lambda x: 0 if x == 16 else 1)
    clean_patient_data['grade'] = clean_patient_data['grade'].apply(lambda x: 0 if x <= 6 else (1 if x == 7 else 2))
    clean_patient_data['risk'] = clean_patient_data['risk'].apply(
        lambda x: 0 if (x == 'Low' or x == 'low') else (2 if (x == 'High' or x == 'high' or x == 'int/high') else 1))

    return clean_patient_data


def add_covariate_data(clean_patient_data, new_data_file='patientAges',  covariate_names=['patientList', 'age']):
    '''
    This function reads in patient data with new variable and attaches it to the main patient data frame and converts
    categorical data into numerical values. It also removes the patient ID so its instantly renewable.
    :param clean_patient_data: The main patient database
    :param new_data_file: The datafile name in deep learning folder
    :param covariate_names: Fist include patient list then the new variables you wish to add
    :return: Returns the data with new variables and no patient ID
    '''

    patient_patient_data = pd.read_csv(r'../Data/Deep_learning_results/' + new_data_file + '.csv')
    clean_patient_data = pd.merge(clean_patient_data, patient_patient_data[covariate_names], how='inner',
                                  on='patientList')

    # Drop patient ID
    patientID_list = clean_patient_data['patientList']
    # print(clean_patient_data[['fractions', 'grade', 'risk']])
    # print(clean_patient_data.head(n=1))
    return clean_patient_data, patientID_list


def test_plot_subplot():
    # map1 = pd.read_csv(r'../Data/Deep_learning_results/covariate_maps/withGrade/csv_results/fractions.csv')
    # map2 = pd.read_csv(r'../Data/Deep_learning_results/covariate_maps/withGrade/csv_results/grade.csv')
    # map3 = pd.read_csv(r'../Data/Deep_learning_results/covariate_maps/withGrade/csv_results/risk.csv')
    # map4 = pd.read_csv(r'../Data/Deep_learning_results/covariate_maps/withGrade/csv_results/autoVolume.csv')
    # map5 = pd.read_csv(r'../Data/Deep_learning_results/covariate_maps/withGrade/csv_results/age.csv')
    map6 = pd.read_csv(r'../Data/Deep_learning_results/covariate_maps/withGrade/csv_results/deltaR.csv')

    # sig_map1 = pd.read_csv(r'../Data/Deep_learning_results/covariate_maps/withGrade/csv_results_sig/fractions_sig.csv')
    # sig_map2 = pd.read_csv(r'../Data/Deep_learning_results/covariate_maps/withGrade/csv_results_sig/grade_sig.csv')
    # sig_map3 = pd.read_csv(r'../Data/Deep_learning_results/covariate_maps/withGrade/csv_results_sig/risk_sig.csv')
    # sig_map4 = pd.read_csv(r'../Data/Deep_learning_results/covariate_maps/withGrade/csv_results_sig/autoVolume_sig.csv')
    # sig_map5 = pd.read_csv(r'../Data/Deep_learning_results/covariate_maps/withGrade/csv_results_sig/age_sig.csv')
    # sig_map6 = pd.read_csv(r'../Data/Deep_learning_results/covariate_maps/withGrade/csv_results_sig/deltaR_sig.csv')


    # f, axes = plt.subplots(3, 2, sharex='col', sharey='row')
    # heat_map1 = sns.heatmap(map1.values, center=0, ax=axes[0,0], cmap='RdBu', vmin=0, vmax=2,
    #             cbar=False)
    # heat_map1.set_xlabel(''); heat_map1.set_ylabel('')
    # heat_map2 = sns.heatmap(map2.values, center=0, ax=axes[0,1], cmap='RdBu', vmin=0, vmax=2,
    #             cbar=False)
    # heat_map2.set_xlabel(''); heat_map2.set_ylabel('')
    #
    # heat_map3 = sns.heatmap(map3.values, center=0, ax=axes[1,0], cmap='RdBu', vmin=0, vmax=2,
    #             cbar=False)
    # heat_map3.set_xlabel(''); heat_map3.set_ylabel('')
    # heat_map4 = sns.heatmap(map4.values, center=0, ax=axes[1,1], cmap='RdBu', vmin=0, vmax=2,
    #             cbar=False)
    # heat_map4.set_xlabel(''); heat_map4.set_ylabel('')
    #
    # heat_map5 = sns.heatmap(map5.values, center=0, ax=axes[2,0], cmap='RdBu', vmin=0, vmax=2,
    #             cbar=False)
    # heat_map5.set_xlabel(''); heat_map5.set_ylabel('')
    # # heat_map6 = sns.heatmap(map6.values, center=0, ax=axes[2,1], cmap='RdBu', vmin=0, vmax=2,
    # #             cbar=False)
    # # heat_map6.set_xlabel(''); heat_map6.set_ylabel('')
    #
    # # Fine-tune figure; make subplots farther from each other.
    # f.subplots_adjust(hspace=0.3)
    #
    # plt.tight_layout()
    # plt.show(block=True)

    # Thresholds without grade
    # map_with_thresholds(map1, [np.exp(0), np.exp(1.8348171773659745)], False)
    # map_with_thresholds(map3, [np.exp(-0.69973541041431797), np.exp(1.4003677492629816)], False)
    # map_with_thresholds(map4, [np.exp(-0.015974724319552071), np.exp(1.1236690064758252)], False)
    # map_with_thresholds(map5, [np.exp(-0.0072990457219818941), np.exp(0.081051212841960679)], False)
    # map_with_thresholds(map6, [np.exp(-1.8004828254133396), np.exp(2.3135563039910414)], False)

    # Thresholds with grade
    # map_with_thresholds(map1, [np.exp(-0.472801192), np.exp(1.880958152)], False)
    # map_with_thresholds(map2, [np.exp(-0.481561145), np.exp(1.51658389)], False)
    # map_with_thresholds(map3, [np.exp(-0.615623492), np.exp(1.393389023)], False)
    # map_with_thresholds(map4, [np.exp(-0.01628528), np.exp(1.163254349)], False)
    # map_with_thresholds(map5, [np.exp(-0.011873742), np.exp(0.09111272)], False)
    map_with_thresholds(map6, [np.exp(-1.727245154), np.exp(2.384893813)], False)
    # map_with_thresholds(sig_map1, [0.05], False)
    # map_with_thresholds(sig_map2, [0.05], False)
    # map_with_thresholds(sig_map3, [0.05], False)
    # map_with_thresholds(sig_map4, [0.05], False)
    # map_with_thresholds(sig_map5, [0.05], False)
    # map_with_thresholds(sig_map6, [0.05], False)

    # data = pd.read_csv(r'../Data/Deep_learning_results/per_vox_cox.csv')
    # g = sns.PairGrid(data, vars=['fractions', 'risk', 'autoContourVolume', 'age'],
    #                  hue='recurrence', palette='RdBu_r')
    # g.map(plt.scatter, alpha=0.8)
    # g.add_legend();
    # plt.show(block=True)


def clean_data(data):
    """
    A function to remove all errors from the data, and replace categorical data
    Errors include: duplicates, NA values, un-wanted columns, CT-identified patients to remove

    :param data: A data frame of all patients
    :return: A data frame of cleaned patients
    """

    cleaned_data = data.copy()
    cleaned_data = cleaned_data.drop_duplicates(subset='patientList')
    cleaned_data = cuts_from_ct_scans(cleaned_data)
    cleaned_data = cleaned_data.drop(['patientNumber', 'patientList', 'recurrence_4years', 'sdDoseDiff', 'volumeContour', 'volumeContourDifference', 'DSC', 'volumeRatio'], axis=1)
    # cleaned_data = cleaned_data.drop(['patientList'], axis=1)
    # field, _ = stack_local_fields(clean_data, 1)
    # del_r = field[17][80][:]
    # clean_data['delta_r'] = del_r
    cleaned_data = numerate_categorical_data(cleaned_data)

    return cleaned_data

def test_categorical_map():
    initial_map = pd.read_csv(r'/Users/Tom/PycharmProjects/Contour-Analysis/Data/Deep_learning_results/deltaRMaps/196703818.csv', header=None).values
    final_map = list(map(lambda y: list(map(lambda x: 0 if x <= -2 else (2 if x >= 2 else 1), y)), initial_map))
    new = final_map.reshape(60,120)
    print(new)
    # print(final_map.shape)
    return

if __name__ == '__main__':
    # read_and_return_patient_stats()
    # dataDirectory = r"../Data/Deep_learning_results/deltaRMaps"
    enhancedDF = pd.read_csv(r'../Data/Deep_learning_results/global_results/all_patients.csv')
    enhancedDF["delta_r"] = stack_local_fields(enhancedDF, 1)[0][17][80][:]
    clean_data = clean_data(enhancedDF)

    cph_global_test(clean_data) # Produce global cox table


    # clean_data.to_csv("../Data/Deep_learning_results/global_results/all_patients_cleaned.csv", index=False)
    # survival_analysis_fractions(enhancedDF)

    # DSC = load_map(r'../Data/Deep_learning_results/covariate_maps/', 'DSC')
    # plot_heat_map(DSC, 1, 1.5)
    # test_plot_subplot()
    # x=0

    # clean_data_and_add_covariates()
    # file_conversion_test(enhancedDF)
    # test_analysis_function(enhancedDF)
