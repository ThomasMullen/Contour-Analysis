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
from lifelines.statistics import logrank_test
from AllPatients import separate_by_recurrence, separate_by_risk
from LocalFilter import load_global_patients, radial_mean_sd_for_patients, partition_patient_data_with_outliers
from plot_functions import plot_heat_map_np, plot_scatter, plot_histogram, plot_heat_map, show_local_fields, \
    test_on_single_map, triangulation_qa, load_map, create_polar_axis
from significance_test import cph_global_test, mann_whitney_test_statistic, pymining_t_test, \
    t_map_with_thresholds, test_superimpose, sample_normality_test, cph_produce_map, \
    global_statistical_analysis, map_with_thresholds, non_parametric_permutation_test, stack_local_fields, normality_map
from DataFormatting import data_frame_to_XDR

sns.set()


def make_average_field(global_df, dataDir=r'../Data//Deep_learning_results/deltaRMaps'):
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
    skew_field = by_row_indexRec.skew()
    # rms_map = np.sqrt(np.mean(np.square(by_row_indexRec.values)))

    return mean_field, variance_field, std_field, skew_field


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
    E = patient_data_base["recurrence_outcome"]
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
    E = patient_df["recurrence_outcome"]
    kmf = KaplanMeierFitter()
    fraction_19 = patient_df["fractions"]
    ix_19 = fraction_19 == 1
    #
    kmf.fit(T[ix_19], E[ix_19], label="19 Fractions")
    a2 = kmf.plot()
    # fit the model for 2nd cohort
    kmf.fit(T[~ix_19], E[~ix_19], label="16 Fractions")
    kmf.plot(ax=a2)
    plt.show()


def survival_analysis_dsc(patient_data_base, category='DSC'):

    category_groups = patient_data_base[category].quantile([.25, .5, .75, 1.0])
    dsc = patient_data_base[category]

    ix_1 = (dsc <= category_groups[0.25])
    ix_2 = (dsc > category_groups[0.25]) & (dsc < category_groups[0.5])
    ix_3 = (dsc > category_groups[0.5]) & (dsc < category_groups[0.75])
    ix_4 = (dsc > category_groups[0.75])

    # fit the model for 1st cohort
    kmf = KaplanMeierFitter()
    T = patient_data_base["timeToEvent"]
    E = patient_data_base["recurrence_outcome"]
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


def survival_analysis(patient_data_base, category, save_name):
    """
    A function to produce KM plots for various categorical data (age, grade and manVol)

    :param patient_data_base:
    :param category:
    :return:
    """

    category_data = patient_data_base[category]

    if category == "age":
        ix_1 = (category_data < 65)
        ix_2 = (category_data >= 65) & (category_data <= 75)
        ix_3 = (category_data > 75)
        label_1 = "< 65 years"
        label_2 = "65 - 75 years"
        label_3 = "> 75 years"
    elif category == "grade":
        ix_1 = (category_data == 3)
        ix_2 = (category_data == 2)
        ix_3 = (category_data == 1)
        ix_4 = (category_data == 0)
        label_1 = "Gleason = 6"
        label_2 = "Gleason = 7"
        label_3 = "Gleason = 8"
        label_4 = "Gleason = 9-10"
    elif category == "manVol":
        category_groups = patient_data_base[category].quantile([.25, .5, .75, 1.0])
        ix_1 = (category_data <= category_groups[0.25])
        ix_2 = (category_data > category_groups[0.25]) & (category_data <= category_groups[0.5])
        ix_3 = (category_data > category_groups[0.5]) & (category_data <= category_groups[0.75])
        ix_4 = (category_data > category_groups[0.75]) & (category_data <= category_groups[1.0])
        label_1 = "< %.0f mm$^3$" % (category_groups[0.25])
        label_2 = "%.0f - %.0f cm$^3$" % (category_groups[0.25], category_groups[0.5])
        label_3 = "%.0f - %.0f cm$^3$" % (category_groups[0.5], category_groups[0.75])
        label_4 = "> %.0f mm$^3$" % (category_groups[0.75])

    else:
        # category_groups = patient_data_base[category].quantile([.1, .5, .9, 1.0])
        # ix_1 = (category_data <= category_groups[0.1])
        # ix_2 = (category_data > category_groups[0.9]) & (category_data <= category_groups[1.0])
        # label_1 = "< %.2f mm" % (category_groups[0.1])
        # label_2 = "> %.2f mm" % (category_groups[0.9])
        category_groups = patient_data_base[category].quantile([.5, 1.0])
        ix_1 = (category_data <= category_groups[0.5])
        ix_2 = (category_data > category_groups[0.5]) & (category_data <= category_groups[1.0])
        label_1 = "< %.2f cm" % (category_groups[0.5])
        label_2 = "> %.2f cm" % (category_groups[0.5])

    # fit the model for 1st cohort
    # ax.plot(x, y, ...)
    # further stuff

    kmf = KaplanMeierFitter()
    T = patient_data_base["eventTime"]
    E = patient_data_base["outcome"]
    if (category != "mean_lower_region") and (category != "mean_upper_region"):
        kmf.fit(T[ix_1], E[ix_1], label=label_1)
        a1 = kmf.plot()
        # fit the model for 2nd cohort
        kmf.fit(T[ix_2], E[ix_2], label=label_2)
        kmf.plot(ax=a1)
        # fit the model for 3rd cohort
        kmf.fit(T[ix_3], E[ix_3], label=label_3)
        kmf.plot(ax=a1)

        if category != "age":
            # fit the model for 4th cohort
            kmf.fit(T[ix_4], E[ix_4], label=label_4)
            kmf.plot(ax=a1)
    else:
        kmf.fit(T[ix_1], E[ix_1], label=label_1)
        a1 = kmf.plot()
        # fit the model for 2nd cohort
        kmf.fit(T[ix_2], E[ix_2], label=label_2)
        kmf.plot(ax=a1)

    plt.style.use('seaborn-ticks')
    sns.set_context("talk")
    plt.rcParams['font.family'] = 'times'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14

    plt.xlim(0, 8.6)
    plt.ylim(0, 1.05)
    plt.xlabel('Time [years]')
    plt.ylabel('Survival function, S(t)')
    plt.legend(loc="lower left", frameon=True, framealpha=1)
    plt.grid(True)
    # ax.set_xticks()
    # ax.set_yticks()
    # # ax.grid()
    # # ax.axis('equal')
    plt.savefig(save_name, bbox_inches='tight')
    plt.show()
    if (category == "mean_lower_region") or (category == "mean_upper_region"):
        results = logrank_test(T[ix_1], T[ix_2], E[ix_1], E[ix_2])
    elif category != "age":
        results = logrank_test(T[ix_1], T[ix_4], E[ix_1], E[ix_4])
    else:
        results = logrank_test(T[ix_1], T[ix_3], E[ix_1], E[ix_3])
    results.print_summary()


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
    clean_patient_data['grade'] = clean_patient_data['grade'].apply(lambda x: 0 if x <= 6 else
    (1 if x == 7 else (2 if x == 8 else 3)))
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


def test_plot_subplot(patient_df):

    covariates = ["deltaR", "age", "grade_6_to_910", "grade_7_to_910", "grade_8_to_910", "manVol"]

    maps = [pd.read_csv(r'/Users/Alex Jenkins/PycharmProjects/Contour-Analysis/Results/CSV/%s_hazardOnly.csv' % x,
                        header=0) for x in covariates]

    #
    # new_patient_df = regional_investigation(maps[0], patient_df, "SV", np.exp(-0.369794153))
    #
    # survival_analysis(new_patient_df, "mean_upper_region", "upper_KM")
    # survival_analysis(new_patient_df, "mean_upper_region", "upper_KM")
    # print("lower_KM")
    # survival_analysis(new_patient_df, "mean_lower_region", "lower_KM")
    # print("lower_KM")
    # survival_analysis(new_patient_df, "mean_upper_region", "upper_KM")
    # print("upper_KM")



    # np.exp(-0.018092986), np.exp(0.015032935)
    # HR_upper = maps[0].mask(maps[0] < 1, 0)
    # HR_lower = maps[0].mask(maps[0] > 1, 1)
    # map_with_thresholds(HR_upper, "Hazard ratio [mm$^{-1}$]", "solid", 0.1, 1.2, 1, ['lime', 'magenta'],
    #                     [np.exp(0.015032935), np.exp(0.75875185)], False)
    # map_with_thresholds(maps[0], "Hazard ratio [mm$^{-1}$]", ["dashed", "solid"], 0.1, 1.2, 0.9, ['magenta', 'magenta'],
    #                     [np.exp(-0.369794153), np.exp(0.75875185)], False)
    #

    # np.exp(-0.369794153),np.exp(0.75875185)
    # map_with_thresholds(maps[0], 0.5, 1, 0.5, ['red', 'lime'], [np.exp(-0.369794153),
    #                                            np.exp(-0.01766508)], False)

    # # mean_field, variance_field, std_field, skew_field = make_average_field(patient_df)
    # map_with_thresholds(maps[0], "Hazard ratio [mm$-1$]", 0.1, 1.2, 0.8, ['red', 'green', 'lime', 'magenta'],
    #                     [np.exp(-6.401438146), np.exp(-0.130110886), np.exp(2), np.exp(2)], False)

    # map_with_thresholds(maps[1], "Hazard ratio [year$^-1$]", 1, 0.5, 1, ['red', 'green', 'lime', 'magenta'], [np.exp(-6.401438146),
    #                                                                                   np.exp(-0.130110886),
    #                                                                                   np.exp(2),
    #                                                                                   np.exp(2)], False)
    #
    # map_with_thresholds(maps[5], "Hazard ratio [cm$^-3$]", 1, 1.2, 0.8, ['red', 'green', 'lime', 'magenta'],
    #                     [], False)

    map_with_thresholds(maps[4], "Hazard ratio [rel. to Gleason scores 9&10]", "dashed", 1, 1.2, 0.8, ['magenta'],
                        [np.exp(-0.288813013)], False)

    map_with_thresholds(maps[3], "Hazard ratio [rel. to Gleason scores 9&10]", "dashed", 1, 1.2, 0.8, ['magenta'],
                        [np.exp(-0.029020724)], False)

    # plot_heat_map(maps[1], 0.9, 1.1, "Hazard ratio [year $^{-1}$]")
    # plot_heat_map(maps[3], 0.9, 1.1, "Hazard ratio [rel. to Gleason scores 9&10]")
    # plot_heat_map(maps[4], 0.9, 1.1, "Hazard ratio [rel. to Gleason scores 9&10]")
    # plot_heat_map(maps[5], 0.9, 1.1, "Hazard ratio [cm$^{-3}$]")


def regional_investigation(HR_map, all_patients, region_of_interest, upper_hazard, dataDir=r'../Data//Deep_learning_results/deltaRMaps'):
    """
    A function that will extract statistics in a specified region of HR maps for all patients.
    - Find the median value of delta R for each patient in the region.

    :param HR_map: A hazard ratio map
    :param all_patients: A data frame featuring all information for all patients
    :param region_of_interest: A string containing the region you wish to analyse
    :param upper_hazard: The upper HR value (>1) on the map, will find all voxel coordinates at values greater than this
    :return: A list of median values in the specified region for all patients (use for KM plots).
    """

    # Import mask of the ROI
    ROI_mask = pd.read_csv(r'/Users/Alex Jenkins/PycharmProjects/Contour-Analysis/Results/CSV/deltaR_%s_HR.csv'
                           % region_of_interest, header=0)

    upper_ROI_mask = ROI_mask.mask(ROI_mask > upper_hazard, 0)
    upper_ROI_mask = upper_ROI_mask.mask(upper_ROI_mask != 0, 1)

    # Loop through all patient maps, and multiply each by each mask, and calculate ROI statistics
    df = pd.DataFrame(all_patients["patientList"])
    dfFiles = df.assign(file_path=lambda df: df["patientList"].map(lambda x: r'%s/%s.csv' % (dataDir, x)))

    upper_array = []
    x = 0
    for f in dfFiles.file_path:
        patient_map = pd.read_csv(f, header=None)
        patient_map = patient_map.drop(patient_map.index[59])
        patient_upper_region = pd.DataFrame(upper_ROI_mask.values * patient_map.values)
        upper_array.append(patient_upper_region.values.mean())
        x = x + 1

    all_patients['mean_upper_region'] = upper_array

    return all_patients


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
    cleaned_data = cleaned_data.drop(['recurrence_4years', 'sdDoseDiff', 'volumeContour', 'volumeRatio'], axis=1)
    # field, _ = stack_local_fields(clean_data, 1)
    # del_r = field[17][80][:]
    # clean_data['delta_r'] = del_r
    # cleaned_data = clean_data.drop(['patientNumber'])
    cleaned_data = numerate_categorical_data(cleaned_data)

    return cleaned_data


def test_categorical_map():
    initial_map = pd.read_csv(r'/Users/Tom/PycharmProjects/Contour-Analysis/Data/Deep_learning_results/deltaRMaps/196703818.csv', header=None).values
    final_map = list(map(lambda y: list(map(lambda x: 0 if x <= -2 else (2 if x >= 2 else 1), y)), initial_map))
    new = final_map.reshape(60,120)
    print(new)
    # print(final_map.shape)
    return


def COM_analysis(COM_df):
    """
    A function which will calculate the distance between the COM_auto and COM_man.

    :param COM_df: A dataframe containing the COM coordinates of the auto and man contours for all patients.
    :param
    :return: COM_df, updated to contain differences in each direction
    """

    COM_df['dx'] = COM_df['COMx'] - COM_df['ManCOMx']
    COM_df['dy'] = COM_df['COMy'] - COM_df['ManCOMy']
    COM_df['dz'] = COM_df['COMz'] - COM_df['ManCOMz']

    return COM_df


if __name__ == '__main__':
    # read_and_return_patient_stats()
    # dataDirectory = r"../Data/Deep_learning_results/deltaRMaps"
    # clean_dataset = clean_data(enhancedDF)
    # print(list(clean_dataset))

    patient_df = pd.read_csv(r'../Data/Deep_learning_results/global_results/19_fraction_FINAL_RESULTS.csv')
    test_plot_subplot(patient_df)
    # patient_df = pd.read_csv(r'../Data/Deep_learning_results/global_results/19_fraction_cleaned_AJ.csv')
    #
    # survival_analysis(patient_df, "age", "age_KM")
    # print("age")
    # survival_analysis(patient_df, "age", "age_KM")
    # print("manVol")
    # survival_analysis(patient_df, "manVol", "manVol_KM")
    # print("grade")
    # survival_analysis(patient_df, "grade", "grade_KM")

    # survival_analysis(patient_df, "age", "age_KM")
    # survival_analysis(patient_df, "manVol", "manVol_KM")
    # survival_analysis(patient_df, "grade", "grade_KM")