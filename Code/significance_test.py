"""
Created Tue 19th Feb
by: Tom and AJ
This source file contains all our significance test functions
"""
import matplotlib.pyplot as plt
import numpy as np
import pymining as pm
import pandas as pd
import seaborn as sns

from scipy import stats as ss
from AllPatients import separate_by_recurrence
from plot_functions import create_polar_axis
from mlxtend.evaluate import permutation_test
from lifelines import CoxPHFitter
from DataFormatting import data_frame_to_XDR


def pymining_t_test(selected_patients):
    # Tag patients with recurrence:1 and non-recurrence:0
    patients_who_recur, patients_who_dont_recur = separate_by_recurrence(selected_patients)
    (rec_fieldMaps, rec_label_array) = stack_local_fields(patients_who_recur, 1)
    (non_rec_field_maps, non_rec_label_array) = stack_local_fields(patients_who_dont_recur, 0)

    # Concatenate the two
    totalPatients = np.concatenate((rec_fieldMaps, non_rec_field_maps), axis=-1)
    labels = np.concatenate((rec_label_array, non_rec_label_array))

    # Now use pymining to get DSC cuts global p value. It should be similar to that from scipy
    global_neg_pvalue, global_pos_pvalue, neg_tthresh, pos_tthresh = pm.permutationTest(totalPatients, labels, 1000)
    t_value_map = pm.imagesTTest(totalPatients, labels)  # no longer.[0] element

    return global_neg_pvalue, global_pos_pvalue, neg_tthresh, pos_tthresh, t_value_map


def t_map_with_thresholds(t_map):
    """
    A function which will apply contours on the t-map, at values of the 5th and 95th percentiles of the
    distribution of the t-map.
    :param t_map: A 2D array of t-values
    :return: A plot of the t-map with p-contours
    """

    critical_t_values = np.percentile(t_map.flatten(), [5, 95])
    # clrs = ['magenta', 'lime', 'orange', 'red']
    plt.contour(t_map, levels=critical_t_values, colors='magenta')
    plt.gca()
    plt.show()


def test_superimpose(t_map, pos_t_dist, neg_t_dist):
    '''
    This will plot the tmap contours superimposed on the t-map which is np.flip vertically
    :param t_map: take in the numpy array t-map
    :return: returns a plots of the contours of the tmap
    '''
    coutour_map = t_map.copy()
    # clrs = ['magenta', 'lime', 'orange', 'red']
    # critical_t_values = np.percentile(coutour_map.flatten(), [5, 95])
    upper_tail = np.percentile(pos_t_dist, 95)
    lower_tail = np.percentile(neg_t_dist, 5)
    plt.contour(coutour_map, levels=[lower_tail, upper_tail], colors=['magenta', 'lime'])
    plt.gca()
    axes = create_polar_axis()
    heat_map = sns.heatmap(t_map, center=0, xticklabels=axes[0], yticklabels=axes[1], cmap='RdBu')
    # heat_map = sns.heatmap(np.flip(t_map, 0), center=0, xticklabels=axes[0], yticklabels=axes[1], cmap='RdBu')
    heat_map.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$', title='t-map')
    plt.show()


def map_with_thresholds(statistic_map, critical_statistic_values=[5, 95], is_percentile=True):
    '''
    This will plot the contours superimposed on the statistic map which is np.flip vertically
    :param t_map: take in the numpy array t-map
    :return: returns a plots of the contours of the tmap
    '''
    coutour_map = statistic_map.copy()

    if is_percentile == True:
        critical_statistic_values = np.percentile(statistic_map.flatten(), critical_statistic_values)
    plt.contour(coutour_map, levels=critical_statistic_values, colors=['magenta', 'lime'])
    plt.gca()
    axes = create_polar_axis()
    heat_map = sns.heatmap(statistic_map, center=0, xticklabels=axes[0], yticklabels=axes[1], cmap='RdBu')
    # heat_map = sns.heatmap(np.flip(t_map, 0), center=0, xticklabels=axes[0], yticklabels=axes[1], cmap='RdBu')
    heat_map.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$', title='')
    plt.show()


def pValueMap(t_to_p_map):
    """
    A function which will create a map of p-values from a map of t-values and thresholds
    Start at the 0th percentile, and iterate up to the 100th percentile in increments of 1.
    Upon each iteration, obtain the tthresh-value at each percentile
    Find the number of points in tthresh above the tthresh-value, to obtain a non-normalised p-value
    Normalise the p-value by dividing by the number of map elements, i.e. the size of t_to_p_map
    :param t_to_p_map: A 2D array of t-values
    :return: A 2D array of p-values
    """

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


def stack_local_fields(global_df, recurrence_label, dataDir=r'../Data/Deep_learning_results/deltaRMaps'):
    """
    :param global_df: either recurring non-recurring global data field
    :param recurrence_label:  =0 for non-recurring or =1 for recurring
    :param dataDir: Data directory containing local field data file
    :return: 3d np array of local field stacked 120x60xnumber of recurrence/non-recurrence i.e [theta x phi x patient_index] and label
    """

    df = pd.DataFrame(global_df["patientList"]).astype(int)
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
        # if i != 17 and i != 247 and i != 256 and i != 277 and i != 281:
        # data_frame_to_XDR(fieldMaps[:, :, i], global_df["patientList"][i])
        i += 1

    return fieldMaps, label_array


def p_value_contour_plot(t_map, t_thresh, percentile_array):
    """
    Take in t-map, t-threshold distribution, and upper/lower tail. Will produce a map with significant contours 0.002,
    0.005, 0.01, 0.05. This uses Andrews method.
    :param t_map: it a map of t-statistic
    :param t_thresh: t threshold distribution
    :param percentile_array: list of percentiles to be contoured
    :return: t-map with p-value contours of a specific tail
    """

    # get t at percentiles of t_thresh
    critical_t_values = np.percentile(t_thresh, percentile_array)
    # contour labels of p-values
    # p_value_names = percentile_array/100
    clrs = ['magenta', 'lime', 'orange']  # , 'red']
    plt.contour(t_map, levels=critical_t_values, colors=clrs)
    plt.gca()
    plt.show()


def global_statistical_analysis(selected_patients):
    '''
    A function that will carry out the statistical analysis of the global data: The dice coefficient and volume
    difference for patients with and without cancer recurrence.

    :param selected_patients: A dataframe of all patients
    '''

    # Split the patients by recurrence
    patients_who_recur, patients_who_dont_recur = separate_by_recurrence(selected_patients)
    print('#rec: %.f, #n_rec: %.f' % (patients_who_recur.shape[0], patients_who_dont_recur.shape[0]))

    # Test the relationship between volume difference and recurrence
    global_p, lower_p, upper_p = non_parametric_permutation_test(patients_who_recur["volumeContourDifference"],
                                                                 patients_who_dont_recur["volumeContourDifference"])
    print('Vdiff: p_value(rec=/=n_rec): %.6f p_value(rec<n_rec): %.6f p_value(rec>n_rec): %.6f' %
          (global_p, lower_p, upper_p))

    # Test the relationship between dice coefficient and recurrence
    global_p, lower_p, upper_p = non_parametric_permutation_test(patients_who_recur["DSC"],
                                                                 patients_who_dont_recur["DSC"])
    print('DSC: p_value(rec=/=n_rec): %.6f p_value(rec<n_rec): %.6f p_value(rec>n_rec): %.6f' %
          (global_p, lower_p, upper_p))

    # Test the relationship between ratio of volumes and recurrence: V_man/V_auto
    global_p, lower_p, upper_p = non_parametric_permutation_test(patients_who_recur["volumeRatio"],
                                                                 patients_who_dont_recur["volumeRatio"])
    print('VRatio: p_value(rec=/=n_rec): %.6f p_value(rec<n_rec): %.6f p_value(rec>n_rec): %.6f' %
          (global_p, lower_p, upper_p))

    # Test the relationship between mean dose diff and recurrence: V_man/V_auto
    global_p, lower_p, upper_p = non_parametric_permutation_test(patients_who_recur["meanDoseDiff"],
                                                                 patients_who_dont_recur["meanDoseDiff"])
    print('VRatio: p_value(rec=/=n_rec): %.6f p_value(rec<n_rec): %.6f p_value(rec>n_rec): %.6f' %
          (global_p, lower_p, upper_p))


def wilcoxon_test(rec_field_maps, nonrec_field_maps):
    # TODO:  implement random sampling

    stat = np.zeros((60, 120))
    p_value = np.zeros((60, 120))

    for x in range(60):
        for y in range(120):
            stat[x][y], p_value[x][y] = ss.wilcoxon(rec_field_maps[x][y][:], nonrec_field_maps[x][y][:74])
    #         56 for low and intermediate combined
    #         74 for high on its own
    return stat, p_value


def mann_whitney_u_test(rec_field_maps, nonrec_field_maps):
    """
    A function to conduct the mann whitney u test per voxel of our radial difference maps

    :param rec_field_maps: recurrence maps
    :param nonrec_field_maps: no recurrence maps
    :return: A map of the u-statistic and the p-value
    """

    stat = np.zeros((60, 120))
    p_value = np.zeros((60, 120))
    for x in range(60):
        for y in range(120):
            stat[x][y], p_value[x][y] = ss.mannwhitneyu(rec_field_maps[x][y][:], nonrec_field_maps[x][y][:],
                                                        alternative='two-sided')
    return stat, p_value


def sample_normality_test(clean_data, theta, azi, sample_size=25):
    '''
    This function performs a normality check on the patient delta df to ensure that it can be used in a t-test
    :param clean_data: patient df
    :param theta: range 0-60
    :param azi: range 0-120
    :param sample_size: size for sample check
    :return:
    '''

    stackd_fields = stack_local_fields(clean_data, 1)
    # Select random sample from the voxel element
    # random_sample = np.random.choice(stackd_fields[0][theta][azi][:], sample_size)
    # perform normality check
    k2, p = ss.normaltest(stackd_fields[0][theta][azi][:])
    alpha = 1e-3
    print("p = {:g}".format(p))
    print("k2 = {:g}".format(k2))
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print("The null hypothesis can be rejected")
    else:
        print("The null hypothesis cannot be rejected")
    return k2, p

def normality_map(clean_data):
    stackd_fields = stack_local_fields(clean_data, 1)
    kurt2 = np.zeros((60, 120))
    p_value = np.zeros((60, 120))
    for x in range(60):
        for y in range(120):
            kurt2[x][y], p_value[x][y] = ss.normaltest(np.random.choice(stackd_fields[0][x][y][:], 30))
    return kurt2, p_value

def wilcoxon_test_statistic(selected_patients):
    """
    Conduct a wilcoxon SIGN rank test

    :param selected_patients: Dataframe of all patients
    :return: maps of the statistic and the p-values
    """

    # Tag patients with recurrence:1 and non-recurrence:0
    patients_who_recur, patients_who_dont_recur = separate_by_recurrence(selected_patients)
    rec_fieldMaps, _ = stack_local_fields(patients_who_recur, 1)
    nonrec_fieldMaps, _ = stack_local_fields(patients_who_dont_recur, 0)
    stat_map, p_map = wilcoxon_test(rec_fieldMaps, nonrec_fieldMaps)
    return stat_map, p_map


def mann_whitney_test_statistic(selected_patients):
    """
    Conduct a mann whitney test SUM rank test

    :param selected_patients: dataframe of all patients
    :return: maps of the statistic and the p-values
    """

    # Tag patients with recurrence:1 and non-recurrence:0
    patients_who_recur, patients_who_dont_recur = separate_by_recurrence(selected_patients)
    rec_fieldMaps, _ = stack_local_fields(patients_who_recur, 1)
    nonrec_fieldMaps, _ = stack_local_fields(patients_who_dont_recur, 0)
    stat_map, p_map = mann_whitney_u_test(rec_fieldMaps, nonrec_fieldMaps)
    return stat_map, p_map


def non_parametric_permutation_test(recurrence_group, no_recurrence_group):
    """
    A non-parametric permutation test for the null hypothesis that patients grouped by cancer recurrence come from
    the same distribution.

    1) Compute the difference (here: mean) of sample x and sample y
    2) Combine all measurements into a single dataset
    3) Draw a permuted dataset from all possible permutations of the dataset in 2.
    4) Divide the permuted dataset into two datasets x' and y' of size n and m, respectively
    5) Compute the difference (here: mean) of sample x' and sample y' and record this difference
    6) Repeat steps 3-5 until all permutations are evaluated
    7) Return the p-value as the number of times the recorded differences were more extreme than the original
    difference from 1. and divide this number by the total number of permutations

    :param selected_patients: A data frame column of a patient characteristic to analyse
    :param test_type: A string either 'two_tail', or 'one_tail' to signify the type of test
    :return: The p-value for the test
    """

    global_p_value = permutation_test(recurrence_group, no_recurrence_group, method='approximate', num_rounds=10000,
                                      seed=0)
    lower_p_value = permutation_test(recurrence_group, no_recurrence_group, func='x_mean < y_mean',
                                     method='approximate', num_rounds=10000, seed=0)
    upper_p_value = permutation_test(recurrence_group, no_recurrence_group, func='x_mean > y_mean',
                                     method='approximate', num_rounds=10000, seed=0)

    return global_p_value, lower_p_value, upper_p_value


def cph_global_test(global_df):
    '''
    Produces PH cox regression and write to an external file marking the history of predicotr variable being removed
    line 1188 in coxph_fitter.py shows cph parameters that we can access its useful!
    This funciton returns a cph stats df
    :param global_df: The clean global data frame with variable removed
    :return: a description of the covariates significance to patient survival
    '''
    cph = CoxPHFitter()
    cph.fit(global_df, duration_col='timeToEvent', event_col='recurrence_outcome', show_progress=True)
    cph_stats = cph.summary
    print(cph_stats)
    return cph_stats


def test():
    return 0


if __name__ == '__main__':
    test()
