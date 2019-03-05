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

    # Test the relationship between volume difference and recurrence
    global_neg_p_value, global_pos_p_value, _, _ = pm.permutationTest(
        selected_patients["volumeContourDifference"], selected_patients["recurrence"], 1000)

    print('Vdiff: Global negative p: %.6f Global positive p: %.6f' % (global_neg_p_value, global_pos_p_value))

    # Test the relationship between dice coefficient and recurrence
    global_neg_p_value, global_pos_p_value, _, _ = pm.permutationTest(
        selected_patients["DSC"], selected_patients["recurrence"], 1000)

    print('Dice: Global negative p: %.6f Global positive p: %.6f' % (global_neg_p_value, global_pos_p_value))


def wilcoxon_test(rec_field_maps, nonrec_field_maps):
    stat = np.zeros((60, 120))
    p_value = np.zeros((60, 120))
    for x in range(60):
        for y in range(120):
            stat[x][y], p_value[x][y] = ss.wilcoxon(rec_field_maps[x][y][:], nonrec_field_maps[x][y][:57])
    print(stat)


def wilcoxon_test_statistics(selected_patients):
    # TODO: randomise the no-recurrence patients
    # Tag patients with recurrence:1 and non-recurrence:0
    patients_who_recur, patients_who_dont_recur = separate_by_recurrence(selected_patients)
    rec_fieldMaps, _ = stack_local_fields(patients_who_recur, 1)
    nonrec_fieldMaps, _ = stack_local_fields(patients_who_dont_recur, 0)
    stat_map, p_map = wilcoxon_test(rec_fieldMaps, nonrec_fieldMaps)
    return stat_map, p_map


def test():
    return 0


if __name__ == '__main__':
    test()
