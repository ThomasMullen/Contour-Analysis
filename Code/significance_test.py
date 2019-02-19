"""
Created Tue 19th Feb
by: Tom and AJ
This source file contains all our significance test functions
"""
import matplotlib.pyplot as plt
import numpy as np
import pymining as pm
from scipy import stats as ss
from AllPatients import separate_by_recurrence
from LocalCombined import stack_local_fields


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