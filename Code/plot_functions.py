import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from AllPatients import separate_by_recurrence
from LocalCombined import make_average_field2

sns.set()


def show_local_fields(global_df, dataDir=r'../Data/OnlyProstateResults/AllFields', file_name='untitled'):
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
    dfFiles.patientList.to_csv('../patient_outliers/' + file_name + '.csv')
    for f in dfFiles.file_path:
        print(dfFiles.iloc[x].patientList)
        # plot_heat_map(pd.read_csv(f, header=None), -5, 5, dfFiles.iloc[x].patientList)
        x = x + 1


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


def plot_histogram(data, colour, bin, name="Single Value"):
    '''
    Plots a histogram of 1d data
    :param data: field name from the global dataset
    :param colour: string of histogram colour
    :param bin: number of bins
    :param name: x axis label
    :return: returns histrgram plot of 1 parameter
    '''
    result = plt.hist(data, bins=bin, alpha=0.5, label='map sd value', color=colour)
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    plt.xlim((min(data), max(data)))
    plt.show()


def plot_histogram_with_two_data_sets(data1, colour1, bin1, data2, colour2, bin2, name="Single Value",
                                      legendPos="upper right"):
    plt.hist(data1, bins=bin1, alpha=0.5, label='Recurrence', color=colour1, normed=True)
    plt.hist(data2, bins=bin2, alpha=0.5, label='No Recurrence', color=colour2, normed=True)
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.legend(loc=legendPos)
    plt.xlim(-20, 60)
    plt.show()


def create_polar_axis():
    """
    defines the ticks on the 2d histogram axis
    :returns: $\phi$ array with values and DSC cuts $\theta$ array with values
    """
    phi = [];
    theta = []
    for i in range(0, 120):
        phi.append('')
    for i in range(0, 60):
        theta.append('')
    # Define ticks
    phi[0] = 0;
    phi[30] = 90;
    phi[60] = 180;
    phi[90] = 270;
    phi[119] = 360;
    theta[0] = -90;
    theta[30] = 0;
    theta[59] = 90
    return phi, theta


def plot_heat_map(data, lower_limit, upper_limit, title=" "):
    """
    defines the ticks on the 2d histogram axis
    :param: data is the field to be plotted
    :param: lower_limit minimum heat colour
    :param: upper_limit maximum heat colour
    :param: title is the title given to the heat map default is empty
    :returns: $\phi$ array with values and DSC cuts $\theta$ array with values
    """
    axes = create_polar_axis()
    heat_map = sns.heatmap(data.as_matrix(), center=0, xticklabels=axes[0], yticklabels=axes[1], vmin=lower_limit,
                           vmax=upper_limit,
                           cmap='RdBu')
    heat_map.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$', title=title)
    plt.show()


def plot_heat_map_np(data, title=" "):
    """
    defines the ticks on the 2d histogram axis uses numpy matrix not df
    :param: data is the field to be plotted
    :param: title is the title given to the heat map default is empty
    :returns: $\phi$ array with values and DSC cuts $\theta$ array with values
    """
    axes = create_polar_axis()
    heat_map = sns.heatmap(data, center=0, xticklabels=axes[0], yticklabels=axes[1], cmap='RdBu')
    heat_map.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$', title=title)
    plt.show()


def plot_scatter(data, colour, legendPos="upper right"):
    '''

    :param data: scatter plot data
    :param colour: colour of dot
    :param legendPos: where legend should be positioned
    :return: scatter plot of volume contour and volum auto-contour
    '''

    # Fitting DSC cuts linear regression for comparison
    fit = np.polyfit(data["volumeContour"], data["volumeContourAuto"], 1)
    fit_fn = np.poly1d(fit)

    # Fitting the graph
    #    fig = plt.figure()
    x = np.linspace(0, 160, 1000)  # Plot straight line
    y = x
    plt.scatter(data["volumeContour"], data["volumeContourAuto"], c=colour, label='All patients')
    plt.plot(x, y, linestyle='solid')  # y = x line
    # plt.plot(x,x,'yo', AllPatients["volumeContour"], fit_fn(AllPatients["volumeContour"]), '--k') # linear fit
    # plt.xlim(0, 150)
    # plt.ylim(0, 130)
    plt.xlabel('Manual contour volume [cm$^3$]')
    plt.ylabel('Automatic contour volume [cm$^3$]')
    plt.legend(loc=legendPos);
    plt.grid(True)
    plt.show()


def save_heat_map(data, lower_limit, upper_limit, save_name=" ", local_folder=" "):
    """
    defines the ticks on the 2d histogram axis
    :param: data is the field to be plotted
    :param: lower_limit minimum heat colour
    :param: upper_limit maximum heat colour
    :param: title is the title given to the heat map default is empty
    :returns: $\phi$ array with values and DSC cuts $\theta$ array with values
    """
    axes = create_polar_axis()
    heat_map = sns.heatmap(data, center=0, xticklabels=axes[0], yticklabels=axes[1], vmin=lower_limit,
                           vmax=upper_limit,
                           cmap='RdBu')
    heat_map.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$')
    plt.show()
    plt.savefig('../Neat-Output-Contour-Analysis/%s/%s.png' % (local_folder, save_name))


def load_map(data_directory, name):
    file = r"%s/%s.csv" % (data_directory, name)
    return pd.read_csv(file, header=None)


def test_on_single_map():
    dataDirectory = r"../Data/OnlyProstateResults/AllFields"
    map = load_map(dataDirectory, "200710358")
    plot_heat_map(map, -2, 2, title=" ")
    # save_heat_map(map, -2, 2, 'testmap', "tester")


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

def main():
    test_on_single_map()


if __name__ == '__main__':
    main()
