import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set()


def plot_histogram(data, colour, bin, name="Single Value"):
    result = plt.hist(data, bins=bin, alpha=0.5, label='map sd value', color=colour)
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    plt.xlim((min(data), max(data)))
    plt.show()


def plot_histogram_with_two_data_sets(data1, colour1, bin1, data2, colour2, bin2,min_range,max_range, name="Single Value",
                                      legendPos="upper right"):
    plt.hist(data1, bins=bin1, alpha=0.5, label='Recurrence', color=colour1, normed=True)
    plt.hist(data2, bins=bin2, alpha=0.5, label='No Recurrence', color=colour2, normed=True)
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.legend(loc=legendPos)
    plt.xlim(min_range, max_range)
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
    defines the ticks on the 2d histogram axis
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

def main():
    test_on_single_map()

if __name__ == '__main__':
    main()