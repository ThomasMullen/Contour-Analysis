
"""
Created on Thu Dec 13 16:25:18 2018

@author: Tom
"""

import numpy as np
import pandas as pd
import seaborn as sns;

from AllPatients import AllPatients, recurrenceGroups

sns.set()
import matplotlib.pyplot as plt


# =============================================================================
# Load patients list data for all fractions
# =============================================================================

SaveDirect = "/Users/Tom/Documents/University/ProstateCode/LocalAnalysis/Final/"

# =============================================================================
# Group the patients by fractions, and recurrence
# =============================================================================
def patientRecurrenceFracs():
    (PatientsWhoRecur, PatientsWhoDontRecur) = load_global_patients().recurrenceGroups()

    # Group patients with fractions
    PatientRecurrencew19Frac = PatientsWhoRecur.groupby('Fractions').get_group(19)
    PatientRecurrencew16Frac = PatientsWhoRecur.groupby('Fractions').get_group(16)
    PatientNonRecurrencew20Frac = PatientsWhoDontRecur.groupby('Fractions').get_group(20)
    PatientNonRecurrencew19Frac = PatientsWhoDontRecur.groupby('Fractions').get_group(19)
    PatientNonRecurrencew16Frac = PatientsWhoDontRecur.groupby('Fractions').get_group(16)

# =============================================================================
# # Read in the patients map and store in correct container
# =============================================================================
# Patient map containers
patientMapRecurrenceContainer = []
patientMapNonRecurrenceContainer = []

corLocal = {'200801658', '200606193', '200610929', '200701370'}

def calcPatientMapSD(patientMap):
    sxx = 0
    mapMean = (sum(patientMap.flatten())) / patientMap.size
    for radDiff in patientMap.flatten():
        sxx = sxx + (radDiff - mapMean) ** 2
    sdValue = np.sqrt(sxx / (patientMap.size - 1))
    mapMax = patientMap.max()
    
    return (mapMean, sdValue, mapMax)


'''
Specify the corrupt patients to be filtered out of analysis
# ===================================================================
'''
def load_global_patients():
    # List the patient ID's of those who are contained in our ATLAS and have corrupted local maps & prothesis
    atlas = {'200806930', '201010804', '201304169', '201100014', '201205737', '201106120', '201204091', '200803943',
             '200901231', '200805565', '201101453', '200910818', '200811563', '201014420'}

    '''we have identified these corrupted from previous contour. we need to check '''
    expected_corrupt_to_check = {} 
    
    #{'200701370', '200700427', '200610929', '200606193', '200600383', '200511824','196708754', '200801658', '201201119', '200911702', '200701370', '200700427','200610929', '200606193', '200600383', '200511824'}


    corrupt = atlas.union(expected_corrupt_to_check)
    allPatients = AllPatients(r"../Data/OnlyProstateResults/GlobalwStage",
                              ['AllData19Frac', 'AllData16Frac_old', 'AllData16Frac', 'AllData19Frac_old'])
    allPatients.removePatients(corrupt)
    return allPatients

def calcPatientMapSD2(dataDir, patientId):
    file = r"%s/%s.csv" % (dataDir, patientId)
    matrix = pd.read_csv(file, header=None).as_matrix()
    result =  calcPatientMapSD(matrix)
    return result

def calcPatientMapSD2_mean(dataDir, patientId):
    (m, s, maxval) =  calcPatientMapSD2(dataDir, patientId)
    return m

def calcPatientMapSD2_sd(dataDir, patientId):
    (m, s, maxval) = calcPatientMapSD2(dataDir, patientId)
    return s

def calcPatientMapSD2_max(dataDir, patientId):
    (m, s, maxval) = calcPatientMapSD2(dataDir, patientId)
    return maxval

'''
Finds the Mean and the SD of the radial difference at each solid angle for each patient. And appends data to global patient df
'''
def radial_mean_sd_for_patients(dataDir, allPatientsDF):
    df = allPatientsDF.assign(mean=lambda df: df["patientList"].map(lambda x: calcPatientMapSD2_mean(dataDir, x)))
    df2 = df.assign(sd=lambda df: df["patientList"].map(lambda x: calcPatientMapSD2_sd(dataDir, x)))
    df3 = df2.assign(maxval=lambda df2: df2["patientList"].map(lambda x: calcPatientMapSD2_max(dataDir, x)))
    return df3


# =============================================================================
# Make arrays for theta and phi axes labels
# =============================================================================

def plotHist(data, colour, bin, name="Single Value"):
    result = plt.hist(data, bins=bin, alpha=0.5, label='map sd value', color=colour)
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    # plt.xlim((min(data), max(data)))
    plt.show()

def plotHist2(data1, colour1, bin1, data2, colour2, bin2, xmin = 0, xmax = 0, name="Single Value",legendPos="upper right"):
    plt.hist(data1, bins=bin1, alpha=0.5, label='Recurrence', color=colour1,normed=True)
    plt.hist(data2, bins=bin2, alpha=0.5, label='No Recurrence', color=colour2,normed=True)
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.legend(loc=legendPos)
    if xmin != 0 and xmax != 0:
        plt.xlim(xmin, xmax)
    plt.show()

'''
plots a heat map of patients radial map: expect df of local field passed in.
'''
def plot_heat_map(data, lower_limit, upper_limit, title=" "):
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
    map = data.as_matrix()
    heat_map = sns.heatmap(map, center=0, xticklabels=phi, yticklabels=theta, vmin=lower_limit, vmax=upper_limit, cmap='RdBu')
    heat_map.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$', title = title)
    plt.show()

def plot_scatter(data, colour, legendPos="upper right"):
    
    # =============================================================================
    # Plot a scatter graph for the volume of contour versus auto-contour
    # =============================================================================
    
    # Fitting a linear regression for comparison
    fit = np.polyfit(data["volumeContour"],data["volumeContourAuto"],1)
    fit_fn = np.poly1d(fit) 
    
    # Fitting the graph
#    fig = plt.figure()
    x = np.linspace(0, 160, 1000) #Plot straight line
    y = x
    plt.scatter(data["volumeContour"], data["volumeContourAuto"],c=colour,label='All patients')
    plt.plot(x,y,linestyle = 'solid') # y = x line
    #plt.plot(x,x,'yo', AllPatients["volumeContour"], fit_fn(AllPatients["volumeContour"]), '--k') # linear fit
    #plt.xlim(0, 150)
    #plt.ylim(0, 130)
    plt.xlabel('Manual contour volume [cm$^3$]')
    plt.ylabel('Automatic contour volume [cm$^3$]')
    plt.legend(loc=legendPos);
    plt.grid(True)
    plt.show()

def partition_patient_data_with_outliers(data, lower_bound, upper_bound, select_what = "sd"):

    lower_cut_off = np.percentile(data[select_what], lower_bound)
    upper_cut_off = np.percentile(data[select_what], upper_bound)
    print("%s, %s" % (lower_cut_off, upper_cut_off))
    
    if select_what == "DSC":
        selected_patients = data[data.DSC.between(lower_cut_off, upper_cut_off)]
        lower_patients_outliers = data[data.DSC < lower_cut_off]
        upper_patients_outliers = data[data.DSC > upper_cut_off]
    elif select_what == "volumeContourDifference":
        selected_patients = data[data.volumeContourDifference.between(lower_cut_off, upper_cut_off)]
        lower_patients_outliers = data[data.volumeContourDifference < lower_cut_off]
        upper_patients_outliers = data[data.volumeContourDifference > upper_cut_off]
    elif select_what == "maxval":
        selected_patients = data[data.maxval.between(lower_cut_off, upper_cut_off)]
        lower_patients_outliers = data[data.maxval < lower_cut_off]
        upper_patients_outliers = data[data.maxval > upper_cut_off]
    else:
        selected_patients = data[data.sd.between(lower_cut_off, upper_cut_off)]
        lower_patients_outliers = data[data.sd < lower_cut_off]
        upper_patients_outliers = data[data.sd > upper_cut_off]
        
    return (selected_patients, lower_patients_outliers, upper_patients_outliers)

def partition_patient_data_with_range(data, lower_bound, upper_bound, select_what):
    ''' A function to return a dataframe of all patients within a certain range '''
    
    lower_cut_off = data.loc[data[select_what] < lower_bound]
    upper_cut_off = data.loc[data[select_what] > upper_bound]
    
    if select_what == "DSC":
        selected_patients = data[data.DSC.between(lower_cut_off, upper_cut_off)]
    elif select_what == "volumeContourDifference":
        selected_patients = data[data.volumeContourDifference.between(lower_cut_off, upper_cut_off)]
    else:
        selected_patients = data
        print("None selected for DSC and Vdiff")
        
    return selected_patients

def return_patient_sample_range(data, size, lower_bound, upper_bound):
    selected_patients, lower_patients_outliers, upper_patients_outliers = partition_patient_data_with_outliers(data, lower_bound, upper_bound)
    return (selected_patients.sample(n=size), lower_patients_outliers, upper_patients_outliers)

def test_filtered():
    dataDirectory = r"../Data/OnlyProstateResults/AllFields"
    outputDirectory = r"../outputResults"
    # (meanVals, sdVals) = extractPatientSDVals(dataDirectory, allPatients.allPatients)
    rawPatientData = load_global_patients() # returns the data cleaned for atlas and corruption
    enhancedDF = radial_mean_sd_for_patients(dataDirectory, rawPatientData.allPatients)
    selected_patients, lower_patients_outliers, upper_patients_outliers = partition_patient_data_with_outliers(
        enhancedDF, 10, 90)
    lower_patients_outliers.to_csv('%s/lower_patients_outliers.csv' % outputDirectory)
    upper_patients_outliers.to_csv('%s/upper_patients_outliers.csv' % outputDirectory)

    patients_who_recur, patients_who_dont_recur = recurrenceGroups(selected_patients)
    # # Plot Histograms with raw data
    # all_patients_with_recurrence, all_patients_without_recurrence = recurrenceGroups(enhancedDF)
    # plotHist2(all_patients_with_recurrence['sd'], 'red', 75,all_patients_without_recurrence['sd'],'green',75, "Standard Deviation of Radial Difference")
    # plotHist(all_patients_with_recurrence['sd'], 'red', 75, "Standard Deviation of Radial Difference Recurrence")
    # plotHist(all_patients_without_recurrence['sd'], 'green', 75, "Standard Deviation of Radial Difference no Recurrence")
    # plotHist2(all_patients_with_recurrence['sd'], 'blue', 25,all_patients_without_recurrence['sd'],'blue',25, "Standard Deviation of Radial Difference")
    #
    # # Plot Histgrams with patients radial Difference SD (cut)
    # plotHist2(patients_who_recur['sd'], 'red', 75,patients_who_dont_recur['sd'],'green',75, "Standard Deviation of Radial Difference")
    # plotHist(patients_who_recur['sd'], 'red', 75, "Standard Deviation of Radial Difference Recurrence")
    # plotHist(patients_who_dont_recur['sd'], 'green', 75, "Standard Deviation of Radial Difference no Recurrence")
    # plotHist2(patients_who_recur['sd'], 'red', 25,patients_who_dont_recur['sd'],'green',25, "Standard Deviation of Radial Difference")
    # plotHist(patients_who_recur['sd'], 'red', 25, "Standard Deviation of Radial Difference Recurrence")
    # plotHist(patients_who_dont_recur['sd'], 'green', 25, "Standard Deviation of Radial Difference no Recurrence")

#     Output sample of patients within certain percentiles
    sample1 = return_patient_sample_range(enhancedDF,5,10,20)
    sample1.to_csv('%s/sample_set1.csv' % outputDirectory)
    sample2 = return_patient_sample_range(enhancedDF, 5, 30, 50)
    sample2.to_csv('%s/sample_set2.csv' % outputDirectory)
    sample3 = return_patient_sample_range(enhancedDF, 5, 50, 80)
    sample3.to_csv('%s/sample_set3.csv' % outputDirectory)
    sample4 = return_patient_sample_range(enhancedDF, 5, 80, 90)
    sample4.to_csv('%s/sample_set4.csv' % outputDirectory)


def main():
    test_filtered()

if __name__ == '__main__':
    main()
# TODO Cut off upper 10 percentile for local sd
# TODO Cut off ContourVolumeDifference Upper and Lower
# TODO Print Histograms
# TODO Use Local Combined Map to produce an average and variance Radial Patient for recurrence and non-recurrence
