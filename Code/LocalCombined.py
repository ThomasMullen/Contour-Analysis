#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 16:25:18 2018

@author: Tom
"""

import numpy as np
import pandas as pd
import seaborn as sns;
import os



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
    
    x = 0
    for f in dfFiles.file_path:
        if x not in corruptMaps:
            fieldMaps.append(pd.read_csv(f, header=None))   
        x = x + 1
        
    masterDF = pd.concat(fieldMaps)
    by_row_indexRec = masterDF.groupby(masterDF.index)
    meanRecurrence = by_row_indexRec.mean()
    varRecurrence = by_row_indexRec.var()
    stdRecurrence = by_row_indexRec.std()

    return (meanRecurrence, varRecurrence, stdRecurrence)

def show_local_fields(global_df, dataDir = r'../Data/OnlyProstateResults/AllFields'):
    ''' A function to print the each patients radial field for inspection '''
    
    df = pd.DataFrame(global_df["patientList"])
    dfFiles = df.assign(file_path=lambda df: df["patientList"].map(lambda x: r'%s/%s.csv' % (dataDir, x)))

    fieldMaps = []
    
    x=0
    for f in dfFiles.file_path:
        fieldMaps.append(pd.read_csv(f, header=None))   
        plot_heat_map(fieldMaps[x],-1,1,x)
        x = x + 1
    

def test_local_field():
    dataDirectory = r"../Data/OnlyProstateResults/AllFields"
    outputDirectory = r"../outputResults"
    # (meanVals, sdVals) = extractPatientSDVals(dataDirectory, allPatients.allPatients)
    rawPatientData = load_global_patients()
    enhancedDF = radial_mean_sd_for_patients(dataDirectory, rawPatientData.allPatients)
    patients_who_recur, patients_who_dont_recur = recurrenceGroups(enhancedDF)

    DSCbins = [0,0.45,0.65,0.75,0.85,0.95,1]
    VolBins = [-55,-40,-30,-20,-10,0,10,20,30,40,55,200]
    
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
    plotHist2(patients_who_recur['sd'], 'red', 75,patients_who_dont_recur['sd'],'green',75, 0.001, 3, "Standard Deviation of radial difference, $\sigma_{\Delta R}$")
    
    # =============================================================================
    #     Cut based on the maximum value of each map, eliminates local map anomalies
    # =============================================================================
        # max value of map histogram before cuts
    plotHist2(patients_who_recur['maxval'], 'red', 75,patients_who_dont_recur['maxval'],'green',75, "Maximum value of radial difference, $max(\Delta R)$")
    
    selected_patients, lower_patients_outliers, upper_patients_outliers = partition_patient_data_with_outliers(selected_patients, 0, 98.5, select_what = "maxval") # 0-99.6 grabs 4 at large std dev # 99.73 std
    lower_patients_outliers.to_csv('%s/lower_patients_outliers_maxval.csv' % outputDirectory)
    upper_patients_outliers.to_csv('%s/upper_patients_outliers_maxval.csv' % outputDirectory)
    patients_who_recur, patients_who_dont_recur = recurrenceGroups(selected_patients)
#    plotHist2(patients_who_recur['volumeContourDifference'], 'r', VolBins, patients_who_dont_recur['volumeContourDifference'], 'g', VolBins, name="Volume difference between contour and auto-contour, $\Delta V$",legendPos="upper right")
#    plotHist2(patients_who_recur['DSC'], 'r', DSCbins, patients_who_dont_recur['DSC'], 'g', DSCbins, name="Dice coefficient",legendPos ="upper left")
       
        # max value map after cuts
    plotHist2(patients_who_recur['maxval'], 'red', 75,patients_who_dont_recur['maxval'],'green',75, 0.001, 5, "Maximum value of radial difference, $max(\Delta R)$")
    
    # =============================================================================
    #     Remove patients based on DSC and Vdiff globally
    # =============================================================================
        
            # DSC before cuts
    plotHist2(patients_who_recur['DSC'], 'red', DSCbins, patients_who_dont_recur['DSC'], 'green', DSCbins, 0.001, 1, name="Dice coefficient",legendPos ="upper left")
    
    # DSC cut & local maps
    selected_patients, lower_patients_outliers, upper_patients_outliers = partition_patient_data_with_outliers(selected_patients, 5, 100, select_what = "DSC") # 0-99.6 grabs 4 at large std dev # 99.73 std
    lower_patients_outliers.to_csv('%s/lower_patients_outliers_DSC.csv' % outputDirectory)
    upper_patients_outliers.to_csv('%s/upper_patients_outliers_DSC.csv' % outputDirectory)
    patients_who_recur, patients_who_dont_recur = recurrenceGroups(selected_patients)
#    plotHist2(patients_who_recur['volumeContourDifference'], 'r', VolBins, patients_who_dont_recur['volumeContourDifference'], 'g', VolBins, name="Volume difference between contour and auto-contour, $\Delta V$",legendPos="upper right")
     
        # DSC after cuts
    plotHist2(patients_who_recur['DSC'], 'red', DSCbins, patients_who_dont_recur['DSC'], 'green', DSCbins, 0.001, 1, name="Dice coefficient",legendPos ="upper left")
    
    
      # Vdiff before cuts
    plotHist2(patients_who_recur['volumeContourDifference'], 'red', VolBins, patients_who_dont_recur['volumeContourDifference'], 'green', VolBins, -55, 100, name="Volume difference between contour and auto-contour, $\Delta V$",legendPos="upper right")
   
    # Remove patients with SV contoured
    StagePatients = selected_patients.groupby('Stage')
    StageT3bPatients = pd.concat([StagePatients.get_group('T3b'), StagePatients.get_group('T3B'), StagePatients.get_group('T3b/T4'),StagePatients.get_group('T4')])    
    selected_patients = selected_patients[~selected_patients['patientList'].isin(StageT3bPatients['patientList'])]
    patients_who_recur, patients_who_dont_recur = recurrenceGroups(selected_patients)
    #show_local_fields(StageT3bPatients,dataDirectory) # print the local fields to see if they are damaged
          
       # Vdiff after SV cuts
    plotHist2(patients_who_recur['volumeContourDifference'], 'red', VolBins, patients_who_dont_recur['volumeContourDifference'], 'green', VolBins, -55, 100, name="Volume difference between contour and auto-contour, $\Delta V$",legendPos="upper right")
    
    # Vdiff cuts
    selected_patients, lower_patients_outliers, upper_patients_outliers = partition_patient_data_with_outliers(selected_patients, 4, 96, select_what = "volumeContourDifference") # 0-99.6 grabs 4 at large std dev # 99.73 std
    lower_patients_outliers.to_csv('%s/lower_patients_outliers_Vdiff.csv' % outputDirectory)
    upper_patients_outliers.to_csv('%s/upper_patients_outliers_Vdiff.csv' % outputDirectory)
    patients_who_recur, patients_who_dont_recur = recurrenceGroups(selected_patients)
    
        
       # Vdiff after cuts
    plotHist2(patients_who_recur['volumeContourDifference'], 'red', VolBins, patients_who_dont_recur['volumeContourDifference'], 'green', VolBins, -55, 100, name="Volume difference between contour and auto-contour, $\Delta V$",legendPos="upper right")
#        # scatter plot of volume contour to auto-contour
    plot_scatter(selected_patients,'r','upper left')
    
    # =============================================================================
    # Maps with data removed
    # =============================================================================
    
    # Maps with corrupt patients fully cut out
    (meanMap1, varMap, stdMap) = load_local_field_recurrence(patients_who_recur, dataDirectory)
    plot_heat_map(meanMap1, -1, 1, 'mean map - patients_who_recur')
    plot_heat_map(varMap, 0, 1, 'variance map - patients_who_recur')
    plot_heat_map(stdMap, 0, 1, 'standard deviation map - patients_who_recur')

    (meanMap2, varMap, stdMap) = load_local_field_recurrence(patients_who_dont_recur, dataDirectory, corruptMaps = [140,55])
    plot_heat_map(meanMap2, -1, 1, 'mean map - patients_who_dont_recur')
    plot_heat_map(varMap, 0, 1, 'variance map - patients_who_dont_recur')
    plot_heat_map(stdMap, 0, 1, 'standard deviation map - patients_who_dont_recur')
    
    plot_heat_map(meanMap1-meanMap2, -0.1, 0.1, 'mean map - patients_who_dont_recur')
    
    # printing local data fields
#    show_local_fields(patients_who_dont_recur,dataDirectory)
    
    
    
    return selected_patients
        

if __name__ == '__main__':
    selected_patients = test_local_field()

# # =============================================================================
# # # Display 2D Heat maps
# # =============================================================================
#
# # f, (recurrenceMean, recurrenceVar) = plt.subplots(1, 2)
# recurrenceMean = sns.heatmap(meanRecurrence,vmax = 1, vmin = -1, center=0,xticklabels=phi,yticklabels=theta)
# recurrenceMean.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$')
# fig = recurrenceMean.get_figure()
# # fig.savefig(tomLocalAnalysis + "16Frac" + "60x60meanRecurrenceMap.png")
# # plt.show()
# # fig.clear()
# #
# #
# # recurrenceVar = sns.heatmap(varRecurrence,vmax = 0.5, vmin = 0, center=0,xticklabels=phi,yticklabels=theta)
# # recurrenceVar.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$')
# # fig2 = recurrenceVar.get_figure()
# # # fig2.savefig(tomLocalAnalysis + "16Frac" +  "60x60varRecurrenceMap.png")
# # # plt.show()
# # # fig2.clear()
# #
# # nonRecurrenceMean = sns.heatmap(meanNonRecurrence,vmax = 0.5, vmin = -0.5, center=0,xticklabels=phi,yticklabels=theta)
# # nonRecurrenceMean.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$')
# # fig3 = nonRecurrenceMean.get_figure()
# # #fig3.savefig(tomLocalAnalysis + "16Frac" +  "60x60meanNonRecurrenceMap.png")
# # # plt.show()
# # # fig3.clear()
# #
# # nonRecurrenceVar = sns.heatmap(varNonRecurrence, vmax = 1, vmin = 0, center=0,xticklabels=phi,yticklabels=theta)
# # nonRecurrenceVar.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$')
# # fig4 = nonRecurrenceVar.get_figure()
# # # fig4.savefig(tomLocalAnalysis + "16Frac" +  "60x60varNonRecurrenceMap.png")
# # # plt.show()
# # # fig4.clear()
# #
# # DifferenceInMean = meanRecurrence - meanNonRecurrence
# # DifferenceInMeanGraph = sns.heatmap(DifferenceInMean, center=0,xticklabels=phi,yticklabels=theta)
# # DifferenceInMeanGraph.set(ylabel='Theta, $\dot{\Theta}$', xlabel='Azimutal, $\phi$')
# # fig5 = DifferenceInMeanGraph.get_figure()
# # # fig5.savefig(tomLocalAnalysis + "16Frac" +  "60x60differenceMeanMap.png")
# # # plt.show()
# # # fig5.clear()
# #
# # # Print patient number of corrupt maps
# # print(str(AllPatients.query("patientList == 200701370").patientNumber))
# # print(str(AllPatients.query("patientList == 200700427").patientNumber))
# # print(str(AllPatients.query("patientList == 200610929").patientNumber))
# # print(str(AllPatients.query("patientList == 200606193").patientNumber))
# # print(str(AllPatients.query("patientList == 200600383").patientNumber))
# # print(str(AllPatients.query("patientList == 200511824").patientNumber))
