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



from Code.AllPatients import recurrenceGroups
from Code.LocalFilter import load_global_patients, radial_mean_sd_for_patients, partition_patient_data_with_outliers, \
    plot_heat_map

sns.set()

# List the patient ID's of those who are contained in our ATLAS and have corrupted local maps & prothesis
atlas = {'200806930','201010804', '201304169', '201100014', '201205737','201106120', '201204091', '200803943', '200901231', '200805565', '201101453', '200910818', '200811563','201014420'}
corrupt = {'196708754','200801658','201201119','200911702','200701370','200700427','200610929','200606193','200600383','200511824'}
corrupt16frac = {'200701370','200700427','200610929','200606193','200600383','200511824'}


def load_local_field_recurrence(global_df, dataDir = r'../Data/OnlyProstateResults/AllFields'):
    df = pd.DataFrame(global_df["patientList"])
    dfFiles = df.assign(file_path=lambda df: df["patientList"].map(lambda x: r'%s/%s.csv' % (dataDir, x)))

    masterDF = pd.DataFrame.empty
    fieldMaps = []
    for f in dfFiles.file_path:
        fieldMaps.append(pd.read_csv(f, header=None))

    masterDF = pd.concat(fieldMaps)
    by_row_indexRec = masterDF.groupby(masterDF.index)
    meanRecurrence = by_row_indexRec.mean()
    varRecurrence = by_row_indexRec.var()
    stdRecurrence = by_row_indexRec.std()

    return (meanRecurrence, varRecurrence, stdRecurrence)


def test_local_field():
    dataDirectory = r"../Data/OnlyProstateResults/AllFields"
    outputDirectory = r"../outputResults"
    # (meanVals, sdVals) = extractPatientSDVals(dataDirectory, allPatients.allPatients)
    rawPatientData = load_global_patients()
    enhancedDF = radial_mean_sd_for_patients(dataDirectory, rawPatientData.allPatients)
    selected_patients, lower_patients_outliers, upper_patients_outliers = partition_patient_data_with_outliers(
        enhancedDF, 10, 90)
    # lower_patients_outliers.to_csv('%s/lower_patients_outliers.csv' % outputDirectory)
    # upper_patients_outliers.to_csv('%s/upper_patients_outliers.csv' % outputDirectory)

    patient_who_recur, patients_who_dont_recur = recurrenceGroups(selected_patients)
    (meanMap, varMap, stdMap) = load_local_field_recurrence(patient_who_recur, dataDirectory)
    plot_heat_map(meanMap, 'mean map - patients_who_recur')
    plot_heat_map(varMap, 'variance map - patients_who_recur')
    plot_heat_map(stdMap, 'standard deviation map - patients_who_recur')

if __name__ == '__main__':
    test_local_field()

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