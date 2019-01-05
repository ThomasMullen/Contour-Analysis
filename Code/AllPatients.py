import pandas as pd
from collections import namedtuple

def separate_by_recurrence(all_patients):
    """
    :param all_patients: The global data of all patients
    :return: group of a global set with patients that have had a
    recurrence in prostate cancer and a global set with no recurrence

    """
    PatientsWhoRecur = pd.concat(
        [all_patients.groupby('Recurrence').get_group('1'), all_patients.groupby('Recurrence').get_group('YES')])
    PatientsWhoDontRecur = pd.concat(
        [all_patients.groupby('Recurrence').get_group('0'), all_patients.groupby('Recurrence').get_group('censor'),
         all_patients.groupby('Recurrence').get_group('NO')])
    return PatientsWhoRecur, PatientsWhoDontRecur


def global_remove_stageT3(all_patients):
    """
    :param all_patients: The global data of all patients including late stages
    :return: returns the global patient data with stages >T3 removed
    """

    late_staged_patients = ('T3b', 'T3b/T4', 'T4', 'T3B')
    early_staged_patients = all_patients[~all_patients['Stage'].isin(late_staged_patients)]
    return early_staged_patients

class AllPatients:
    def __init__(self, dataDir, fileNames):
        filePath = r'%s/%s.csv' % (dataDir, fileNames[0])
        dataFrames = [pd.read_csv(r'%s/%s.csv' % (dataDir, x)) for x in fileNames]
        self.allPatients = pd.concat(dataFrames)
        self.originalAllPatients = self.allPatients

    def removePatients(self, patientSetToBeRemoved):
        self.allPatients = self.allPatients[~self.allPatients['patientList'].isin(patientSetToBeRemoved)]

    '''
    Group patients by recurrence 
    returns named tuple of PatientsWhoRecur, PatientsWhoDontRecur from the 
    '''
    def recurrenceGroups(self):
        return separate_by_recurrence(self.allPatients)

    def remove_stageT3(self):
        return global_remove_stageT3(self.allPatients)

def testIt():
    testAp = AllPatients(r"../Data/OnlyProstateResults/Global", ['AllData19Frac', 'AllData16Frac_old', 'AllData16Frac', 'AllData19Frac_old'])
    # Atlas or Corrupt
    atlas = {'200806930', '201010804', '201304169', '201100014', '201205737', '201106120', '201204091', '200803943',
             '200901231', '200805565', '201101453', '200910818', '200811563', '201014420'}
    corrupt19Frac = {200710358,200705181}  # '196708754','200801658','201201119','200911702','200701370','200700427','200610929','200606193','200600383','200511824'
    corrupt16Frac = {}  # '200701370','200700427','200610929','200606193','200600383','200511824'

    testAp.removePatients(atlas)
    testAp.removePatients(corrupt19Frac)
    testAp.removePatients(corrupt16Frac)

    cleanedData = testAp.allPatients



if __name__ == '__main__':
    testIt()