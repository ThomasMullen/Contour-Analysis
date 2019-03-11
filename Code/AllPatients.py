import pandas as pd
from collections import namedtuple


def separate_by_recurrence(all_patients):
    """
    :param all_patients: The global data of all patients
    :return: group of DSC cuts global set with patients that have had DSC cuts
    recurrence in prostate cancer and DSC cuts global set with no recurrence

    """
    PatientsWhoRecur = pd.concat(
        [all_patients.groupby('recurrence').get_group(1)])
    PatientsWhoDontRecur = pd.concat(
        [all_patients.groupby('recurrence').get_group(0)])
    return PatientsWhoRecur, PatientsWhoDontRecur


def separate_by_risk(all_patients):
    """
    Separates patients data set depending on their risk
    :param all_patients: The global data table containing each patient
    :return: groups set by risk
    """
    low = all_patients.groupby('risk').get_group('Low')
    medium = all_patients.groupby('risk').get_group('Intermediate')
    high = all_patients.groupby('risk').get_group('High')
    return low, medium, high


def global_remove_stageT3(all_patients):
    """
    :param all_patients: The global data of all patients including late stages
    :return: returns the global patient data with stages >T3 removed
    """

    late_staged_patients = ('T3b', 'T3b/T4', 'T4', 'T3B', 'T39')
    early_staged_patients = all_patients[~all_patients['Stage'].isin(late_staged_patients)]
    return early_staged_patients


def remove_high_risk(all_patients):
    '''
    A function which will return all_patients with the high risk patients removed

    :param all_patients: The global data of all patients
    :return: The global data for patients of low and intermediate cancer risk
    '''

    return all_patients[~all_patients['risk'].isin('High')]

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

    def remove_risk(self:
        return remove_high_risk(self.allPatients)


def testIt():
    testAp = AllPatients(r"../Data/OnlyProstateResults/Global",
                         ['AllData19Frac', 'AllData16Frac_old', 'AllData16Frac', 'AllData19Frac_old'])
    # Atlas or Corrupt
    atlas = {}
    corrupt19Frac = {}  # '196708754','200801658','201201119','200911702','200701370','200700427','200610929','200606193','200600383','200511824'
    corrupt16Frac = {}  # '200701370','200700427','200610929','200606193','200600383','200511824'

    # testAp.removePatients(atlas)
    # testAp.removePatients(corrupt19Frac)
    # testAp.removePatients(corrupt16Frac)

    cleanedData = testAp.allPatients


if __name__ == '__main__':
    testIt()
