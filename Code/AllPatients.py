import pandas as pd
from collections import namedtuple




'''
Group patients by recurrence 
returns named tuple of PatientsWhoRecur, PatientsWhoDontRecur from the 
'''
def recurrenceGroups(allPatients):
    # Group patients by recurrence
    AllPatientsGrouped = allPatients.groupby('Recurrence')
    PatientsWhoRecur = pd.concat([AllPatientsGrouped.get_group('1'), AllPatientsGrouped.get_group('YES')])
    PatientsWhoDontRecur = pd.concat([AllPatientsGrouped.get_group('0'), AllPatientsGrouped.get_group('censor'),
                                      AllPatientsGrouped.get_group('NO')])
    return PatientsWhoRecur, PatientsWhoDontRecur


class AllPatients:
    def __init__(self, dataDir, fileNames):
        filePath = r'%s/%s.csv' % (dataDir, fileNames[0])
        # TODO remove random sample
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
        return recurrenceGroups(self.allPatients)



def testIt():
    testAp = AllPatients(r"../Data/OnlyProstateResults/Global", ['AllData19Frac', 'AllData16Frac_old', 'AllData16Frac', 'AllData19Frac_old'])
    # Atlas or Corrupt
    atlas = {'200806930', '201010804', '201304169', '201100014', '201205737', '201106120', '201204091', '200803943',
             '200901231', '200805565', '201101453', '200910818', '200811563', '201014420'}
    corrupt19Frac = {}  # '196708754','200801658','201201119','200911702','200701370','200700427','200610929','200606193','200600383','200511824'
    corrupt16Frac = {}  # '200701370','200700427','200610929','200606193','200600383','200511824'

    testAp.removePatients(atlas)
    testAp.removePatients(corrupt19Frac)
    testAp.removePatients(corrupt16Frac)

    cleanedData = testAp.allPatients



if __name__ == '__main__':
    testIt()