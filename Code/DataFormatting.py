"""
A file where we shall conduct all the re-formatting of our data
"""

import xdrlib


def data_frame_to_XDR(patient_map, map_name):
    """
    A function to convert a pandas data frame to XDR file format.
    :param patient_map: A pandas data frame of the patient map to convert to XDR format
    :param map_name: A string which will be your converted file name, must be the patient ID.
    """

    p = xdrlib.Packer()
    p.pack_farray(7200, patient_map.flatten(), p.pack_double)

    newFile = open('../Data/Deep_learning_results/mapsXDR/%s.xdr' % (map_name), "wb")
    newFile.write(p.get_buffer())
    newFile.close()
