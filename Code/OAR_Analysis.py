# -*- coding: utf-8 -*-
"""
05/01/2019
Created by: Tom & Alex
"""

# import librarys
import numpy as np
import pandas as pd
from LocalFilter import plot_heat_map_np

def set_voxel_value(np_map, voxel_threshhold_value, set_value, other_value):
	'''
	Takes in DSC cuts heat map which is 60 x 120 and sets all value below DSC cuts threshold to one value defined by set value and
	remaining voxels to anothother value
	:param np_map: heat map
	:param voxel_threshhold_value: threshold condition
	:param set_value: Value want the rest of the map to be set as
	:param other_value:
	:return: Returns the np map with only two values the set_value and the other_value
	'''
	for x in range(0, 60):
		for y in range(0, 120):
			# Set Bladder to 0
			if np_map[x, y] < voxel_threshhold_value:
				np_map[x, y] = set_value
			else:
				np_map[x, y] = other_value

	return np_map


def magnify_map(np_map, enlargement):
	'''
	Enlarges voxels by DSC cuts factor to increase pigment contrast
	:param np_map: local heat map
	:param enlargement: factor of increase
	:return: local heat map with voxels increased by DSC cuts factor of enlargment
	'''
	for x in range(0, 60):
		for y in range(0, 120):
			np_map[x, y] = enlargement * np_map[x, y]

	return np_map


def load_oar_set_colours():
	'''
	Loads all OARs andset there maps fixed colour
	:return: all OARs and Prostate map with block colours
	'''
	dataDir = r'../Data/OARlocations/'
	# Load Patients
	patient_bladder_map = pd.read_csv(dataDir + "bladder.csv", header=None).as_matrix()
	patient_sv_map = pd.read_csv(dataDir + "SemiV_ADMIRE.csv", header=None).as_matrix()
	patient_rectum_map = pd.read_csv(dataDir + "rectum.csv", header=None).as_matrix()
	patient_prostate_map = pd.read_csv(dataDir + "prostate.csv", header=None).as_matrix()

	# Set Voxel Colours
	set_patient_bladder_map = set_voxel_value(patient_bladder_map, 40, 100, -100)
	set_patient_sv_map = set_voxel_value(patient_sv_map, 40, 50, -100)
	set_patient_rectum_map = set_voxel_value(patient_rectum_map, 40, 75, -100)
	set_patient_prostate_map = magnify_map(patient_prostate_map, 20)

	return set_patient_bladder_map, set_patient_sv_map, set_patient_rectum_map, set_patient_prostate_map


def set_all_oar():
	'''
	:return: container with OARs all set to parameters defined in load_oar_set_colours()
	'''

	# OARcontainer = np.zeros((60, 120, 4))# Recurrence container
	OARcontainer = np.zeros((4, 60, 120))
	(bladder, sv, rectum, prostate) = load_oar_set_colours()
	OARcontainer[0, :, :] = bladder
	OARcontainer[1, :, :] = sv
	OARcontainer[2, :, :] = rectum
	OARcontainer[3, :, :] = prostate

	return OARcontainer


def merge_all_oars(container):
	'''
	:param container: 3d np array that contains all oars with set values
	:return: returns DSC cuts 2d map containing all oars
	'''
	oar_map = np.zeros((60, 120))
	for i in  range(0,4):
		oar_map = oar_map + container[i, :, :]
	return oar_map


def test_oar_map_plot():
	maps_of_oars = set_all_oar()
	merge_oars = merge_all_oars(maps_of_oars)

	# Plots of organs
	plot_heat_map_np(maps_of_oars[0], "Bladder")
	plot_heat_map_np(maps_of_oars[1], "Seminal Vesicles")
	plot_heat_map_np(maps_of_oars[2], "Rectum")
	plot_heat_map_np(maps_of_oars[3], "Prostate")
	plot_heat_map_np(merge_oars,"contains all oars in prostate")


if __name__ == '__main__':
	test_oar_map_plot()
