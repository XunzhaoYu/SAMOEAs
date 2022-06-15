# -*- coding: UTF-8 -*-
import yaml
import xlrd
from time import time

from problems.DTLZ.DTLZ import *
#from SAMOEAs.ParEGO.ParEGO import *
from SAMOEAs.MOEADEGO.MOEADEGO import *
#from SAMOEAs.OREA.OREA import *
#from tools.data_IO import load_PF

desired_width = 160
np.set_printoptions(linewidth=desired_width)
np.set_printoptions(precision=4, suppress=True)


""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Last update: 2022-June-15.
MOSAEAs Tester for DTLZ benchmark functions.
"""
cfg_filename = 'config_DTLZ.yml'
with open(cfg_filename,'r') as ymlfile:
    config = yaml.load(ymlfile)

name = 'DTLZ1'
dataset = DTLZ1(config)

# get the Pareto Front of DTLZ
"""
pf = load_PF(name)
"""
pf_path = config['path_pf'] + name + " PF " + str(config['y_dim']) + "d "+str(5000)+".xlsx"
pf_data = xlrd.open_workbook(pf_path).sheets()[0]
n_rows = pf_data.nrows
pf = np.zeros((n_rows, config['y_dim']))
for index in range(n_rows):
    pf[index] = pf_data.row_values(index)
#"""

iteration_max = 1
for iteration in range(0, iteration_max):
    time1 = time()
    current_iteration = str(iteration + 1).zfill(2)
    #alg = ParEGO(config, name, dataset, pf, b_default_path=True)
    alg = MOEADEGO(config, name, dataset, pf, b_default_path=True)
    #alg = OREA(config, name, dataset, pf, b_default_path=True)
    alg.run(current_iteration)
    t = time() - time1
    print("run time: {:.0f} mins, {:.2f} secs.".format(t // 60, t % 60))
    solution, minimum = alg.get_result()
    print("solution: ", type(solution))
    print(solution)
    print("minimum: ", type(minimum))
    print(minimum)

