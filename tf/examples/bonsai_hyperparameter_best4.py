from __future__ import print_function
import os
import pickle
from sklearn.grid_search import ParameterGrid

import bonsaipreprocess
import tensorflow as tf
import numpy as np
import multiprocessing
import os
import sys
import pickle
#sys.path.insert(0, '../')

param_grid = {
"e" : [42],
"d" : [1,2,4,6,7],
"p" : [10,15,20],
"b"  : [32,64,128],
"s" : [3.0],
"lr" : [0.01,0.05],
"rZ" : [0.001, 0.0001, 0.000001],
"rW" : [0.01, 0.001, 0.00001]
}

grid = ParameterGrid(param_grid)
list_dicts = []
bestr2 = 0.0
curr2 = {}
f = open("bonsai_hyperparameters_s4.sh","w+")

for params in grid:
    f.write("python bonsai_example_s4.py -dir ../../Inverter_1_2017-03-30 "+ "-e " + str(params["e"]) + " -d " + str(params["d"]) + " -p " + str(params["p"])
     + " -b " + str(params["b"]) + " -s " + str(params["s"]) + " -lr " + str(params["lr"]) + " -rZ " + str(params["rZ"]) + " -rW " + str(params["rW"]) )
    f.write("\n")
f.close()
'''
#os.system("python bonsai_example.py -dir ../../Inverter_1_2017-03-30"+" -d "+str(params["d"])+" -b "+str(params["b"])+" -s "+str(params["s"])+" -lr "+str(params["lr"])+" -rZ "+str(params["rZ"])+" -rW "+str(params["rW"]))
vals = {"e" : str(params["e"]) ,"d" : str(params["d"]) , "b" : str(params["b"]) , "s" : str(params["s"]) , "lr" : str(params["lr"]) , "p" : str(params["p"]) }#, "rW" : str(params["rW"]), "p" : str(params["p"]) }
curr2 = model_create(vals)
curr2["e"] = params["e"]
curr2["d"] = params["d"]
curr2["p"] = params["p"]
curr2["b"] = params["b"]
curr2["lr"] = params["lr"]
curr2["s"] = params["s"]
#curr2["rZ"] = params["rZ"]
#curr2["rW"] = params["rW"]
print ("Parameters : ",curr2,file = sys.stdout)
list_dicts.append(curr2)
#For each run, append the results and then pickle it on top. Useful when we want to stop early.
#Change the directory as per the dataset.
print ("Dumping !!")
pickle.dump(list_dicts,open("../../Inverter_1_2017-03-30/list_dicts-s1.pkl","wb"))
'''
'''
print ("---------------------------")
print ("End of Param Search : ")
best_params = {}
for i in range(len(list_dicts)):
    if(i==0):
        best_params = list_dicts[i].copy()
        continue
    #Focus is only on RMSE.
    if (list_dicts[i]["rmse"] < best_params["rmse"]):
        best_params = list_dicts[i].copy()

print ("Best so far : ",best_params)

sys.stdout.close()
'''
