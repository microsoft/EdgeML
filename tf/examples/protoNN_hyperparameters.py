import os
import pickle
from sklearn.grid_search import ParameterGrid

param_grid = {
"e" : [250],
"np" : [10,20,40,60,80,150],
"p" : [5,10,15,20],
"lr" : [0.05,0.01,0.1,0.0001]
}

grid = ParameterGrid(param_grid)
list_dicts = []
bestr2 = 0.0
for params in grid:
    os.system("python protoNN_example.py -dir ../../Inverter_1_2017-03-30"+" -e "+str(params["e"]) +" -np "+str(params["np"])+" -p "+str(params["p"])+" -lr "+str(params["lr"]) )
    ndict = pickle.load(open("protoNN_dict.pkl","rb"))
    print ("Parameters : ",ndict)
    curr2 = ndict.copy()
    curr2["e"] = params["e"]
    curr2["np"] = params["np"]
    #curr2["g"] = params["g"]
    curr2["p"] = params["p"]
    curr2["lr"] = params["lr"]
    list_dicts.append(curr2)
    #For each run, append the results and then pickle it on top. Useful when we want to stop early.
    #Change the directory as per the dataset.
    pickle.dump(list_dicts,open("../../Inverter_1_2017-03-30/protoNN_dicts.pkl","wb"))
    #+" -g "+str(params["g"])

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
