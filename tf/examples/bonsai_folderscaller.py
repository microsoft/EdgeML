"""
Iterate through a folder, containing a set of folders, with each containing a train and test npy, array.

Usage :
Given a directory containing a set of folders, of inverter data, running this script with the path of the directory, will
run the Bonsai model, for each and every inverter data folder, and save the results, in the same folder.

Command Line Argument to be given : Name of the directory.
"""

import os
import sys

contents = os.listdir("../../"+sys.argv[1]+"/")

path = "../../"+sys.argv[1]+"/"
contents = [directory for directory in os.listdir(path) if os.path.isdir(path+directory)]
for x in contents:
    print ("\n\n")
    print ("---------------------------------------------------")
    print ("Dataset : ",x)
    print (os.system("python ../../tf/examples/bonsai_example.py -dir " + "../../"+ sys.argv[1]+"/"+x.split("/")[-1]+ " -p 15 -b 32 -e 1500 -d 1"))
