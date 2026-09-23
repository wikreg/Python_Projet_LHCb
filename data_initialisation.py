import uproot as ur
import pandas as pd

data = ur.open("data/MasterclassData_2012_all.root")

#Inspecting the different Tbranches of the TTree (Python Mapping)
print(data.keys())

#Inspecting the different keys of the two Tbranches
decay_trees = ur.open("data/MasterclassData_2012_all.root:DecayTree")
print(decay_trees.keys())

#Loading the two Tbranches into pandas DataFrames
decay_tree_1 = data["DecayTree;1"].arrays(library="pd")
decay_tree_2 = data["DecayTree;2"].arrays(library="pd")

#Inspecting the first few rows of the two DataFrames
print(decay_tree_1.head())
print()
print(decay_tree_2.head())

#They look the same but just in case we will keep them as two seperate DataFrames. 
#They might have different values that might be revelant to the study of D0.

#That concludes the first step to the data analisis.