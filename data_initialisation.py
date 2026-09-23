import uproot as ur
import pandas as pd

data = ur.open("data/MasterclassData_2012_all.root")

# Inspecting the different Tbranches of the TTree (Python Mapping)
print(data.keys())

# Inspecting the different keys of the two Tbranches
decay_trees = ur.open("data/MasterclassData_2012_all.root:DecayTree")
print(decay_trees.keys())

# Loading the two Tbranches into pandas DataFrames
decay_tree_1 = data["DecayTree;1"].arrays(library="pd")
decay_tree_2 = data["DecayTree;2"].arrays(library="pd")

# Inspecting the first few rows of the two DataFrames
print(decay_tree_1.head())
print()
print(decay_tree_2.head())

#They look the same but just in case we will keep them as two seperate DataFrames. 
#They might have different values that might be revelant to the study of D0.

#That concludes the first step to the data analisis.


# For your information this part is coded in pair with AI that explains the use of a dictionary and the try/except function.
# Creation of the function "charge_donnees"
def charge_donnees(name_of_file):

    # We will considere that every root file that we want to convert into a pandas DataFrame will be in the 'data' folder.
    # Opening the root file with the given name
    file = ur.open("data/" + name_of_file + ".root")

    # Creation of an empty dictionary to store the Trees and their data
    trees = {}

    # Loop to be sure that we dont forget any Trees from the root file
    for keys in file.keys():
        print(f"loading {keys}")

        # Try/Except function to skip the data that is not TTrees and load the data that is a TTree
        try:
            trees[keys] = file[keys].arrays(library="pd")
        except:
            print(f"Skipping {keys} - it is not a TTree")

    # print to see what the data looks like
    # Here i asked AI to help me make it more aestheticaly pleasing
    for tree_name, df in trees.items():
        print(f"\n--- Data for {tree_name} ---")
        print(df.head())
        print()
    
    return trees

# Testing the 'charge_donnees' function with the given data set for the project
charge_donnees("MasterclassData_2012_all")

# Both the first part of my code that i coded alone and the part coded with AI do basicaly the same thing. The only exception is that the AI part can be applied to any root file.

