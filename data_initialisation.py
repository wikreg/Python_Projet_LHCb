import uproot as ur
import pandas as pd
import zfit as zf
# I decided to use plotly because of how easy it is to interact with the plots and read them. I used it previously for an internship
import plotly.express as px
import os

data = ur.open("data/MasterclassData_2012_all.root")

# Inspecting the different Tbranches of the TTree (Python Mapping)
print(data.keys())
print()

# Inspecting the different keys of the two Tbranches
decay_trees = ur.open("data/MasterclassData_2012_all.root:DecayTree")
print(decay_trees.keys())
print()

# Loading the two Tbranches into pandas DataFrames
decay_tree_1 = data["DecayTree;1"].arrays(library="pd")
decay_tree_2 = data["DecayTree;2"].arrays(library="pd")

# Inspecting the first few rows of the two DataFrames
print(decay_tree_1.head())
print()
print(decay_tree_2.head())
print()

#They look the same but just in case we will keep them as two seperate DataFrames. 
#They might have different values that might be revelant to the study of D0.

#That concludes the first step to the data analisis.


# For your information this part is coded in pair with AI that would explain the use of more advanced techniques like a dictionary and the try/except function.
# But I still typed everything myself to make sure i understand the code and slightly memorise it.

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

# Loading the trees into a new variable
Trees = charge_donnees("MasterclassData_2012_all")

# Both the first part of my code that i coded alone and the part coded with AI do basicaly the same thing. The only exception is that the AI part can be applied to any root file.

                                        # Creation of the function 'dessine_branche' that takes a DataFrame and a branch name and saves a histogram of it.

def dessine_branche(df, branch_name):

    # Checking if the branch name is in the data frame and saving a histogram of it
    if branch_name in df.columns:
        # Creating a histogram using plotly express
        fig = px.histogram(df, 
                           x=branch_name, 
                           title = f"Histogram of {branch_name}", 
                           color_discrete_sequence = ['blue'])

        # Updating the lables
        fig.update_layout(xaxis_title=branch_name, 
                          yaxis_title="Frequency", 
                          template="plotly_white")

        # Saving Histogram as a web page
        html_filename = f"histogram_{branch_name}.html"
        # Making it so that the histograms html files are saved in the folder "histograms_plots"
        full_path = os.path.join("histograms_plots", html_filename)
        fig.write_html(full_path)

    else:
        # Error if wrong branch name
        print(f"Branch '{branch_name}' not found.")


# Testing the 'dessine_branche' function

df_decay = Trees.get("DecayTree;2") # DecayTree 2 is more up to date so we will be using this one

# Checking what branches are in the TTree
print(df_decay.columns) # This gives me the info that i can make all of the histograms by using this as the name of the differents branches

# # checking if the data frame exists and making the histograms
if df_decay is not None:
    for branch in df_decay.columns:
        dessine_branche(df_decay, branch)

