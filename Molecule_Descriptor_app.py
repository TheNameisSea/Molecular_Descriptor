#-------------------- Imports --------------------#

import streamlit as st      # For deploying the program as a web app
import pandas as pd     # For working with dataframes
import numpy as np      # For working with arrays
from rdkit import Chem      # Chemical properties calculation
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
import requests     # For web scraping 


#-------------------- CONST --------------------#

CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"        # Link of the CACTUS website - a website that contains chemical stuff


#-------------------- Custom Functions --------------------#

# Function created based on:
#   dataprofessor (2020) solubility-app [Source code]. https://github.com/dataprofessor/solubility-app
# Added new algorithms compared to the source code to calculate: the H-bond acceptor, the H-bond donor and the Carbon proportion of the molecule 
def generateDescriptors(smiles, verbose=False):
    molecule_data= []      # A list for storing inputed molecules
    carbon = np.zeros(len(smiles))      # A list for storing the number of carbon in each molecule
    for element in smiles:      # Get element from the inputed Simplified Molecular Input Line Entry System
        molecule=Chem.MolFromSmiles(element)
        molecule_data.append(molecule)
        
        for atom in element:        # Count the number of carbon in a molecule
            if atom == "C":
                carbon[smiles.index(element)] += 1

    baseData = np.zeros(1)    # Create a base numpy array
    num_smiles = 0     # I is for recording the number of SMILES inputed
    
    for mol in molecule_data:
        aromatic_atoms = []     # A list for storing aromatic atoms in a molecule
        for atom in range(mol.GetNumAtoms()):
            aromatic_atoms.append(mol.GetAtomWithIdx(atom).GetIsAromatic())       # Get a list of aromatic atoms in the molecule
        aromatic_count = 0
        for i in aromatic_atoms:
            if i==True:
                aromatic_count += 1
        HeavyAtom = Descriptors.HeavyAtomCount(mol)     # Calculate the number of heavy atom
        AromaticProportion = aromatic_count/HeavyAtom       # Calculate the aromatic proportion

        LogPoctanol = Descriptors.MolLogP(mol)     # Calculate the partition coefficient or octanol
        MolecularWeight = Descriptors.MolWt(mol)     # Calculate the molecule's weight
        RotatableBonds = Descriptors.NumRotatableBonds(mol)     # Calculate the number of non-ring bond in the molecule
        CarbonProportion = carbon[molecule_data.index(mol)] / (len(smiles[molecule_data.index(mol)]) + 1)     # Calculate the carbon proportion 
        H_bond_donor = rdMolDescriptors.CalcNumHBD(mol)     # Calulate the Hydrogen bond donor
        H_bond_acceptor = rdMolDescriptors.CalcNumHBA(mol)      # Calulate the Hydrogen bond acceptor

        # Create an array with calculated values
        features = np.array([LogPoctanol,       
                        MolecularWeight,
                        RotatableBonds,
                        AromaticProportion,
                        CarbonProportion,
                        H_bond_donor,
                        H_bond_acceptor])

        if(num_smiles==0):
            baseData=features    # If there are only 1 inputed molecule then make the calculated values become the array
        else:
            baseData=np.vstack([baseData, features])     # If there are > 1 inputed molecule then add the calculated values as the next row of the array
        num_smiles += 1

    columnNames=["LogPoctanol","Molecular Weight","Rotatable Bonds","Aromatic Proportion", "Carbon Proportion","H-bond donor count", "H-bond acceptor count"]       # Columns name for pandas dataframe
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)       # Return the molecule's descriptor as a dataframe 

    return descriptors

#   Oliver Scott (2020) <answer for a question on stackoverflow>. https://stackoverflow.com/questions/64329049/converting-smiles-to-chemical-name-or-iupac-name-using-rdkit-or-other-python-mod
def smiles_to_iupac(smiles):
    rep = "iupac_name"      # Add this into the link after the molecule descriptor to find its iupac name on the website
    try:
        url = CACTUS.format(smiles, rep)    # Create an url from the CACTUS link, the smiles of the molecule with /iupac_name at the end 
        response = requests.get(url)        # Request data from the created url
        response.raise_for_status()     # Check for errors
        return response.text        # Return the response in unicode
    except:
        return "Unknown Molecule" 

def drawSmiles(smiles_list):
    molecular_models = [Chem.MolFromSmiles(x) for x in smiles_list]     # Create a list of molecules from the inputed smiles
    for m in range(len(molecular_models)):      
        try:
            fig = Draw.MolToMPL(molecular_models[m])        # Create a figure of the molecule
            st.write(f"{smiles_to_iupac(SMILES[m])}: {SMILES[m]}")      #Print the molecule's iupac name
            st.write("Molecular Model:")
            st.write(fig)       # Print the figure
            st.write("")
        except:
            st.write("An exception occured")


#-------------------- Main --------------------#

st.set_page_config(page_title="Molecular Descriptors", page_icon="🔬", layout="centered", initial_sidebar_state="expanded", menu_items=None)

# Page Title
st.write("""
# Molecular Descriptor Web App
This app calculates the **Molecular Descriptors** values of molecules!
""")

# Input molecules (Side Panel)
st.sidebar.header('User Input Features')

# Read SMILES input
SMILES_input = "CC(=O)OC1=CC=CC=C1C(=O)O\nC(C(=O)O)C(CC(=O)O)(C(=O)O)O\nC1=CC=C(C=C1)N"

SMILES = st.sidebar.text_area("Input SMILES", SMILES_input)     
SMILES = SMILES.split('\n')

st.header('Input SMILES')
st.write(SMILES) 

# Calculate molecular descriptors
st.header('Computed molecular descriptors')
try:
    X = generateDescriptors(SMILES)
    st.write(X)
except:
    st.write("Invalid SMILES")

# Draw the molecule
st.header('Molecular model')
drawSmiles(SMILES)
