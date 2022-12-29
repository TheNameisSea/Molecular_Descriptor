#-------------------- General Info --------------------#
# Project name: Molecular Descriptor app
# Author: Do Chi Hai - github: https://github.com/TheNameisSea
# Functions: Calculate the molecular descriptors of molecules, draw 2d/3d models of molecules
# Github: https://github.com/TheNameisSea/Molecular_Descriptor.git

#-------------------- Imports --------------------#

import streamlit as st      # For deploying the program as a web app
import pandas as pd     # For working with dataframes
import numpy as np      # For working with arrays
from rdkit import Chem      # Chemical properties calculation
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from stmol import showmol
import py3Dmol
from rdkit.Chem import AllChem
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
    num_smiles = 0     # This is for recording the number of smiles processed
    
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

#   Oliver Scott (2020) <answer to a question on stackoverflow>. https://stackoverflow.com/questions/64329049/converting-smiles-to-chemical-name-or-iupac-name-using-rdkit-or-other-python-mod
def smiles_to_iupac(smiles):
    rep = "iupac_name"      # Add this into the link after the molecule descriptor to find its iupac name on the website
    try:
        url = CACTUS.format(smiles, rep)    # Create an url from the CACTUS link, the smiles of the molecule with /iupac_name at the end 
        response = requests.get(url)        # Request data from the created url
        response.raise_for_status()     # Check for errors
        return response.text        # Return the response in unicode
    except:
        return "Unknown Molecule" 

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():     # Get atoms
        atom.SetAtomMapNum(atom.GetIdx())       #Get index of atoms
    return mol

def displaySmiles(smiles_list, include_index):
    molecular_models = [Chem.MolFromSmiles(x) for x in smiles_list]     # Get smiles list
    if include_index:
        for m in range(len(molecular_models) - 1):      # len must -1 for skipping the 1st dummy item
            fig = Draw.MolToMPL(mol_with_atom_index(molecular_models[m+1]))     # Draw molecule with index as matplotlib fig, index must +1 for skipping the 1st dummpy item
            st.write(f"{smiles_to_iupac(smiles_list[m+1])}: {smiles_list[m+1]}")      # Print iupac name and smiles
            st.write("Molecular Model:")
            st.write(fig)       
            st.write("")        # Spaces for the aestheticity:)))
    else:
         for m in range(len(molecular_models) - 1):
            fig = Draw.MolToMPL((molecular_models[m+1]))
            st.write(f"{smiles_to_iupac(smiles_list[m+1])}: {smiles_list[m+1]}")
            st.write("Molecular Model:")
            st.write(fig)
            st.write("") 

#   napoles-uach (2021) Medium_Mol (app4) [Source code]. https://github.com/napoles-uach/Medium_Mol
def render_mol(smi):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))       # Get mol from smiles and add Hydrogens if needed
    AllChem.EmbedMolecule(mol)      # Prepare the molecule
    block = Chem.MolToMolBlock(mol)     # Turn mol to block

    view = py3Dmol.view(width=350, height=350)      # Create a view (place holder) for the molecule
    view.addModel(block,'mol')      # Add the molecule to the view
    view.setStyle({'stick':{}})     # Styling
    view.setBackgroundColor('white')    
    view.zoomTo()       # Focus on the molecule
    showmol(view, height=350, width=350)        # Display 


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
SMILES_input = "CC(=O)OC1=CC=CC=C1C(=O)O\nC(C(=O)O)C(CC(=O)O)(C(=O)O)O\nC1=CC=C(C=C1)N"     # Example inputs

SMILES = st.sidebar.text_area("Input SMILES", SMILES_input)         # Create an input area with the example inputs
SMILES = "C\n" + SMILES # Adds C as a dummy first item to prevent rdkit's errors
SMILES = SMILES.split('\n')     # Seperate items in the list by "\n"

display = st.sidebar.checkbox('Display molecular structures')       # Check if want to display model

if display:
    include_index = st.sidebar.checkbox('Display indexes')      # Check if want to display atoms' indexes
    multidimensional = st.sidebar.checkbox('Display 3d model of molecules')     # Check if want to display 3d model

st.header('Input SMILES')
st.write(SMILES[1:])        # Display inputs, skip the dummy first item

# Calculate molecular descriptors
st.header('Computed molecular descriptors')
try:
    X = generateDescriptors(SMILES)
    st.write(X[1:])     # Display molecular descriptors, skip the dummy first item
except:
    st.write("Invalid SMILES")

# Draw the molecule

if display and multidimensional:
    st.header('Molecular model')
    col1, col2 = st.columns(2)
    with col1:
        displaySmiles(SMILES, include_index)        # Display 2d models
    with col2:
        for smiles in SMILES[1:]:
            if SMILES.index(smiles) == 1:
                st.write("")        # Spaces for the aestheticity:)))
                st.write("") 
                st.write("") 
                st.write("")
                st.write("")
                st.write("")
            else:
                st.write("") 
                st.write("") 
                st.write("") 
                st.write("")

            st.write(render_mol(smiles))        # Display 3d models
elif display:
    st.header('Molecular model')
    displaySmiles(SMILES, include_index)
