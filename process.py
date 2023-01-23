import os
import numpy as np
from tqdm import tqdm
from dscribe.descriptors import SOAP
from pymatgen.io.ase import AseAtomsAdaptor
import torch
from torch.utils.data import TensorDataset, Dataset

# Function to load data from file
def load_data(path):
    # Exit code if MP database does not exist
    if not os.path.exists(path):
        print('MP database not found')
        exit()

    # Load data from file
    print('Loading data...')
    data=np.load(path, allow_pickle=True).item()    # allow_pickle=True is required to load the pickled objects. The data is stored as numpy array, so we use .item() to convert it to its original type.
    # data is a dictionary with material IDs as keys and lists of properties as values
    return data 

# Function to get list of all material IDs with valid piezoelectricity values
def get_piezo_ids(data):
    # Get list of all material IDs with valid piezoelectricity values
    ids_piezo=[]
    print('Selecting materials...')
    for key in tqdm(data.keys()):
        if data[key][5] is not None:
            ids_piezo.append(key)

    print('Number of materials: ', len(ids_piezo))

    return ids_piezo

# Function to get piezoelectricity values
def get_targets(data, ids):
    # Add piezoelectricity values to dictionary
    targets=[]
    for key in ids:
        targets.append(data[key][5])
    
    return targets

# Function to keep only the perovskites in data
def get_perovskites(X, Y, ids, formula_pretty, formula_anonymous):
    indices=[]
    for i in range(len(X)):
        if formula_anonymous[i] == 'ABC3' and formula_pretty[i][-2:] == 'O3':
            indices.append(i)

    X=X[indices]
    Y=Y[indices]
    ids=ids[indices]
    return X, Y, ids

# Function to get SOAP descriptors
def get_soap_descriptors(data, ids, processing_args):
    # Get list of all elements
    elements=[]
    for key in ids:
        elements+=data[key][0]
    elements=list(set(elements))     # Remove duplicates

    # Convert elements to strings
    elements=[str(element) for element in elements]
    print('Number of elements: ', len(elements))
    print('Elements: ', elements)

    # Make SOAP descriptors of structures
    soap=SOAP(species=elements, rcut=processing_args["SOAP_rcut"], nmax=processing_args["SOAP_nmax"], lmax=processing_args["SOAP_lmax"], sigma=processing_args["SOAP_sigma"], average='inner', crossover=False, sparse=False)
    print('Making SOAP descriptors...')
    descriptors=[]
    for key in tqdm(ids):
        atoms=AseAtomsAdaptor().get_atoms(data[key][4]) # Create ase Atoms object from pymatgen Structure object
        descriptor=soap.create(atoms)   # Make SOAP descriptor
        descriptors.append(descriptor)     # Add descriptor to list
    print('Number of descriptor features: ', len(descriptors[0]))
    return descriptors

# Function to get data
def get_data(processing_args):
    if processing_args['reprocess'] or not os.path.exists(os.path.join(processing_args['data_path'], 'processed/data.npz')):
        # Load data
        data=load_data(os.path.join(processing_args['data_path'], 'MP_full.npy'))
        ids=get_piezo_ids(data)

        # Get SOAP descriptors
        X=get_soap_descriptors(data, ids, processing_args)

        # Get targets
        Y=get_targets(data, ids)

        # Get formula pretty
        formula_pretty=[]
        for key in ids:
            formula_pretty.append(data[key][1])

        # Get formula anonymous
        formula_anonymous=[]
        for key in ids:
            formula_anonymous.append(data[key][2])

        # Convert to numpy arrays
        X=np.array(X)
        Y=np.array(Y)
        ids=np.array(ids)
        formula_pretty=np.array(formula_pretty)
        formula_anonymous=np.array(formula_anonymous)

        # Create folder for processed data if it does not exist
        if not os.path.isdir(os.path.join(processing_args['data_path'], 'processed')):
            os.mkdir(os.path.join(processing_args['data_path'], 'processed'))

        # Save as npz file
        print('Saving data...')
        np.savez(os.path.join(processing_args['data_path'], 'processed/data.npz'), X=X, Y=Y, ids=ids, formula_pretty=formula_pretty, formula_anonymous=formula_anonymous)

    else:
        # Load data
        print('Loading data...')
        data=np.load(os.path.join(processing_args['data_path'], 'processed/data.npz'), allow_pickle=True)   # allow_pickle=True is required to load the pickled objects.
        X=data['X']         # The array is accessed using the key 'X'. 
        Y=data['Y']
        ids=data['ids']
        formula_pretty=data['formula_pretty']
        formula_anonymous=data['formula_anonymous']

    if processing_args['perovskite_only']:
        X, Y, ids=get_perovskites(X, Y, ids, formula_pretty, formula_anonymous)

    return X, Y, ids

"""
What is the difference between np.save and np.savez?

np.save is used to save a single array. np.savez is used to save multiple arrays into a single file. The arrays are saved as a dictionary. The keys to the different arrays can be assigned by the user at the time of saving.
"""