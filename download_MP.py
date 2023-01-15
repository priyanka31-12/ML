from mp_api.client import MPRester  # Library for accessing MP database
import os                           
from tqdm import tqdm               # Library for showing progress bar
import numpy as np                  
import argparse                     # Library for parsing arguments



"""
How to use:
    "python download_MP.py" : Download MP database if it does not exist
    "python download_MP.py --refresh" : Download MP database and overwrite existing database

Confused? Check out Part 1 to see how the argument thing works.
"""



# Part 1:
parser = argparse.ArgumentParser(description='Download MP database')    # Create argument parser

parser.add_argument('--refresh', action='store_true', help='Refresh MP database')   # Add argument to refresh MP database. action='store_true' means that if the argument is present, the value is True. Otherwise, the value is False.
args=parser.parse_args()    # Parse arguments. 'args' now stores 'refresh=True' or 'refresh=False' based on the presence of the argument.

"""
Arguments are stored in args as attributes. For example, if the argument is '--refresh', then args.refresh will be True or False.
"""



# Part 2:
# Exit code if MP database already exists and refresh is not requested
if os.path.exists('MP_full.npz') and not args.refresh:
    print('MP database already downloaded')
    exit()



# Part 3:
key=os.environ['MP_API_KEY']    # Get API key for materials project from environment variables. This is required to access the MP database using the API.
mpr=MPRester(key)               # Create MP API rester object. This is used to access the MP database.
MP_full=mpr.summary.search()    # Get all materials from MP. This is a list of material objects.
print(MP_full[0])               # Print first material in database.



# Part 4:
MP_full_dict={}                 # Create empty dictionary.
print('Processing...')
# tqdm is used with for loops to show a progress bar. It is not necessary, but it looks cool :) 
for material in tqdm(MP_full):
    # Add material to dictionary with material ID and list of properties as values
    MP_full_dict[material.material_id]=[material.elements, material.formula_pretty, material.formula_anonymous, material.symmetry, material.structure, material.e_ij_max]
print('Saved...')
np.save('data/MP_full.npy', MP_full_dict)  # Save dictionary to file
print('MP database downloaded')

"""
np.save saves the data as a numpy array.

Elements such as 'structure' are objects. How are objects stored?

np.save has a keyword argument allow_pickle=True. pickle is used to serialize objects and store them.

When we load them later using numpy.load, we can use the keyword argument allow_pickle=True to load the objects. By default, allow_pickle=False for np.load.
"""



"""
Checklist (update to True if you have understood the code):
Part 1: True
Part 2: True
Part 3: True
Part 4: True

Understood it all? Give yourself a pat on the back and move on to process.py.
"""