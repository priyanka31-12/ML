import numpy as np
from sklearn import svm
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import plotly.express as px
import random
import yaml
import torch
from torch.utils.data import random_split
import torch.nn as nn
import argparse
import process
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import time
import torch.nn.functional as F
import copy
from sklearn.model_selection import GridSearchCV, cross_val_score

# Arguments parsing
parser=argparse.ArgumentParser(description='Run model on MP database')
parser.add_argument('--run_mode', type=str, help='Run mode: Training, CV. Default: Training')
parser.add_argument('--model', type=str, help='Model to use: SVM, NN. Default: NN')
parser.add_argument('--modify', action='store_true', help='Modify data. Default: False')
parser.add_argument('--transform_mode', type=str, help='Transformation mode: log, root, none. Default: none')
parser.add_argument('--reps', type=int, default=1, help='Number of repetitions. Default: 1')
parser.add_argument('--verbose', action='store_true', default=True, help='Verbose mode. Default: True')
parser.add_argument('--reprocess', action='store_true', help='Reprocess data. Default: False')
args=parser.parse_args()

# Load config file
config=yaml.load(open('config.yml', 'r'), Loader=yaml.FullLoader)

# Modify config file based on arguments

"""
First we check if an argument for run_mode is present by comparing it with None (the argument defaults to None if it isn't provided). If it is present, we overwrite the value in the config file.

We store the value of run_mode in a variable ("Training"/"CV") in #2.

Initial state of config["Job"]:
{"Job": {"run_mode": "Training", "Training": {"model": "NN", "seed": 0}, "CV": {"model": "NN", "cv_folds": 10, "seed": 0}}}

We need only those values which correspond to the run_mode. So we overwrite the value of config["Job"] with the value of config["Job"][run_mode] in #3.

Final state of config["Job"] after #3:
{"Job": {"model": "NN", "seed": 0}} (if run_mode="Training")
{"Job": {"model": "NN", "cv_folds": 10, "seed": 111111}} (if run_mode="CV")
"""
#1
if args.run_mode != None:
    config['Job']['run_mode']=args.run_mode
#2
run_mode=config['Job']['run_mode']
#3
config['Job']=config['Job'][run_mode]

# Same thing we did for run_mode, we do for model.
if args.model != None:
    config['Job']['model']=args.model
model=config['Job']['model']
config['Model']=config['Model'][model]

# Modify config file based on arguments
if args.modify != None:
    config['Processing']['modify']=args.modify
if args.transform_mode != None:
    config['Processing']['transform_mode']=args.transform_mode
if args.reprocess != None:
    config['Processing']['reprocess']=args.reprocess
if config['Job']['seed']==0:
    config['Job']['seed']=random.randint(0, 100000) # Generate a random seed if seed=0

# Print config file
print(config)

# Function to plot histogram for e_ij_max values
def visualize_data(Y):
    print('Visualizing data...')
    fig=px.histogram(x=Y, nbins=100, title='Distribution of e_ij_max values')
    fig.show()

# Function to transform data
def transform_data(Y, mode=None):

    # Apply log transformation to e_ij_max values
    if mode=='log':
        try:
            Y=torch.log(Y+1e-6) # Add a small value to avoid log(0).
        except:
            Y=np.log(Y+1e-6)    # We use numpy.log() if torch.log() fails since torch.log() works only on torch tensors.
    
    # Apply square root transformation to e_ij_max values
    if mode=='root':
        try:
            Y=torch.sqrt(Y)
        except:
            Y=np.sqrt(Y)    # We use numpy.sqrt() if torch.sqrt() fails since torch.sqrt() works only on torch tensors.

    return Y

# Function to revert the transformation
def revert_transform(Y, mode=None):

    # Revert log transformation to e_ij_max values
    if mode=='log':
        try:
            Y=torch.exp(Y)-1e-6
        except:
            Y=np.exp(Y)-1e-6    # We use numpy.exp() if torch.exp() fails since torch.exp() works only on torch tensors.
    
    # Revert square root transformation to e_ij_max values
    if mode=='root':
        Y=Y**2  

    return Y

# Function to keep only upto 500 materials with e_ij_max values less than 1
def modify_data(X, Y, ids):
    indices=np.where(Y<1)[0]    # Get indices of materials with e_ij_max values less than 1
    indices=list(indices)       # Convert to list
    random.shuffle(indices)     # Shuffle the list

    indices2=np.where(Y>=1)[0]  # Get indices of materials with e_ij_max values greater than or equal to 1
    indices2=list(indices2)     # Convert to list
    
    # Keep only upto 500 of the materials with e_ij_max values less than 1
    indices=indices[:500]

    # Combine the indices
    indices=indices+indices2    # list+list concatenates the lists

    # Keep only the materials with indices in the list
    X=X[indices]
    Y=Y[indices]
    ids=ids[indices]

    return X, Y, ids

# Function to split data into train, validation and test sets
def split_data(dataset, train_ratio=0.8, val_ratio=0.0, test_ratio=0.2, model='SVM', seed=0):
    """
    Returns:
        train_dataset, val_dataset, test_dataset

        train_dataset: Training set
            type: tuple of numpy arrays. First element is X, second element is Y, third element is ids. (for SVM)
            type: MPDataset object. (for NN)
        val_dataset: Validation set
            type: tuple of numpy arrays. First element is X, second element is Y, third element is ids. (for SVM)
            type: MPDataset object. (for NN)
        test_dataset: Test set
            type: tuple of numpy arrays. First element is X, second element is Y, third element is ids. (for SVM)
            type: MPDataset object. (for NN)
    """
    print('Splitting data...')

    # Check if sum of split ratios is less than or equal to 1
    if (train_ratio+val_ratio+test_ratio)>1:
        print('Invalid split ratios. Sum of split ratios should be less than or equal to 1.')
        exit()

    # Get lengths of train, validation and test sets. Unused length is required for random_split function in PyTorch since it requires the sum of lengths of the splits to be equal to the length of the dataset.
    train_length=int(len(dataset)*train_ratio)
    val_length=int(len(dataset)*val_ratio)
    test_length=int(len(dataset)*test_ratio)
    unused_length=len(dataset)-(train_length+val_length+test_length)

    if model=='SVM':
        indices=list(range(len(dataset)))   # Get indices of dataset
        random.Random(seed).shuffle(indices)    # Shuffle the indices
        train_indices=indices[:train_length]    # Get indices of train set
        val_indices=indices[train_length:train_length+val_length]   # Get indices of validation set
        test_indices=indices[train_length+val_length:train_length+val_length+test_length]   # Get indices of test set

        train_dataset=dataset[train_indices]    # Get train set
        val_dataset=dataset[val_indices]        # Get validation set
        test_dataset=dataset[test_indices]      # Get test set

    if model=='NN':
        # Split the dataset into train, validation and test sets
        train_dataset, val_dataset, test_dataset, unused_dataset=random_split(dataset, [train_length, val_length, test_length, unused_length], generator=torch.Generator().manual_seed(seed))

    return train_dataset, val_dataset, test_dataset

# Function to split data into cross validation folds
def split_data_CV(dataset, cv_folds, model='SVM', seed=0):
    """
    Returns:
        cv_dataset

        cv_dataset: Cross validation folds
            type: list of tuples of numpy arrays. Each element of the list is a fold. First element of each tuple is X, second element is Y, third element is ids. (for SVM)
            type: list of MPDataset objects. Each element of the list is a fold. (for NN)
    """
    print('Splitting data')

    dataset_size=len(dataset)   # Get size of dataset
    fold_length=int(dataset_size/cv_folds)  # Get length of each fold
    
    if model=='SVM':
        indices=list(range(len(dataset)))   # Get indices of dataset
        random.Random(seed).shuffle(indices)    # Shuffle the indices
        cv_dataset=[dataset[indices[i*fold_length:(i+1)*fold_length]] for i in range(cv_folds)] # Split the dataset into cv_folds folds
        return cv_dataset

    if model=='NN':
        unused_length=dataset_size-(fold_length*cv_folds)   # Get length of unused data
        folds=[fold_length for i in range(cv_folds)]        # Get list of fold lengths
        folds.append(unused_length)                         # Append unused length to the list
        cv_dataset=random_split(dataset, folds)             # Split the dataset into cv_folds folds
        return cv_dataset[0:cv_folds]                       # Return the first cv_folds folds, since the last fold is the unused data

# Function to get the model
def model_setup(input_size, model_params, job_params):
    if job_params['model']=='SVM':
        model=Pipeline([
            ('scaler', StandardScaler()),
            ('model', svm.SVR(**model_params))
        ])

    if job_params['model']=='NN':

        # Define the model
        class Model(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, fc_count):
                super(Model, self).__init__()
                
                self.fcs=nn.ModuleList()    # ModuleList to store the fully connected layers
                self.fcs.append(nn.Linear(input_size, hidden_size)) # Add the first fully connected layer
                # Add the remaining fully connected layers
                for i in range(fc_count):
                    self.fcs.append(nn.Linear(hidden_size, hidden_size))

                self.out_fc=nn.Linear(hidden_size, output_size)   # Add the output fully connected layer

            def forward(self, x):
                # Forward pass through the fully connected layers
                for i in range(len(self.fcs)):
                    x=F.relu(self.fcs[i](x))

                x=self.out_fc(x)    # Forward pass through the output fully connected layer
                return x

        model=Model(input_size, model_params['dim'], 1, model_params['fc_count'])   # Create the model

    return model

# Function to train on a single epoch
def trainer(device, model, loader, loss_fn, optimizer, mode='none'):

    model.train()   # Set model to training mode
    total_loss=0    # Variable to store total loss
    outputs=[]      # Variable to store outputs

    for data in loader:
        X=data[0].float().to(device)    # Get input data. Convert to float and send to device
        Y=data[1].float().to(device)    # Get target data. Convert to float and send to device
        Y=Y.view(-1, 1)                 # Reshape target data to (batch_size, 1)
        Y=transform_data(Y, mode=mode)  # Transform target data
        Y_pred=model(X)                 # Get model predictions
        loss=loss_fn(Y_pred, Y)         # Calculate loss
        Y_pred=revert_transform(Y_pred, mode=mode).squeeze().cpu().detach().numpy().tolist()    # Revert the transformation, squeeze the output to (batch_size,), transfer to cpu, detach the output from the PyTorch computational graph, convert to numpy array, convert to list and store in outputs
        outputs+=Y_pred                 # list+list=concatenated list :)
        total_loss+=loss.item()         # Add loss to total loss. loss is a tensor, so we need to convert it to a scalar using item()
        optimizer.zero_grad()           # Reset gradients
        loss.backward()                 # Calculate gradients
        optimizer.step()                # Update weights

    total_loss=total_loss/len(loader)   # Calculate average loss
    return outputs, total_loss

# Function to evaluate data
def evaluator(device, model, loader, loss_fn, mode='none'):
    model.eval()    # Set model to evaluation mode
    total_loss=0    # Variable to store total loss
    outputs=[]      # Variable to store outputs

    # Disable gradient calculation since we are not training
    with torch.no_grad():
        for data in loader:
            X=data[0].float().to(device)    # Get input data. Convert to float and send to device
            Y=data[1].float().to(device)    # Get target data. Convert to float and send to device
            Y=Y.view(-1, 1)                 # Reshape target data to (batch_size, 1)
            Y_pred=model(X)                 # Get model predictions
            Y_pred=revert_transform(Y_pred, mode=mode)  # Revert the transformation
            loss=loss_fn(Y_pred, Y)         # Calculate loss
            Y_pred=Y_pred.squeeze().cpu().detach().numpy().tolist() # Squeeze the output to (batch_size,), transfer to cpu, detach the output from the PyTorch computational graph, convert to numpy array, convert to list and store in outputs
            outputs+=Y_pred                 # list+list=concatenated list :)
            total_loss+=loss.item()         # Add loss to total loss. loss is a tensor, so we need to convert it to a scalar using item()

    total_loss=total_loss/len(loader)       # Calculate average loss
    return outputs, total_loss

# Function to train NN
def train_NN(device, train_loader, val_loader, model, model_params, loss_method, mode='none', verbose=True):

    # Define the loss function
    loss_fn=getattr(nn, loss_method)() # What is getattr? https://stackoverflow.com/questions/3061/calling-a-function-of-a-module-by-using-its-name-a-string

    # Define the optimizer
    optimizer=getattr(torch.optim, model_params['optimizer'])(model.parameters(), lr=model_params['lr'])

    best_model=None     # Variable to store the best model according to validation loss
    best_val_loss=1e10  # Variable to store the best validation loss

    # Train the model
    print('Training NN...')
    t1=time.time()      # Start timer
    train_losses=[]     # Variable to store training losses
    val_losses=[]       # Variable to store validation losses
    # Loop over epochs
    for epoch in tqdm(range(model_params['epochs'])):
        train_outputs, train_loss=trainer(device, model, train_loader, loss_fn, optimizer, mode=mode)   # Train on a single epoch

        train_losses.append(train_loss)       # Add to training losses

        val_loss=0      # Variable to store validation loss
        val_outputs=[]  # Variable to store validation outputs
        # Evaluate on validation data if validation data is provided
        if val_loader is not None:
            val_outputs, val_loss=evaluator(device, model, val_loader, loss_fn, mode=mode)
        val_losses.append(val_loss)           # Add to validation losses

        # Save the best model according to validation loss
        if val_loss<best_val_loss:
            best_val_loss=val_loss
            best_model=copy.deepcopy(model)
        
        # Print training and validation losses
        if verbose:
            if (epoch+1)%10==0:
                print('Epoch: ', epoch+1, ' Train loss: ', train_losses[-1], ' Val loss: ', val_losses[-1])

    t2=time.time()  # Stop timer
    print('Time taken to train NN: ', t2-t1, ' seconds')    # Print time taken to train NN

    return best_model, train_losses, val_losses

# Function to evaluate NN
def evaluate_NN(device, model, test_loader, mode='none'):

    # Define the loss function
    loss_fn=getattr(nn, 'L1Loss')()

    print('Evaluating NN...')
    t1=time.time()  # Start timer
    test_outputs, test_loss=evaluator(device, model, test_loader, loss_fn, mode=mode)   # Evaluate on test data
    t2=time.time()  # Stop timer
    print('Time taken to evaluate NN: ', t2-t1, ' seconds')   # Print time taken to evaluate NN

    return test_loss

# Function to train SVM
def train_SVM(X_train, Y_train, model, mode='none', verbose=True):
    Y_train=transform_data(Y_train, mode=mode)
    print('Training SVM...')    
    t1=time.time()  # Start timer
    model.fit(X_train, Y_train) # Train SVM
    t2=time.time()  # Stop timer
    print('Time taken to train SVM: ', t2-t1, ' seconds')   # Print time taken to train SVM
    return model

# Function to evaluate SVM
def evaluate_SVM(model, X_test, Y_test, mode='none'):
    print('Evaluating SVM...')
    t1=time.time()  # Start timer
    Y_pred=model.predict(X_test)    # Predict on test data
    Y_pred=revert_transform(Y_pred, mode=mode)  # Revert the transformation
    score=mean_absolute_error(Y_test, Y_pred)   # Calculate mean absolute error
    t2=time.time()  # Stop timer
    print('Time taken to evaluate SVM: ', t2-t1, ' seconds')    # Print time taken to evaluate SVM
    return score

# Function to train models
def run_train(dataset, input_size, model_params, train_params, process_params, job_params, verbose=False):

    # Split data into train, validation and test sets
    train_dataset, val_dataset, test_dataset=split_data(dataset, train_params['train_ratio'], train_params['val_ratio'], train_params['test_ratio'], job_params['model'], job_params['seed'])

    if job_params['model']=='SVM':
        model=model_setup(input_size, model_params, job_params) # Setup SVM
        model=train_SVM(train_dataset[0], train_dataset[1], model, mode=process_params['transform_mode'], verbose=verbose)  # Train SVM
        loss=evaluate_SVM(model, test_dataset[0], test_dataset[1], mode=process_params['transform_mode'])   # Evaluate SVM
        print('Mean absolute error: ', loss)    # Print mean absolute error
        print('\n')
        return loss

    if job_params['model']=='NN':
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check if GPU is available
        print('Device: ', device)   # Print device
        model=model_setup(input_size, model_params, job_params) # Setup NN
        model=model.to(device)  # Move model to device

        # Get data loaders for all the splits
        train_loader=DataLoader(train_dataset, batch_size=model_params['batch_size'], shuffle=True, pin_memory=True)
        val_loader=DataLoader(val_dataset, batch_size=model_params['batch_size'], shuffle=True, pin_memory=True)
        test_loader=DataLoader(test_dataset, batch_size=model_params['batch_size'], shuffle=True, pin_memory=True)

        model, train_losses, val_losses=train_NN(device, train_loader, val_loader, model, model_params, train_params['loss'], mode=process_params['transform_mode'], verbose=verbose)   # Train NN
        loss=evaluate_NN(device, model, test_loader, mode=process_params['transform_mode'])  # Evaluate NN
        print('Mean absolute error: ', loss)    # Print mean absolute error
        print('\n')
        return loss

# Function to run cross validation
def run_CV(dataset, input_size, model_params, train_params, process_params, job_params, verbose=False):

    # Get cross validation splits
    dataset=split_data_CV(dataset, job_params['cv_folds'], job_params['model'], job_params['seed'])

    if job_params['model']=='SVM':
        losses=[]   # List to store losses
        # Run cross validation
        for i in range(job_params['cv_folds']):
            print('Fold: ', i+1)
            model=model_setup(input_size, model_params, job_params) # Setup SVM
            X_train=[x[0] for j, x in enumerate(dataset) if j!=i]   # Get training data. Exclude the current fold
            Y_train=[x[1] for j, x in enumerate(dataset) if j!=i]   # Get training labels. Exclude the current fold
            X_train=np.concatenate(X_train, axis=0)                 # Concatenate training data since it is a list of numpy arrays
            Y_train=np.concatenate(Y_train, axis=0)                 # Concatenate training labels since it is a list of numpy arrays
            X_test=dataset[i][0]                                    # Get test data. It is the current fold
            Y_test=dataset[i][1]                                    # Get test labels. It is the current fold
            model=train_SVM(X_train, Y_train, model, mode=process_params['transform_mode'], verbose=verbose) # Train SVM   
            loss=evaluate_SVM(model, X_test, Y_test, mode=process_params['transform_mode']) # Evaluate SVM
            print('Mean absolute error: ', loss)    # Print mean absolute error
            print('\n')
            losses.append(loss) # Append loss to list
        print(losses)   # Print list of losses
        return np.mean(losses)

    if job_params['model']=='NN':
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Check if GPU is available
        print('Device: ', device)   # Print device
        losses=[]   # List to store losses
        for i in range(job_params['cv_folds']):
            print('Fold: ', i+1)    
            model=model_setup(input_size, model_params, job_params)     # Setup NN
            model=model.to(device)                                      # Move model to device
            train_dataset=[x for j, x in enumerate(dataset) if i!=j]    # Get training data. Exclude the current fold
            train_dataset=ConcatDataset(train_dataset)                  # Concatenate training data since it is a list of datasets
            test_dataset=dataset[i]                                     # Get test data. It is the current fold

            # Get data loaders for all the splits
            train_loader=DataLoader(train_dataset, batch_size=model_params['batch_size'], shuffle=True, pin_memory=True)
            test_loader=DataLoader(test_dataset, batch_size=model_params['batch_size'], shuffle=True, pin_memory=True)

            model, train_losses, val_losses=train_NN(device, train_loader, None, model, model_params, train_params['loss'], mode=process_params['transform_mode'], verbose=verbose) # Train NN
            loss=evaluate_NN(device, model, test_loader, mode=process_params['transform_mode']) # Evaluate NN
            print('Mean absolute error: ', loss)    # Print mean absolute error
            print('\n')
            losses.append(loss) # Append loss to list
        print(losses)   # Print list of losses
        return np.mean(losses)

if __name__=='__main__':
    X, Y, ids=process.get_data(config['Processing'])  # Get data

    # Define dataset class
    class MPDataset(Dataset):
        def __init__(self, X, Y, ids):
            self.X=X
            self.Y=Y
            self.ids=ids

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx], self.ids[idx]

    dataset=MPDataset(X, Y, ids)    # Define dataset
    input_size=dataset[0][0].shape[0]   # Get input size

    print("Number of samples: ", len(dataset))  # Print number of samples
    
    losses=[]   # List to store losses

    # Repeat experiment for the number of repetitions
    for i in range(args.reps):
        
        print('Rep: ', i+1)

        if run_mode=='Training':
            loss=run_train(dataset, input_size, config['Model'], config['Training'], config['Processing'], config['Job'], verbose=True)
            losses.append(loss)
            print('Training loss: ', loss)
            print('\n')

        if run_mode=='CV':
            loss=run_CV(dataset, input_size, config['Model'], config['Training'], config['Processing'], config['Job'], verbose=False)
            losses.append(loss)
            print('CV loss: ', loss)
            print('\n')
    
    print(losses)
    print(f'mean: {np.mean(losses)}, std: {np.std(losses)}, min: {np.min(losses)}, max: {np.max(losses)}')
    