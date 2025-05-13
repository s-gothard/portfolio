import numpy as np
from copy import deepcopy
from collections import Counter
from collections import defaultdict
import pandas as pd

# Sarah Gothard | DSCI 633


class my_normalizer:
  def __init__(self, norm = "Min-Max", axis = 1):
    self.norm = norm
    self.axis = axis
    
  def fit(self,X):
    X_array = np.asarray(X)
    
    self.offsets = [] # I understand this to be the part of the numerator that is not the column to be transformed
    self.scalers = [] # I understand this to be the denomiator
    
    if self.axis == 1:
      for col in range(X_array.shape[1]): #range for number of columns
        x = X_array[:,col]
        
        if self.norm == "Min-Max":
          offset = np.min(x)
          scaler = np.max(x) - offset
          
        elif self.norm == "Standard_Score":
          offset = x.mean()
          scaler = x.std()
        
        elif self.norm == "L1":
          offset = 0 #since the numerator is the vector supplied by X_array
          scaler = np.sum(np.abs(x))
          
        elif self.norm == "L2":
          offset = 0 #since numerator is the vector
          scaler = np.sqrt(np.sum(x**2))
          
        else:
          raise Exception("Incorrect: Please chose 'Min-Max','L1','L2', or 'Standard_Score")
        
        self.offsets.append(offset)
        self.scalers.append(scaler)
    
    elif self.axis == 0:
      for row in range(X_array.shape[0]): #range for number of rows
        x = X_array[row]
        
        if self.norm == "Min-Max":
          offset = np.min(x)
          scaler = np.max(x) - offset
          
        elif self.norm == "Standard_Score":
          offset = x.mean()
          scaler = x.std()
        
        elif self.norm == "L1":
          offset = 0 #since the numerator is the vector supplied by X_array
          scaler = np.sum(np.abs(x))
          
        elif self.norm == "L2":
          offset = 0 #since numerator is the vector
          scaler = np.sqrt(np.sum(x**2))
          
        else:
          raise Exception("Incorrect: Please chose 'Min-Max','L1','L2', or 'Standard_Score")
        
        self.offsets.append(offset)
        self.scalers.append(scaler)
        
    else:
      raise Exception("Enter Valid Axis (0 or 1)")
    
  def transform(self, X): #because of the use of offets as pieces of the numerator, i took advantage of the "transform" function from hints file
    X_norm = deepcopy(np.asarray(X))
    if self.axis == 1:
      for col in range(X_norm.shape[1]): # copied code above and updated for he x_norm variable to find the range using number of columns, i understand this better
        X_norm[:,col] = (X_norm[:,col] - self.offsets[col])/self.scalers[col]
    elif self.axis == 0:
      for row in range(X_norm.shape[0]):
        X_norm[row] = (X_norm[row] - self.offsets[row])/self.scalers[row]
    else:
      raise Exception("Enter Valid Axis (0 or 1)")
    return X_norm
  
  def fit_transform(self, X):
    self.fit(X)
    return self.transform(X)
  
  def stratified_sampling(y, ratio, replace = True):
    if ratio <=0 or ratio >=1:
      raise Exception("Raitio must be 0<ratio <1")
    y_array = np.asarray(y)
    sample = []
    
    lab_indc = np.unique(y_array)
    
    for label in lab_indc:
        indc = np.where(y_array == label)[0]  
        num_samples = int(np.ceil(ratio * len(indc)))  
        
        if replace:
            sampled_indices = np.random.choice(indc, size=num_samples, replace=True)
        else:
            sampled_indices = np.random.choice(indc, size=num_samples, replace=False)
        
        sample.extend(sampled_indices) #chose extend function and not append so I can add each element from the passed through to the sample object.

    return sample
    
        
        
      
          
        

