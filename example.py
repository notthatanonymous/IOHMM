from __future__ import  division

import json
import warnings


import numpy as np
import pandas as pd


from IOHMM import SupervisedIOHMM
from IOHMM import OLS, CrossEntropyMNL


warnings.simplefilter("ignore")


speed = pd.read_csv('examples/data/speed.csv')


states = {}
corr = np.array(speed['corr'])
for i in range(len(corr)):
    state = np.zeros((2,))
    if corr[i] == 'cor':
        states[i] = np.array([0,1])
    else:
        states[i] = np.array([1,0])



# we choose 2 hidden states in this model
SHMM = SupervisedIOHMM(num_states=2)

# we set only one output 'rt' modeled by a linear regression model
SHMM.set_models(model_emissions = [OLS()], 
                model_transition=CrossEntropyMNL(solver='lbfgs'),
                model_initial=CrossEntropyMNL(solver='lbfgs'))

# we set no covariates associated with initial/transitiojn/emission models
SHMM.set_inputs(covariates_initial = [], covariates_transition = [], covariates_emissions = [[]])

# set the response of the emission model
SHMM.set_outputs([['rt']])

# set the data and ground truth states
SHMM.set_data([[speed, states]])


SHMM.train()

print("\n\n\n")

print(f"Score: {np.sqrt(SHMM.model_emissions[1][0].dispersion)}")
