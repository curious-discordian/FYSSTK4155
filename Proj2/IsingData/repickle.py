## Have to repickle these bothersome components,
## In order to use them in python 2.
## USE PYTHON 3 first.
import pickle
import numpy as np
def repickle(t,root="./"):
    data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=%.2f.pkl'%t,'rb'))
    #newdata = np.unpackbits(data).astype(int).reshape(-1,1600)
    with open('Ising2DFM_rePickled_L40_T=%.2f.pkl'%t,'wb') as f:
        pickle.dump(data, f,2)
    #return np.unpackbits(data).astype(int).reshape(-1,1600)

[repickle(t/4.) for t in range(1,17,1)]

