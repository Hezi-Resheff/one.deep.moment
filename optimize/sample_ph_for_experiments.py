from util import *
import numpy as np
import pickle as pkl
import os
import sys
# Stop optimization when the loss hits this value
MIN_LOSS_EPSILON = 1e-7
sys.path.append(os.path.abspath(".."))
from utils_sample_ph import *

from utils import *

df_dat = pd.DataFrame([])


for ind in range(20000):

    try:
        orig_size = np.random.randint(25,100)
        a, A, moms = sample(orig_size)
        fitted_moms =   5
        moms = compute_first_n_moments(a, A, fitted_moms)
        ms = torch.tensor(np.array(moms).flatten())

        curr_ind = df_dat.shape[0]

        for mom in range(1,6):
            df_dat.loc[curr_ind, 'mom_'+str(mom)] = ms[mom-1].item()

        df_dat.loc[curr_ind, 'ph_orig_size'] = orig_size
    except:
        print('bad ph')


path_ph  = r'C:\Users\Eshel\workspace\data'

pkl.dump(df_dat, open(os.path.join(path_ph, 'PH_set.pkl'), 'wb'))



