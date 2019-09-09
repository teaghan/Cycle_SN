import h5py
import numpy as np
from training_fns import vec_bin_array

data_file_obs = '/scratch/obriaint/Cycle_SN/data/aspcapStar_dr14.h5'

datasets = ['test','cv','train', 'cluster']

# Pixel bits in bad bits are indexed according to 
# https://www.sdss.org/dr14/algorithms/bitmasks/#APOGEE_PIXMASK
num_bits = 16
bad_bits=np.array([12])

# Reverse order for proper indexing
bad_bits = (num_bits-1)+(bad_bits)*-1
# Sort to enable indexing
bad_bits = np.sort(bad_bits.astype(int))

with h5py.File(data_file_obs, "r+") as F_obs:
    for dataset in datasets:
        print('dataset: ' + dataset)
        x_msk = F_obs['APOGEE mask_spectrum ' + dataset][:]
        
        bin_msk = []
        for i, m in enumerate(x_msk):
            if i%10000==0:
                print('%i of %i complete.'%(i, len(x_msk)))
            # Turn integer flags into rows of bit vectors
            spec_msk_bin = vec_bin_array(np.array(m, dtype=int), num_bits)

            # Locate bad pixels
            bad_pixels, _ = np.where(spec_msk_bin[:,bad_bits]==1.)

            # Turn integer flags into a vector binary mask
            spec_msk_bin = np.ones_like(m)
            spec_msk_bin[bad_pixels]= 0.
            # Save mask
            bin_msk.append(spec_msk_bin)
            
        bin_msk = np.array(bin_msk)
        print(bin_msk.shape)
            
        #msk_ds = F_obs.create_dataset('APOGEE bit_mask_spectrum ' + dataset, 
        #                              bin_msk.shape, dtype="i")
        msk_ds = F_obs['APOGEE bit_mask_spectrum ' + dataset]
        msk_ds[:] = bin_msk
