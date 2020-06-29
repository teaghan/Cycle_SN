import numpy as np
import os
import configparser
from distutils import util

import torch

import sys
sys.path.append('../')
from network import CycleSN
from training_fns import batch_to_cuda, create_synth_batch, CSNDataset
#from lbfgsnew import LBFGSNew

model_name = 'kurucz_to_apogee_67'
num_spec = 10
dataset = 'train' # or 'test'

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('Using GPU!')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    
# Directories
csn_dir = '..'
config_dir = os.path.join(csn_dir, 'configs/')
model_dir = os.path.join(csn_dir, 'models/')
data_dir = os.path.join(csn_dir, 'data/')

# Model configuration
config = configparser.ConfigParser()
config.read(config_dir+model_name+'.ini')
architecture_config = config['ARCHITECTURE']
print('\nCreating model: %s'%model_name)
print('\nConfiguration:')
for key_head in config.keys():
    if key_head=='DEFAULT':
        continue
    print('  %s' % key_head)
    for key in config[key_head].keys():
        print('    %s: %s'%(key, config[key_head][key]))
        
# DATA FILES
data_file_obs = os.path.join(data_dir, config['DATA']['data_file_obs'])
spectra_norm_file = os.path.join(data_dir, config['DATA']['spectra_norm_file'])
emulator_fn = os.path.join(model_dir, config['DATA']['emulator_fn'])

# BUILD THE NETWORKS

print('\nBuilding networks...')
model = CycleSN(architecture_config, emulator_fn, use_cuda=use_cuda)

model_filename =  os.path.join(model_dir,model_name+'.pth.tar')

print('\nLoading saved model...')
# Load model info
checkpoint = torch.load(model_filename, map_location=lambda storage, loc: storage)

# Load model weights
model.load_state_dict(checkpoint['cycle_model'])

# Normalization data for the spectra
x_mean, x_std = np.load(spectra_norm_file)
model.x_mean = x_mean
model.x_std = x_std

mask_synth_lines = bool(util.strtobool(config['TRAINING']['mask_synth_lines']))
if mask_synth_lines:
    print('Using line mask.')
    # Load line mask
    line_mask = np.load(data_dir+'mock_missing_lines.npz')['total_mask']
    line_mask = torch.from_numpy(np.array(line_mask, dtype=np.uint8))
else:
    # Don't use line mask
    line_mask = None

# A set of observed spectra
obs_dataset = CSNDataset(data_file_obs, dataset=dataset, x_mean=x_mean, 
                             x_std=x_std, line_mask=None)
obs_batch = obs_dataset.__getitem__(np.arange(num_spec)) 
# Generate synth batch of matching spectra
synth_batch = create_synth_batch(model, x_mean, x_std, obs_batch['y'], 
                                 line_mask=line_mask, use_cuda=use_cuda)

# Switch to GPU
if use_cuda:
    obs_batch = batch_to_cuda(obs_batch)
    synth_batch = batch_to_cuda(synth_batch)
    
# Create split latent variables
model.eval_mode()
with torch.no_grad():
    zsh_obs, zsp_obs = model.obs_to_z(obs_batch['x'])
    
## Least squares fitting function    

# assumes:
#  - weight ~ 1/sigma (and not 1/sigma**2)
#  - model can be called as model(params)
#  - params as passed is a torch.nn.Parameter,and has been initialized
def least_squares_fit(model, data, weight, params):
    loss_fn = torch.nn.MSELoss(reduction='mean')
    data *= weight
  
    # L-BFGS-B optimization
    optimizer = torch.optim.LBFGS([params], lr=0.01, max_iter=100, line_search_fn='strong_wolfe')
    n_epochs = 10
    for epoch in range(n_epochs):
        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            fit = weight * model(params)
            loss = loss_fn(fit, data)
            if loss.requires_grad:
                loss.backward()
            return loss
        optimizer.step(closure)
        min_loss = closure().item()
    #print("  lbgfs loss ", min_loss)
    return min_loss

mse = torch.nn.MSELoss(reduction='mean')

y_preds = []
# Loop over all spectra
for i in range(num_spec):
    print("Spectrum ", i)
    x_weight = obs_batch['x_msk'][i:i+1] / obs_batch['x_err'][i:i+1]
    x_obs = obs_batch['x'][i:i+1]
    model.cur_z_sp = zsp_obs[i:i+1]
    # initialize stellar parameters with ThePayne guess
    y_payne = (obs_batch['y'][i:i+1] - model.y_min) / (model.y_max - model.y_min) - 0.5
    params = torch.nn.Parameter(y_payne, requires_grad=True)
    params_copy = params.clone()
    # the x_obs seems to be stochastic, rerunning the cell gives different numbers. to be checked - 
    print("  data check ", x_weight.mean(), x_obs.mean(), model.cur_z_sp.mean(), y_payne.mean())
    print("  init loss payne : ", mse(x_weight*synth_batch['x'][i:i+1], x_obs*x_weight).item())
    print("  init loss full  : ", mse(x_weight*model.y_to_obs(params), x_obs*x_weight).item())
    #loss_payne = least_squares_fit(starnet.payne, x_obs, x_weight, params)
    #print("  final loss payne: ", loss_payne)
    loss_full = least_squares_fit(model.y_to_obs, x_obs, x_weight, params)
    print("  final loss full : ", loss_full)
    y_pred = (params + 0.5) * (model.y_max - model.y_min) + model.y_min 
    y_preds.append(y_pred.data.cpu().numpy())
np.save('../data/y_preds_%s.npy'%dataset, np.array(y_preds))