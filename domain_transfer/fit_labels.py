import numpy as np
import os
import configparser
from distutils import util
import time

import torch

import sys
cur_dir = os.path.dirname(__file__)
csn_dir = os.path.join(cur_dir, '../')
print(csn_dir, csn_dir)
sys.path.append(csn_dir)
from network import CycleSN
from training_fns import batch_to_cuda, create_synth_batch, CSNDataset
#from lbfgsnew import LBFGSNew

model_name = 'kurucz_to_apogee_1'
dataset = 'train' # or 'test'

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('Using GPU!')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    
# Directories
config_dir = os.path.join(csn_dir, 'configs/')
model_dir = os.path.join(csn_dir, 'models/')
data_dir = os.path.join(csn_dir, 'data/')

# Check for pre-existing labels
results_filename =  os.path.join(data_dir,'y_preds_%s_%s.npy'%(model_name,dataset))
if os.path.exists(results_filename):
    y_preds = np.load(results_filename)
    start_indx = len(y_preds)
else:
    start_indx = 0
    
if len(sys.argv)>1:
    data_dir = sys.argv[1]

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
    
# Create split latent variables
model.eval_mode()
    
## Least squares fitting function    

# assumes:
#  - weight ~ 1/sigma (and not 1/sigma**2)
#  - model can be called as model(params)
#  - params as passed is a torch.nn.Parameter,and has been initialized
def least_squares_fit(model, data, weight, params):
    loss_fn = torch.nn.MSELoss(reduction='mean')
    data *= weight
    
    params_found = False
    count_nans = 0
    orig_params = params.clone()
    while params_found is False:
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
        if np.isnan(min_loss):
            params.data = orig_params.data
            count_nans+=1
        else:
            params_found=True
        if count_nans>5:
            params.data = orig_params.data
            params_found=True
        #print("  lbgfs loss ", min_loss)
    return min_loss

mse = torch.nn.MSELoss(reduction='mean')
cp_start_time = time.time()
# Loop over all spectra
for cur_indx in range(start_indx, len(obs_dataset)):
    print("Spectrum ", cur_indx)
    obs_batch = obs_dataset.__getitem__(cur_indx) 
    
    # Switch to GPU
    if use_cuda:
        obs_batch = batch_to_cuda(obs_batch)
    # Create z_sp
    with torch.no_grad():
        zsh_obs, zsp_obs = model.obs_to_z(obs_batch['x'].unsqueeze(0))
    model.cur_z_sp = zsp_obs
    
    x_weight = obs_batch['x_msk'].unsqueeze(0) / obs_batch['x_err'].unsqueeze(0)
    #x_obs = obs_batch['x']
    
    # initialize stellar parameters with ThePayne guess
    y_payne = (obs_batch['y'].unsqueeze(0) - model.y_min) / (model.y_max - model.y_min) - 0.5
    params = torch.nn.Parameter(y_payne, requires_grad=True)
    #params_copy = params.clone()
    # the x_obs seems to be stochastic, rerunning the cell gives different numbers. to be checked - 
    #print("  data check ", x_weight.mean(), obs_batch['x'].unsqueeze(0).mean(), model.cur_z_sp.mean(), y_payne.mean())
    #print("  init loss payne : ", mse(x_weight*synth_batch['x'][i:i+1], x_obs*x_weight).item())
    print("  init loss full  : ", mse(x_weight*model.y_to_obs(params), obs_batch['x'].unsqueeze(0)*x_weight).item())
    #loss_payne = least_squares_fit(starnet.payne, x_obs, x_weight, params)
    #print("  final loss payne: ", loss_payne)
    loss_full = least_squares_fit(model.y_to_obs, obs_batch['x'].unsqueeze(0), x_weight, params)
    print("  final loss full : ", loss_full)
    y_pred = (params + 0.5) * (model.y_max - model.y_min) + model.y_min 
    
    if cur_indx==0:
        y_preds = np.array(y_pred.data.cpu().numpy())
    else:
        y_preds = np.vstack((y_preds, y_pred.data.cpu().numpy()))
        
    # Save every 15 minutes
    if time.time() - cp_start_time >= 15*60:
        np.save(results_filename, y_preds)
        cp_start_time = time.time()