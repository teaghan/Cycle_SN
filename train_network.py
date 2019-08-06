# import packages
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import h5py
from collections import defaultdict
import sys
import os
import glob
import time
import configparser
from collections import OrderedDict
from distutils import util

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary

from training_fns import (parseArguments, weighted_masked_mse_loss, PayneObservedDataset, 
                          create_synth_batch, batch_to_cuda, train_iter, evaluation_checkpoint)
from network import CycleSN

from The_Payne.utils import read_in_neural_network

np.random.seed(1)

# Check for GPU
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('Using GPU!')
    #dtype = torch.cuda.FloatTensor
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    #dtype = torch.FloatTensor
    torch.set_default_tensor_type('torch.FloatTensor')

# Collect the command line arguments
args = parseArguments()
model_name = args.model_name
verbose_iters = args.verbose_iters
cp_time = args.cp_time
data_dir = args.data_dir
fixed_vmacro = args.fixed_vmacro

# Directories
cur_dir = os.path.dirname(__file__)
config_dir = os.path.join(cur_dir, 'configs/')
model_dir = os.path.join(cur_dir, 'models/')
progress_dir = os.path.join(cur_dir, 'progress/')
if args.data_dir is None:
    data_dir = os.path.join(cur_dir, 'data/')

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

# TRAINING PARAMETERS
batchsize = int(config['TRAINING']['batchsize'])
learning_rate_encoder = float(config['TRAINING']['learning_rate_encoder'])
learning_rate_decoder = float(config['TRAINING']['learning_rate_decoder'])
learning_rate_discriminator = float(config['TRAINING']['learning_rate_discriminator'])
loss_weight_rec = float(config['TRAINING']['loss_weight_rec'])
loss_weight_cc = float(config['TRAINING']['loss_weight_cc'])
loss_weight_gen = float(config['TRAINING']['loss_weight_gen'])
loss_weight_dis = float(config['TRAINING']['loss_weight_dis'])
lr_decay_batch_iters = float(config['TRAINING']['lr_decay_batch_iters'])
lr_decay = float(config['TRAINING']['lr_decay'])
total_batch_iters = float(config['TRAINING']['total_batch_iters'])
use_real_as_true = bool(util.strtobool(config['TRAINING']['use_real_as_true']))
mask_synth_lines = bool(util.strtobool(config['TRAINING']['mask_synth_lines']))

# BUILD THE NETWORKS

# Load the Payne pre-trained weights
emulator_coeffs = read_in_neural_network()

print('\nBuilding networks...')
model = CycleSN(architecture_config, emulator_coeffs, use_cuda=use_cuda)

# Display model architectures
'''
print('\n\nSYNTHETIC EMULATOR ARCHITECTURE:\n')
summary(model.emulator, input_size=(25,))
print('\n\nENCODER_synth and ENCODER_obs ARCHITECTURE:\n')
summary(model.encoder_synth, input_size=(1, int(architecture_config['num_pixels'])))
print('\n\nENCODER_sh ARCHITECTURE:\n')
summary(model.encoder_sh, input_size=model.enc_interm_shape)
if model.use_split:
    print('\n\nENCODER_sp ARCHITECTURE:\n')
    summary(model.encoder_sp, input_size=model.enc_interm_shape)
    print('\n\nDECODER_sp ARCHITECTURE:\n')
    summary(model.decoder_sp, input_size=model.z_sp_shape)
print('\n\nDECODER_sh ARCHITECTURE:\n')
summary(model.decoder_sh, input_size=model.z_sh_shape)
print('\n\nDECODER_synth and DECODER_obs ARCHITECTURE:\n')
summary(model.decoder_synth, input_size=model.dec_interm_shape)
print('\n\nDISCRIM_synth and DISCRIM_obs ARCHITECTURE:\n')
summary(model.discriminator_synth, input_size=[(1, int(architecture_config['num_pixels'])),
                                               model.z_sh_shape])
'''
print('\n\nSYNTHETIC EMULATOR ARCHITECTURE:\n')
print(model.emulator)
print('\n\nENCODER_synth and ENCODER_obs ARCHITECTURE:\n')
print(model.encoder_synth)
print('\n\nENCODER_sh ARCHITECTURE:\n')
print(model.encoder_sh)
if model.use_split:
    print('\n\nENCODER_sp ARCHITECTURE:\n')
    print(model.encoder_sp)
    print('\n\nDECODER_sp ARCHITECTURE:\n')
    print(model.decoder_sp)
print('\n\nDECODER_sh ARCHITECTURE:\n')
print(model.decoder_sh)
print('\n\nDECODER_synth and DECODER_obs ARCHITECTURE:\n')
print(model.decoder_synth)
print('\n\nDISCRIM_synth and DISCRIM_obs ARCHITECTURE:\n')
print(model.discriminator_synth)

# Construct optimizers
if model.use_split:
    optimizer_rec_and_gen = torch.optim.Adam([{'params': model.encoder_synth.parameters(), "lr": learning_rate_encoder},
                                              {'params': model.encoder_obs.parameters(), "lr": learning_rate_encoder},
                                              {'params': model.encoder_sh.parameters(), "lr": learning_rate_encoder},
                                              {'params': model.encoder_sp.parameters(), "lr": learning_rate_encoder},
                                              {'params': model.decoder_synth.parameters(), "lr": learning_rate_decoder},
                                              {'params': model.decoder_obs.parameters(), "lr": learning_rate_decoder},
                                              {'params': model.decoder_sh.parameters(), "lr": learning_rate_decoder},
                                              {'params': model.decoder_sp.parameters(), "lr": learning_rate_decoder}],
                                             weight_decay = 0, betas=(0.5, 0.999))
else:
    optimizer_rec_and_gen = torch.optim.Adam([{'params': model.encoder_synth.parameters(), "lr": learning_rate_encoder},
                                              {'params': model.encoder_obs.parameters(), "lr": learning_rate_encoder},
                                              {'params': model.encoder_sh.parameters(), "lr": learning_rate_encoder},
                                              {'params': model.decoder_synth.parameters(), "lr": learning_rate_decoder},
                                              {'params': model.decoder_obs.parameters(), "lr": learning_rate_decoder},
                                              {'params': model.decoder_sh.parameters(), "lr": learning_rate_decoder}],
                                             weight_decay = 0, betas=(0.5, 0.999))
optimizer_dis = torch.optim.Adam([{'params': model.discriminator_synth.parameters(), "lr": learning_rate_discriminator},
                                  {'params': model.discriminator_obs.parameters(), "lr": learning_rate_discriminator}],
                                 weight_decay = 0, betas=(0.5, 0.999))
lr_scheduler_rg = torch.optim.lr_scheduler.StepLR(optimizer_rec_and_gen, 
                                                  step_size=lr_decay_batch_iters, 
                                                  gamma=lr_decay)
lr_scheduler_dis = torch.optim.lr_scheduler.StepLR(optimizer_dis, 
                                                   step_size=lr_decay_batch_iters, 
                                                   gamma=lr_decay)

# Loss functions
gan_loss = nn.BCELoss()
distance_loss = weighted_masked_mse_loss

# Check for pre-trained weights
model_filename =  os.path.join(model_dir,model_name+'.pth.tar')
if os.path.exists(model_filename):
    fresh_model = False
else:
    fresh_model = True
    
# Load pretrained model
if fresh_model:
    print('\nStarting fresh model to train...')
    cur_iter = 1
    losses = defaultdict(list)
else:
    print('\nLoading saved model to continue training...')
    # Load model info
    checkpoint = torch.load(model_filename, map_location=lambda storage, loc: storage)
    cur_iter = checkpoint['batch_iters']+1
    losses = dict(checkpoint['losses'])
    
    # Load optimizer states
    optimizer_rec_and_gen.load_state_dict(checkpoint['optimizer_rec_and_gen'])
    optimizer_dis.load_state_dict(checkpoint['optimizer_dis'])
    
    # Load model weights
    model.load_state_dict(checkpoint['cycle_model'])
    
# DATA

# Normalization data for the spectra
x_mean, x_std = np.load(spectra_norm_file)

# Load line mask
line_mask = np.load(data_dir+'mock_missing_lines.npz')['total_mask']
line_mask = torch.from_numpy(np.array(line_mask, dtype=np.uint8))
    
# Load the Payne labels

# [Teff, Logg, Vturb [km/s],
# [C/H], [N/H], [O/H], [Na/H], [Mg/H],
# [Al/H], [Si/H], [P/H], [S/H], [K/H],
# [Ca/H], [Ti/H], [V/H], [Cr/H], [Mn/H],
# [Fe/H], [Co/H], [Ni/H], [Cu/H], [Ge/H],
# C12/C13, Vmacro [km/s]
labels_payne = np.load(data_dir+'mock_all_spectra_no_noise_resample_prior_large.npz')['labels'].T

# Perturb the payne labels within a range to create our synthetic training batches.
# These perturbations are in the same order as the labels.
perturbations = [100., 0.1, 0.2, *np.repeat(0.1, 20), 5., 2.]

# Training dataset to loop through
obs_dataset = PayneObservedDataset(data_file_obs, obs_domain='PAYNE', dataset='train', 
                                   x_mean=x_mean, x_std=x_std, collect_x_mask=False)
obs_train_dataloader = DataLoader(obs_dataset, batch_size=batchsize, shuffle=True, num_workers=6)

# Validation set that consists of matching pairs in the synthetic and observed domains
obs_val_set = obs_dataset.__getitem__(1000, dataset='val', return_labels=True, collect_preceeding=True) 
synth_val_set = create_synth_batch(model, x_mean, x_std, y=obs_val_set['y'], line_mask=line_mask)
# Switch to GPU
if use_cuda:
    obs_val_set = batch_to_cuda(obs_val_set)
    synth_val_set = batch_to_cuda(synth_val_set)

def train_network(cur_iter):
    print('Training the network...')
    print('Progress will be displayed every %i iterations and the model will be saved every %i minutes.'%
          (verbose_iters,cp_time))
    # Train the neural networks
    losses_cp = defaultdict(list)
    cp_start_time = time.time()
    
    while cur_iter < total_batch_iters:
        
        for obs_train_batch in obs_train_dataloader:
            # Create synthetic batch from the distribution of the original Payne training set.
            # We will use the "line_mask" to set some lines to the continuum.
            # We will also fix the vmacro labels to 15km/s.
            synth_train_batch = create_synth_batch(model, x_mean, x_std, 
                                                   batchsize=len(obs_train_batch['x']),
                                                   line_mask=line_mask, 
                                                   labels_payne=labels_payne, 
                                                   perturbations=perturbations, 
                                                   fixed_vmacro=fixed_vmacro)
            if use_cuda:
                obs_train_batch = batch_to_cuda(obs_train_batch)
                synth_train_batch = batch_to_cuda(synth_train_batch)
            
            # Train an iteration
            losses_cp = train_iter(model, obs_train_batch, synth_train_batch, 
                                   distance_loss, gan_loss, loss_weight_rec, 
                                   loss_weight_cc, loss_weight_gen, loss_weight_dis,
                                   optimizer_rec_and_gen, optimizer_dis, 
                                   lr_scheduler_rg, lr_scheduler_dis, 
                                   use_real_as_true, losses_cp, use_cuda)
            
            # Evaluate val set and display losses
            if cur_iter % verbose_iters == 0:
                
                losses_cp = evaluation_checkpoint(model, obs_val_set, 
                                                  synth_val_set, distance_loss, 
                                                  losses_cp)
                
                # Calculate averages
                for k in losses_cp.keys():
                    losses[k].append(np.mean(losses_cp[k]))
                losses['batch_iters'].append(cur_iter)

                # Print current status
                print('\nBatch Iterations: %i/%i ' % (cur_iter, total_batch_iters))
                print('Training Losses:')
                print('\t|   Rec   |   CC    |   Gen   |   Dis   |')
                print('Synth   | %0.5f | %0.5f | %0.5f | %0.5f |' % 
                      (losses['rec_synth'][-1], losses['cc_synth'][-1], losses['gen_synth'][-1], 
                       np.mean([losses['dis_real_synth'][-1], losses['dis_fake_synth'][-1]])))
                print('Obs     | %0.5f | %0.5f | %0.5f | %0.5f |' % 
                      (losses['rec_obs'][-1], losses['cc_obs'][-1], losses['gen_obs'][-1], 
                       np.mean([losses['dis_real_obs'][-1], losses['dis_fake_obs'][-1]])))
                
                print('Validation Scores:')
                if model.use_split:
                    print('| x_synthobs | x_obssynth  | zsh_synth | zsh_obs |   zsh   |   zsp   |')
                    print('|   %0.5f  |   %0.5f   |  %0.5f  | %0.5f | %0.5f | %0.5f |' % 
                          (losses['x_synthobs_val'][-1], losses['x_obssynth_val'][-1], losses['zsh_synth_val'][-1], 
                           losses['zsh_obs_val'][-1], losses['zsh_val'][-1], losses['zsp_val'][-1]))
                else:
                    print('| x_synthobs | x_obssynth  | zsh_synth |  zsh_obs  |   zsh   |')
                    print('|   %0.5f  |   %0.5f   |  %0.5f  |  %0.5f  | %0.5f |' % 
                          (losses['x_synthobs_val'][-1], losses['x_obssynth_val'][-1], losses['zsh_synth_val'][-1], 
                           losses['zsh_obs_val'][-1], losses['zsh_val'][-1]))
                print('\n') 

                # Save losses to file to analyze throughout training. 
                np.save(os.path.join(progress_dir, model_name+'_losses.npy'), losses) 
                # Reset checkpoint loss dict
                losses_cp = defaultdict(list)
                # Free some GPU memory
                torch.cuda.empty_cache()

            # Increase the iteration
            cur_iter += 1
            
            # Save periodically
            if time.time() - cp_start_time >= cp_time*60:
                print('Saving network...')

                torch.save({'batch_iters': cur_iter,
                            'losses': losses,
                            'optimizer_rec_and_gen' : optimizer_rec_and_gen.state_dict(),
                            'optimizer_dis' : optimizer_dis.state_dict(),
                            'cycle_model' : model.state_dict()}, 
                            model_filename)
                
                cp_start_time = time.time()
                    
            
# Run the training
if __name__=="__main__":
    train_network(cur_iter)