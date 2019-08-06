import numpy as np
import h5py
import argparse
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("model_name", help="Name of model.", type=str)

    # Optional arguments
    
    # How often to display the losses
    parser.add_argument("-v", "--verbose_iters", 
                        help="Number of batch  iters after which to evaluate val set and display output.", 
                        type=int, default=10000)
    
    # How often to display save the model
    parser.add_argument("-ct", "--cp_time", 
                        help="Number of minutes after which to save a checkpoint.", 
                        type=float, default=15)
    # Alternate data directory than cycgan/data/
    parser.add_argument("-dd", "--data_dir", 
                        help="Different data directory from cycle-gan dir.", 
                        type=str, default=None)
    # Whether or not to freeze vmacro on the synthetic side
    parser.add_argument("-fv", "--fixed_vmacro", 
                        help="Fix vmacro to 15km/s in synthetic domain.", 
                        type=str, default=False)
    
    # Parse arguments
    args = parser.parse_args()
    
    args.fixed_vmacro = str2bool(args.fixed_vmacro)

    return args

def weighted_masked_mse_loss(pred, target, error, mask):
    return torch.mean(((pred - target)*mask/error) ** 2)

def create_synth_batch(model, x_mean, x_std, y=None, batchsize=8,
                       line_mask=None, labels_payne=None, perturbations=None, 
                       return_labels=False, fixed_vmacro=False, use_cuda=True):
    if y is None:
        # Collect a batch of stellar parameters from the Payne training distribution
        y = np.copy(labels_payne[np.random.randint(len(labels_payne), size=batchsize)])

        # Apply random perturbation within limits
        y += np.array([np.random.uniform(-1*p, p, size=batchsize) for p in perturbations]).T
    
        # Correct minimum Vturb, C12/C13, and Vmacro values
        for i in [2,23,24]:
            y[y[:,i]<np.min(labels_payne[:,i]),i] = np.min(labels_payne[:,i])
        
        if fixed_vmacro:
            # Set the Vmacro to 15km/s.
            y[:,24] = 15.
        
        y = Variable(torch.Tensor(y.astype(np.float32)))

    # Create a batch of synthetic spectra
    x = model.y_to_synth(y, use_cuda=use_cuda)
    
    if line_mask is not None:
        # Mask lines
        x[:,line_mask == 1.] = 1.

    # Normalize the spectra
    x = (x - x_mean) / x_std
    
    # Only select last 7167 pixels
    x = x[:,47:]
    
    if return_labels:    
        return {'x':x, 'x_err':torch.ones(x.size(), dtype=torch.float32), 
                'x_msk':torch.ones(x.size(), dtype=torch.float32), 'y':y} 
    else:    
        return {'x':x, 'x_err':torch.ones(x.size(), dtype=torch.float32), 
                'x_msk':torch.ones(x.size(), dtype=torch.float32)} 
    
def batch_to_cuda(batch):
    for k in batch.keys():
        batch[k] = batch[k].cuda()
    return batch

class PayneObservedDataset(Dataset):
    
    """
            
    """

    def __init__(self, data_file_obs, obs_domain, dataset='train', x_mean=1., x_std=1., 
                 collect_x_mask=False):
        
        self.data_file_obs = data_file_obs
        self.obs_domain = obs_domain
        self.dataset = dataset
        self.x_mean = x_mean
        self.x_std = x_std
        self.collect_x_mask = collect_x_mask
        
    def __len__(self):
        with h5py.File(self.data_file_obs, "r") as F_obs:
            num_samples =  len(F_obs[self.obs_domain+' spectrum '+self.dataset])
        return num_samples
    
    def __getitem__(self, idx, dataset=None, return_labels=False, collect_preceeding=False):
        if dataset==None:
            dataset=self.dataset
        
        with h5py.File(self.data_file_obs, "r") as F_obs: 
            if collect_preceeding:
                # Collect all samples before idx
                x = torch.from_numpy(F_obs[self.obs_domain + ' spectrum ' + dataset][:idx,47:].astype(np.float32))
                x_err = torch.from_numpy(F_obs[self.obs_domain + ' error_spectrum ' + dataset][:idx,47:].astype(np.float32))
                if self.collect_x_mask:
                    x_msk = torch.from_numpy(F_obs[self.obs_domain+' bit_mask_spectrum '+dataset][:idx,47:].astype(np.float32))
                else:
                    x_msk = torch.ones(x.size(), dtype=torch.float32)
                if return_labels:
                    y = torch.from_numpy(F_obs[self.obs_domain + ' labels ' + dataset][:idx].astype(np.float32)) 
            else:
                # Collect all samples before idx
                x = torch.from_numpy(F_obs[self.obs_domain + ' spectrum ' + dataset][idx,47:].astype(np.float32))
                x_err = torch.from_numpy(F_obs[self.obs_domain + ' error_spectrum ' + dataset][idx,47:].astype(np.float32))
                if self.collect_x_mask:
                    x_msk = torch.from_numpy(F_obs[self.obs_domain+' bit_mask_spectrum '+dataset][idx,47:].astype(np.float32))
                else:
                    x_msk = torch.from_numpy(np.ones(x.shape).astype(np.float32))
                if return_labels:
                    y = torch.from_numpy(F_obs[self.obs_domain + ' labels ' + dataset][idx].astype(np.float32)) 
            # Normalize the spectra
            x = (x - self.x_mean) / self.x_std
            
            # Add one to the spectra errors to ensure that the minimum
            # error is 1. This helps avoid huge losses.
            x_err += 1
        if return_labels:    
            return {'x':Variable(x), 'x_err':x_err, 'x_msk':x_msk, 'y':Variable(y)} 
        else:    
            return {'x':x, 'x_err':x_err, 'x_msk':x_msk} 
        
        
def train_iter(model, obs_train_batch, synth_train_batch, distance_loss, gan_loss, 
               loss_weight_rec, loss_weight_cc, loss_weight_gen, loss_weight_dis,
               optimizer_rec_and_gen, optimizer_dis, lr_scheduler_rg, lr_scheduler_dis, 
               use_real_as_true, losses_cp, use_cuda):
    
    # Discriminator targets
    batch_ones = torch.ones((len(obs_train_batch['x']), 1), dtype=torch.float32)
    batch_zeros = torch.zeros((len(obs_train_batch['x']), 1), dtype=torch.float32)

    # Switch to GPU
    if use_cuda:
        batch_ones = batch_ones.cuda()
        batch_zeros = batch_zeros.cuda()

    # Train an iteration on the reconstruction and generator processes
    model.rec_and_gen_train_mode()

    # Encoding
    zsh_synth = model.synth_to_z(synth_train_batch['x'].detach())
    if model.use_split:
        zsh_obs, zsp_obs = model.obs_to_z(obs_train_batch['x'].detach())
    else:
        zsh_obs = model.obs_to_z(obs_train_batch['x'].detach())

    # Reconstruction
    x_synthsynth = model.z_to_synth(zsh_synth)
    if model.use_split:
        x_obsobs = model.z_to_obs(zsh_obs, zsp_obs)
    else:
        x_obsobs = model.z_to_obs(zsh_obs)

    # Cross-domain mapping
    if model.use_split:
        # Here we use the z_split from x_obs to generate x_synthobs
        x_synthobs = model.z_to_obs(zsh_synth, zsp_obs)
    else:
        x_synthobs = model.z_to_obs(zsh_synth)
    x_obssynth = model.z_to_synth(zsh_obs)

    # Cycle-Reconstruction
    zsh_obssynth = model.synth_to_z(x_obssynth)
    if model.use_split:
        zsh_synthobs, zsp_synthobs = model.obs_to_z(x_synthobs)
        # Here we again use the original z_split from x_obs to cycle-reconstuct x_obssynthobs
        x_obssynthobs = model.z_to_obs(zsh_obssynth, zsp_obs)
    else:
        zsh_synthobs = model.obs_to_z(x_synthobs)
        x_obssynthobs = model.z_to_obs(zsh_obssynth)
    x_synthobssynth = model.z_to_synth(zsh_synthobs)

    # Run discriminator predictions
    c_synth_fake = model.critic_synth(x_obssynth, zsh_obs)
    if model.use_split:
        c_obs_fake = model.critic_obs(x_synthobs, zsh_synth, zsp_synthobs)
    else:
        c_obs_fake = model.critic_obs(x_synthobs, zsh_synth)

    # Evaluate losses
    loss_rec_synth = distance_loss(pred=x_synthsynth, 
                                   target=synth_train_batch['x'], 
                                   error=synth_train_batch['x_err'], 
                                   mask=synth_train_batch['x_msk'])
    loss_rec_obs = distance_loss(pred=x_obsobs, 
                                 target=obs_train_batch['x'], 
                                 error=obs_train_batch['x_err'], 
                                 mask=obs_train_batch['x_msk'])
    loss_cc_synth = distance_loss(pred=x_synthobssynth, 
                                  target=synth_train_batch['x'], 
                                  error=synth_train_batch['x_err'], 
                                  mask=synth_train_batch['x_msk'])
    loss_cc_obs = distance_loss(pred=x_obssynthobs, 
                                target=obs_train_batch['x'], 
                                error=obs_train_batch['x_err'], 
                                mask=obs_train_batch['x_msk'])
    loss_gen_synth = gan_loss(c_synth_fake, batch_ones)
    loss_gen_obs = gan_loss(c_obs_fake, batch_ones)
    loss_total_rec_gen = (loss_weight_rec*(loss_rec_synth + loss_rec_obs) + 
                          loss_weight_cc*(loss_cc_synth + loss_cc_obs) +
                          loss_weight_gen*(loss_gen_synth + loss_gen_obs))

    # Back propogate
    optimizer_rec_and_gen.zero_grad()
    loss_total_rec_gen.backward()
    # Adjust network weights
    optimizer_rec_and_gen.step()    
    # Adjust learning rate
    lr_scheduler_rg.step()

    losses_cp['rec_synth'].append(loss_rec_synth.data.item())
    losses_cp['rec_obs'].append(loss_rec_obs.data.item())
    losses_cp['cc_synth'].append(loss_cc_synth.data.item())
    losses_cp['cc_obs'].append(loss_cc_obs.data.item())
    losses_cp['gen_synth'].append(loss_gen_synth.data.item())
    losses_cp['gen_obs'].append(loss_gen_obs.data.item())

    # Train an iteration on the discriminator processes
    model.dis_train_mode()

    # Discriminator predictions on true samples
    if use_real_as_true:
        c_synth_real = model.critic_synth(synth_train_batch['x'].detach(), zsh_synth.detach())
        if model.use_split:
            c_obs_real = model.critic_obs(obs_train_batch['x'].detach(), zsh_obs.detach(), zsp_obs.detach())
        else:
            c_obs_real = model.critic_obs(obs_train_batch['x'].detach(), zsh_obs.detach())
    else:
        c_synth_real = model.critic_synth(x_synthsynth.detach(), zsh_synth.detach())
        if model.use_split:
            c_obs_real = model.critic_obs(x_obsobs.detach(), zsh_obs.detach(), zsp_obs.detach())
        else:
            c_obs_real = model.critic_obs(x_obsobs.detach(), zsh_obs.detach())
    # Discriminator predictions on generated samples
    c_synth_fake = model.critic_synth(x_obssynth.detach(), zsh_obs.detach())
    if model.use_split:
        c_obs_fake = model.critic_obs(x_synthobs.detach(), zsh_synth.detach(), zsp_synthobs.detach())
    else:
        c_obs_fake = model.critic_obs(x_synthobs.detach(), zsh_synth.detach())


    loss_dis_real_synth = gan_loss(c_synth_real, batch_ones)
    loss_dis_real_obs = gan_loss(c_obs_real, batch_ones)
    loss_dis_fake_synth = gan_loss(c_synth_fake, batch_zeros)
    loss_dis_fake_obs = gan_loss(c_obs_fake, batch_zeros)

    loss_total_dis = loss_weight_dis*(loss_dis_real_synth + loss_dis_real_obs +
                                      loss_dis_fake_synth + loss_dis_fake_obs)

    # Back propogate
    optimizer_dis.zero_grad()
    loss_total_dis.backward()
    # Adjust network weights
    optimizer_dis.step()    
    # Adjust learning rate
    lr_scheduler_dis.step()

    losses_cp['dis_real_synth'].append(loss_dis_real_synth.data.item())
    losses_cp['dis_real_obs'].append(loss_dis_real_obs.data.item())
    losses_cp['dis_fake_synth'].append(loss_dis_fake_synth.data.item())
    losses_cp['dis_fake_obs'].append(loss_dis_fake_obs.data.item())
    
    return losses_cp

def evaluation_checkpoint(model, obs_val_set, synth_val_set, distance_loss, losses_cp):
    # Evaluate validation set
    model.eval_mode()

    # Encoding
    zsh_synth = model.synth_to_z(synth_val_set['x'].detach())
    if model.use_split:
        zsh_obs, zsp_obs = model.obs_to_z(obs_val_set['x'].detach())
    else:
        zsh_obs = model.obs_to_z(obs_val_set['x'].detach())

    # Cross-domain mapping
    if model.use_split:
        # Here we use the z_split from x_obs to generate x_synthobs
        x_synthobs = model.z_to_obs(zsh_synth, zsp_obs)
    else:
        x_synthobs = model.z_to_obs(zsh_synth)
    x_obssynth = model.z_to_synth(zsh_obs)

    # Cycle-Encoding
    zsh_obssynth = model.synth_to_z(x_obssynth)
    if model.use_split:
        zsh_synthobs, zsp_synthobs = model.obs_to_z(x_synthobs)
    else:
        zsh_synthobs = model.obs_to_z(x_synthobs)


    # Compute max and min of each latent variable
    max_z_sh = torch.max(torch.cat((zsh_synth, zsh_obs, 
                                    zsh_synthobs, zsh_obssynth), 0), 
                         dim=0).values
    min_z_sh = torch.min(torch.cat((zsh_synth, zsh_obs, 
                                    zsh_synthobs, zsh_obssynth), 0), 
                         dim=0).values
    if model.use_split:
        max_z_sp = torch.max(torch.cat((zsp_obs, zsp_synthobs), 0), 
                             dim=0).values
        min_z_sp = torch.min(torch.cat((zsp_obs, zsp_synthobs), 0), 
                             dim=0).values
    # Normalize each latent variable between 0 and 1 across the entire batch
    zsh_synth_norm = (zsh_synth-min_z_sh)/(max_z_sh-min_z_sh)
    zsh_obs_norm = (zsh_obs-min_z_sh)/(max_z_sh-min_z_sh)
    zsh_synthobs_norm = (zsh_synthobs-min_z_sh)/(max_z_sh-min_z_sh)
    zsh_obssynth_norm = (zsh_obssynth-min_z_sh)/(max_z_sh-min_z_sh)
    if model.use_split:
        zsp_obs_norm = (zsp_obs-min_z_sp)/(max_z_sp-min_z_sp)
        zsp_synthobs_norm = (zsp_synthobs-min_z_sp)/(max_z_sp-min_z_sp)
    # Compute error
    zsh_synth_rec_score = torch.mean(torch.abs(zsh_synth_norm-zsh_synthobs_norm))
    zsh_obs_rec_score = torch.mean(torch.abs(zsh_obs_norm-zsh_obssynth_norm))
    zsh_score = torch.mean(torch.abs(zsh_obs_norm-zsh_synth_norm))
    if model.use_split:
        zsp_score = torch.mean(torch.abs(zsp_obs_norm-zsp_synthobs_norm))

    # Generator scores
    x_synthobs_score = distance_loss(pred=x_synthobs, 
                                     target=obs_val_set['x'], 
                                     error=obs_val_set['x_err'], 
                                     mask=obs_val_set['x_msk'])
    x_obssynth_score = distance_loss(pred=x_obssynth, 
                                     target=synth_val_set['x'], 
                                     error=synth_val_set['x_err'], 
                                     mask=synth_val_set['x_msk'])

    losses_cp['zsh_synth_val'].append(zsh_synth_rec_score.data.item())
    losses_cp['zsh_obs_val'].append(zsh_obs_rec_score.data.item())
    losses_cp['zsh_val'].append(zsh_score.data.item())
    if model.use_split:
        losses_cp['zsp_val'].append(zsp_score.data.item())
    losses_cp['x_synthobs_val'].append(x_synthobs_score.data.item())
    losses_cp['x_obssynth_val'].append(x_obssynth_score.data.item())
    
    return losses_cp