# no sigmoid
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable

from distutils import util
import numpy as np
    
def build_emulator(model_fn, dim_in=25, num_neurons=300, num_pixel=7214, use_cuda=True):
    
    # Create layers
    model = torch.nn.Sequential(torch.nn.Linear(dim_in, num_neurons),
                                torch.nn.LeakyReLU(0.01),
                                torch.nn.Linear(num_neurons, num_neurons),
                                torch.nn.LeakyReLU(0.01),
                                torch.nn.Linear(num_neurons, num_pixel))
    
    # Load model info
    checkpoint = torch.load(model_fn, map_location=lambda storage, loc: storage)
    y_min = checkpoint['y_min']
    y_max = checkpoint['y_max']
    
    # Load model weights
    model.load_state_dict(checkpoint['Payne'])
    
    # Change to GPU
    if use_cuda:
        model = model.cuda()
        y_min = y_min.cuda()
        y_max = y_max.cuda()
    
    return model, y_min, y_max
    
def init_weights(m):
    """
    Glorot uniform initialization for network.
    """
    if 'conv' in m.__class__.__name__.lower():
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def compute_out_size(in_size, mod):
    """
    Compute output size of Module `mod` given an input with size `in_size`.
    """
    
    f = mod.forward(autograd.Variable(torch.Tensor(1, *in_size)))
    return f.size()[1:]

def same_padding(input_pixels, filter_len, stride=1):
    effective_filter_size_rows = (filter_len - 1) + 1
    output_pixels = (input_pixels + stride - 1) // stride
    padding_needed = max(0, (output_pixels - 1) * stride + 
                         effective_filter_size_rows - input_pixels)
    padding = max(0, (output_pixels - 1) * stride +
                        (filter_len - 1) + 1 - input_pixels)
    rows_odd = (padding % 2 != 0)    
    return padding // 2


def build_encoder(input_filts, conv_filts, conv_strides, conv_filt_lens, activation,
                  out_norm=True, z_filts=25, output_z=True, init=True, use_cuda=True):
    
    layers = []
    
    # Conv layers
    for filts, strides, filter_length in zip(conv_filts, conv_strides, conv_filt_lens):   
        layers.append(nn.Conv1d(input_filts, filts, filter_length, strides))
        layers.append(activation)
        input_filts = filts
       
    if output_z==True:
        # Latent output
        layers.append(nn.Conv1d(input_filts, z_filts, 1, 1))
        if out_norm:
            layers.append(nn.InstanceNorm1d(z_filts))
    model = torch.nn.Sequential(*layers)
    
    if init:
        # Initialize weights and biases
        model.apply(init_weights)
        
    if use_cuda:
        # Switch to GPU
        model = model.cuda()

    return model

def build_decoder(input_filts, conv_filts, conv_strides, conv_filt_lens, activation,
                  output_x=True, init=True, use_cuda=True):
    
    layers = []
    
    # Conv layers (reverse order of encoders)
    for filts, strides, filter_length in zip(reversed(conv_filts), reversed(conv_strides), reversed(conv_filt_lens)):
        #if strides>1:
        layers.append(nn.ConvTranspose1d(input_filts, filts, filter_length, strides))
        #else:      
        #    layers.append(nn.Conv1d(input_filts, filts, filter_length, strides))
        layers.append(activation)
        input_filts = filts
       
    if output_x==True:
        # Spectrum output
        layers.append(nn.Conv1d(input_filts, 1, 1, 1))
    model = torch.nn.Sequential(*layers)
    
    if init:
        # Initialize weights and biases
        model.apply(init_weights)
        
    if use_cuda:
        model = model.cuda()

    return model

def build_discriminator(input_filts, conv_filts, conv_strides, conv_filt_lens, activation,
                        padding=0, init=True, use_cuda=True):
    
    layers = []
    
    # Spectra conv layers
    for filts, strides, filter_length in zip(conv_filts, conv_strides, conv_filt_lens):
            layers.append(nn.Conv1d(input_filts, filts, filter_length, strides, padding=padding))
            layers.append(activation)
            input_filts = filts
       
    model = torch.nn.Sequential(*layers)
    
    if init:
        # Initialize weights and biases
        model.apply(init_weights)
        
    if use_cuda:
        model = model.cuda()

    return model

class Discrimninator(nn.Module):

    def __init__(self, input_filts_z, num_pixels, z_dim, 
                 conv_filts_x, conv_strides_x, conv_filt_lens_x, 
                 conv_filts_z, conv_strides_z, conv_filt_lens_z,
                 activ_fn, init=True, use_cuda=True):
                
        super(Discrimninator, self).__init__()
        
        # Layers applied to spectra
        self.discriminator_x = build_discriminator(1, conv_filts_x, 
                                                   conv_strides_x, 
                                                   conv_filt_lens_x, 
                                                   activ_fn, init=init,
                                                   use_cuda=use_cuda)
        # Padding for latent-variables to keep the same shape
        if conv_filt_lens_z[0]>1:
            padding = same_padding(z_dim, conv_filt_lens_z[0], stride=conv_strides_z[0])
        else:
            padding = 0
        
        # Layers applied to latent variables
        self.discriminator_z = build_discriminator(input_filts_z, 
                                                   conv_filts_z, 
                                                   conv_strides_z, 
                                                   conv_filt_lens_z, 
                                                   activ_fn, 
                                                   padding=padding, init=init,
                                                   use_cuda=use_cuda)
        
        # Calculate output sizes
        dis_x_output_shape = compute_out_size((1,num_pixels), self.discriminator_x)
        dis_z_output_shape = compute_out_size((input_filts_z, z_dim), self.discriminator_z)
        fc_input_dim = np.prod(list(dis_x_output_shape))+np.prod(list(dis_z_output_shape))
        
        # Output layer
        self.fc = nn.Linear(fc_input_dim, 1)
        self.activ_out = torch.nn.Sigmoid()
        
    def forward(self, x, z):
        cur_in1 = self.discriminator_x(x).view(x.size(0), -1)
        cur_in2 = self.discriminator_z(z).view(x.size(0), -1)
        cur_in1 = torch.cat((cur_in1, cur_in2), 1)
        cur_in1 = self.fc(cur_in1)
        cur_in1 = self.activ_out(cur_in1)
        return cur_in1
                                      

class CycleSN(nn.Module):

    def __init__(self, architecture_config, emulator_fn=None, use_cuda=True):
                
        super(CycleSN, self).__init__()
                
        # Read configuration
        num_pixels = int(architecture_config['num_pixels'])
        activation = architecture_config['activation']
        conv_filts_ae_dom = eval(architecture_config['conv_filts_ae_dom'])
        conv_filt_lens_ae_dom = eval(architecture_config['conv_filt_lens_ae_dom'])
        conv_strides_ae_dom = eval(architecture_config['conv_strides_ae_dom'])
        conv_filts_ae_sh = eval(architecture_config['conv_filts_ae_sh'])
        conv_filt_lens_ae_sh = eval(architecture_config['conv_filt_lens_ae_sh'])
        conv_strides_ae_sh = eval(architecture_config['conv_strides_ae_sh'])
        conv_filts_ae_sp = eval(architecture_config['conv_filts_ae_sp'])
        conv_filt_lens_ae_sp = eval(architecture_config['conv_filt_lens_ae_sp'])
        conv_strides_ae_sp = eval(architecture_config['conv_strides_ae_sp'])
        enc_out_norm = bool(util.strtobool(architecture_config['enc_out_norm']))
        shared_z_filters = int(architecture_config['shared_z_filters'])
        split_z_filters = int(architecture_config['split_z_filters'])
        conv_filts_dis_x = eval(architecture_config['conv_filts_dis_x'])
        conv_strides_dis_x = eval(architecture_config['conv_strides_dis_x'])
        conv_filt_lens_dis_x = eval(architecture_config['conv_filt_lens_dis_x'])
        conv_filts_dis_z = eval(architecture_config['conv_filts_dis_z'])
        conv_strides_dis_z = eval(architecture_config['conv_strides_dis_z'])
        conv_filt_lens_dis_z = eval(architecture_config['conv_filt_lens_dis_z'])
        # This variable is used for spectra fitting only:
        self.cur_z_sp = None
        self.x_mean = None
        self.x_std = None
                
        # Whether or not to use a split latent-space
        if split_z_filters>0:
            self.use_split = True
        else:
            self.use_split = False
        
        # Create emulator
        (self.emulator, self.y_min, self.y_max) = build_emulator(model_fn=emulator_fn, 
                                                                 use_cuda=use_cuda)
        
        # Define activation function
        if activation.lower()=='sigmoid':
            activ_fn = torch.nn.Sigmoid()
        elif activation.lower()=='leakyrelu':
            activ_fn = torch.nn.LeakyReLU(0.1)
        elif activation.lower()=='relu':
            activ_fn = torch.nn.ReLU()
        
        # Build encoding networks
        self.encoder_synth = build_encoder(1, conv_filts_ae_dom, 
                                           conv_strides_ae_dom, 
                                           conv_filt_lens_ae_dom, 
                                           activ_fn,
                                           output_z=False, init=True,
                                           use_cuda=use_cuda)
        self.encoder_obs = build_encoder(1, conv_filts_ae_dom, 
                                         conv_strides_ae_dom, 
                                         conv_filt_lens_ae_dom, 
                                         activ_fn,
                                         output_z=False, init=True,
                                         use_cuda=use_cuda)
        self.encoder_sh = build_encoder(conv_filts_ae_dom[-1], 
                                        conv_filts_ae_sh, 
                                        conv_strides_ae_sh, 
                                        conv_filt_lens_ae_sh, 
                                        activ_fn,
                                        out_norm=enc_out_norm,
                                        z_filts=shared_z_filters,
                                        output_z=True, init=True,
                                        use_cuda=use_cuda)
        if self.use_split:
            self.encoder_sp = build_encoder(conv_filts_ae_dom[-1], 
                                            conv_filts_ae_sp, 
                                            conv_strides_ae_sp, 
                                            conv_filt_lens_ae_sp, 
                                            activ_fn,
                                            out_norm=enc_out_norm,
                                            z_filts=split_z_filters,
                                            output_z=True, init=True,
                                            use_cuda=use_cuda)
        # Build decoding networks    
        self.decoder_sh = build_decoder(shared_z_filters, 
                                        conv_filts_ae_sh, 
                                        conv_strides_ae_sh, 
                                        conv_filt_lens_ae_sh, 
                                        activ_fn, output_x=False, init=True,
                                        use_cuda=use_cuda)
        if self.use_split:
            self.decoder_sp = build_decoder(split_z_filters, 
                                            conv_filts_ae_sp, 
                                            conv_strides_ae_sp, 
                                            conv_filt_lens_ae_sp, 
                                            activ_fn, output_x=False, init=True,
                                            use_cuda=use_cuda)
        # Calculate the number of input filters for the domain decoders
        if len(conv_filts_ae_sh)>0:
            dec_synth_z_filts = conv_filts_ae_sh[0]
            if self.use_split:
                dec_obs_z_filts = conv_filts_ae_sh[0]+conv_filts_ae_sp[0]
            else:
                dec_obs_z_filts = conv_filts_ae_sh[0]
        else:
            dec_synth_z_filts = shared_z_filters
            if self.use_split:
                dec_obs_z_filts = shared_z_filters+split_z_filters
            else:
                dec_obs_z_filts = shared_z_filters
        self.decoder_synth = build_decoder(dec_synth_z_filts, 
                                           conv_filts_ae_dom, 
                                           conv_strides_ae_dom, 
                                           conv_filt_lens_ae_dom, 
                                           activ_fn, output_x=True, init=True,
                                           use_cuda=use_cuda)
        
        
        self.decoder_obs = build_decoder(dec_obs_z_filts, 
                                         conv_filts_ae_dom, 
                                         conv_strides_ae_dom, 
                                         conv_filt_lens_ae_dom, 
                                         activ_fn, output_x=True, init=True,
                                         use_cuda=use_cuda)
        
        # Infer output shapes of each model
        self.enc_interm_shape = compute_out_size((1,num_pixels), self.encoder_synth)
        self.z_sh_shape = compute_out_size(self.enc_interm_shape, self.encoder_sh)
        if self.use_split:
            self.z_sp_shape = compute_out_size(self.enc_interm_shape, self.encoder_sp)
        self.dec_interm_shape = compute_out_size(self.z_sh_shape, self.decoder_sh)

        # Build discriminator networks
        self.discriminator_synth = Discrimninator(self.z_sh_shape[0],
                                                    num_pixels,
                                                    self.z_sh_shape[1],
                                                    conv_filts_dis_x, 
                                                    conv_strides_dis_x, 
                                                    conv_filt_lens_dis_x, 
                                                    conv_filts_dis_z, 
                                                    conv_strides_dis_z, 
                                                    conv_filt_lens_dis_z,
                                                    activ_fn, init=True,
                                                    use_cuda=use_cuda)
        if self.use_split:
            dis_obs_z_filts = self.z_sh_shape[0]+self.z_sp_shape[0]
        else:
            dis_obs_z_filts = self.z_sh_shape[0]
            
        self.discriminator_obs = Discrimninator(dis_obs_z_filts,
                                                    num_pixels,
                                                    self.z_sh_shape[1],
                                                    conv_filts_dis_x, 
                                                    conv_strides_dis_x, 
                                                    conv_filt_lens_dis_x, 
                                                    conv_filts_dis_z, 
                                                    conv_strides_dis_z, 
                                                    conv_filt_lens_dis_z,
                                                    activ_fn, init=True,
                                                    use_cuda=use_cuda)
            
        
    def y_to_synth(self, y, use_cuda=True):
        if use_cuda:
            y = y.cuda()
            y_min = self.y_min.cuda()
            y_max = self.y_max.cuda()
        else:
            y = y.cpu()
            y_min = self.y_min.cpu()
            y_max = self.y_max.cpu()
            
        y = (y - y_min)/(y_max-y_min) - 0.5
        return self.emulator(y)
    
    def synth_to_z(self, x):        
        return self.encoder_sh(self.encoder_synth(x.unsqueeze(1)))
    
    def z_to_synth(self, z_sh):
        return self.decoder_synth(self.decoder_sh(z_sh)).squeeze(1)
    
    def obs_to_z(self, x):
        interm = self.encoder_obs(x.unsqueeze(1))
        if self.use_split:
            return self.encoder_sh(interm), self.encoder_sp(interm)
        else:
            return self.encoder_sh(interm)
        
    def z_to_obs(self, z_sh, z_sp=None):
        if self.use_split:
            return self.decoder_obs(torch.cat((self.decoder_sh(z_sh), 
                                          self.decoder_sp(z_sp)), 1)).squeeze(1)
        else:
            return self.decoder_obs(self.decoder_sh(z_sh)).squeeze(1)               
                               
    def critic_synth(self, x, z_sh):
        return self.discriminator_synth(x.unsqueeze(1), z_sh)
    
    def critic_obs(self, x, z_sh, z_sp=None):
        if self.use_split:
            return self.discriminator_obs(x.unsqueeze(1), torch.cat((z_sh, z_sp), 1))
        else:
            return self.discriminator_obs(x.unsqueeze(1), z_sh)
        
    def y_to_obs(self, y):
        '''For spectra fitting. Assumes the labels are already normalized and 
        you have assigned a value for self.cur_z_sp'''
        # Produce synthetic spectrum
        x = self.emulator(y)
        # Normalize the spectrum
        x = (x - self.x_mean) / self.x_std
        # Only select last 7167 pixels
        x = x[:,47:]
        # Produce shared z
        z_sh = self.synth_to_z(x)
        # Produce observed spectrum
        return self.z_to_obs(z_sh, self.cur_z_sp)
    
    def y_to_synth_norm(self, y):
        '''For spectra fitting in synth domain. Assumes the labels are already normalized.'''
        # Produce synthetic spectrum
        x = self.emulator(y)
        # Normalize the spectrum
        x = (x - self.x_mean) / self.x_std
        # Only select last 7167 pixels
        x = x[:,47:]
        return x
        
    def rec_and_gen_train_mode(self):
        self.emulator.eval()
        self.encoder_synth.train()
        self.encoder_obs.train()
        self.encoder_sh.train()
        if self.use_split:
            self.encoder_sp.train()
        self.decoder_synth.train()
        self.decoder_obs.train()
        self.decoder_sh.train()
        if self.use_split:
            self.decoder_sp.train()
        self.discriminator_synth.eval()
        self.discriminator_obs.eval()
        
    def dis_train_mode(self):
        self.emulator.eval()
        self.encoder_synth.eval()
        self.encoder_obs.eval()
        self.encoder_sh.eval()
        if self.use_split:
            self.encoder_sp.eval()
        self.decoder_synth.eval()
        self.decoder_obs.eval()
        self.decoder_sh.eval()
        if self.use_split:
            self.decoder_sp.eval()
        self.discriminator_synth.train()
        self.discriminator_obs.train()
        
    def eval_mode(self):
        self.emulator.eval()
        self.encoder_synth.eval()
        self.encoder_obs.eval()
        self.encoder_sh.eval()
        if self.use_split:
            self.encoder_sp.eval()
        self.decoder_synth.eval()
        self.decoder_obs.eval()
        self.decoder_sh.eval()
        if self.use_split:
            self.decoder_sp.eval()
        self.discriminator_synth.eval()
        self.discriminator_obs.eval()
    
    def forward(self, x):
        return x