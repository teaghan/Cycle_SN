import numpy as np
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def plot_progress(losses, y_lims=[(0,0.05),(0,15),(0,0.13)], use_split=True, savename=None):
    
    fig = plt.figure(figsize=(18,9))

    gs = gridspec.GridSpec(2, 2)

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    ax1.set_title('Training Distance Losses', fontsize=30)
    ax2.set_title('Training GAN Losses', fontsize=30)
    ax3.set_title('Validation Scores', fontsize=30)

    ax1.plot(losses['batch_iters'], losses['rec_synth'],
             label=r'$\mathcal{X}_{synth\rightarrow synth}$')
    ax1.plot(losses['batch_iters'], losses['rec_obs'],
             label=r'$\mathcal{X}_{obs\rightarrow obs}$')
    ax1.plot(losses['batch_iters'], losses['cc_synth'],
             label=r'$\mathcal{X}_{synth\rightarrow obs\rightarrow  synth}$')
    ax1.plot(losses['batch_iters'], losses['cc_obs'],
             label=r'$\mathcal{X}_{obs\rightarrow synth\rightarrow obs}$')
    ax1.set_ylabel('Loss',fontsize=25)
    ax1.set_ylim(*y_lims[0])

    ax2.plot(losses['batch_iters'], losses['gen_obs'],
             label=r'$\mathcal{X}_{synth\rightarrow obs}$')
    ax2.plot(losses['batch_iters'], losses['gen_synth'],
             label=r'$\mathcal{X}_{obs\rightarrow synth}$')
    ax2.plot(losses['batch_iters'], losses['dis_real_synth'],label=r'C$_{synth}(\mathcal{X}_{real})$')
    ax2.plot(losses['batch_iters'], losses['dis_fake_synth'],label=r'C$_{synth}(\mathcal{X}_{fake})$')
    ax2.plot(losses['batch_iters'], losses['dis_real_obs'],label=r'C$_{obs}(\mathcal{X}_{real})$')
    ax2.plot(losses['batch_iters'], losses['dis_fake_obs'],label=r'C$_{obs}(\mathcal{X}_{fake})$')
    ax2.set_ylabel('Loss',fontsize=25)
    ax2.set_ylim(*y_lims[1])
    
    ax3.plot(losses['batch_iters'], losses['x_synthobs_val'],label=r'$\mathcal{X}_{synth\rightarrow obs}$')
    ax3.plot(losses['batch_iters'], losses['x_obssynth_val'],label=r'$\mathcal{X}_{obs\rightarrow synth}$')
    ax3.plot(losses['batch_iters'], losses['zsh_synth_val'],label=r'$\mathcal{Z}_{sh,synth}$')
    ax3.plot(losses['batch_iters'], losses['zsh_obs_val'],label=r'$\mathcal{Z}_{sh,obs}$')
    ax3.plot(losses['batch_iters'], losses['zsh_val'],label=r'$\mathcal{Z}_{shared}$')
    if use_split:
        ax3.plot(losses['batch_iters'], losses['zsp_val'],label=r'$\mathcal{Z}_{split}$ Score')
    ax3.set_ylabel('Score',fontsize=25)
    ax3.set_ylim(*y_lims[2])
    
    print(np.min(losses['x_synthobs_val']), np.min(losses['x_obssynth_val']), np.min(losses['zsh_synth_val']), 
         np.min(losses['zsh_obs_val']), np.min(losses['zsh_val']), end=' ')
    if use_split:
        print(np.min(losses['zsp_val']))
        
    print('\n', np.min(losses['rec_synth']), np.min(losses['rec_obs']), np.min(losses['cc_synth']), 
         np.min(losses['cc_obs']))

    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.set_xlabel('Batch Iterations',fontsize=25)
        ax.tick_params(labelsize=20)
        ax.legend(fontsize=22, ncol=2)
        ax.grid(True)

    plt.tight_layout()
    
    if savename is not None:
        plt.savefig(savename)
        
    plt.show()
    
def combine_apogee_chips(spectrum):
    
    # Define edges of detectors
    blue_chip_begin = 322
    blue_chip_end = 3242
    green_chip_begin = 3648
    green_chip_end = 6048   
    red_chip_begin = 6412
    red_chip_end = 8306 
    
    # Separate combined spectrum into chips
    blue_sp = spectrum[:,blue_chip_begin:blue_chip_end]
    green_sp = spectrum[:,green_chip_begin:green_chip_end]
    red_sp = spectrum[:,red_chip_begin:red_chip_end]
    # Recombine spectra
    return np.hstack((blue_sp,green_sp,red_sp))

def separate_apogee_chips(spectrum):
    
    # Define edges of detectors
    blue_chip_begin = 322
    blue_chip_end = 3242
    green_chip_begin = 3648
    green_chip_end = 6048   
    red_chip_begin = 6412
    red_chip_end = 8306 
    
    # Separate combined spectrum into chips
    blue_sp = spectrum[:,blue_chip_begin:blue_chip_end]
    green_sp = spectrum[:,green_chip_begin:green_chip_end]
    red_sp = spectrum[:,red_chip_begin:red_chip_end]

    return blue_sp,green_sp,red_sp

def apstarwavegrid(log10lambda_o=4.179,  dlog10lambda=6e-6, nlambda=8575):
    wave_grid =  10.**np.arange(log10lambda_o, 
                                log10lambda_o+nlambda*dlog10lambda,
                                dlog10lambda)
    wave_grid = wave_grid.reshape((1,8575))
    blue_wave, green_wave, red_wave = separate_apogee_chips(wave_grid)
    return combine_apogee_chips(wave_grid).reshape((7214,))

def run_tsne(data_a, data_b, perplex):

    m = len(data_a)

    # Combine data
    t_data = np.row_stack((data_a,data_b))

    # Convert data to float64 matrix. float64 is need for bh_sne
    t_data = np.asarray(t_data).astype('float64')
    t_data = t_data.reshape((t_data.shape[0], -1))

    # Run t-SNE    
    vis_data = TSNE(n_components=2, perplexity=perplex).fit_transform(t_data)
    
    # Separate 2D into x and y axes information
    vis_x_a = vis_data[:m, 0]
    vis_y_a = vis_data[:m, 1]
    vis_x_b = vis_data[m:, 0]
    vis_y_b = vis_data[m:, 1]
    
    return vis_x_a, vis_y_a, vis_x_b, vis_y_b

def tsne_domain_analysis(x_synth_val, x_obs_val, zsh_synth, zsh_obs, 
                         x_synthobs, x_obssynth, x_synthobssynth, x_obssynthobs,
                         savename=None):
    print('Analyzing original spectra')
    # Compare original spectra
    A_txa, A_tya, B_txa, B_tya = run_tsne(x_synth_val, x_obs_val, perplex=80)

    print('Analyzing shared latent-space')
    # Compare shared latent-space representations
    Az_txa, Az_tya, Bz_txa, Bz_tya = run_tsne(zsh_synth, zsh_obs, perplex=80)

    print('Analyzing observed generated spectra')
    # Compare x_AB to x_B
    AB_txb, AB_tyb, B_txb, B_tyb = run_tsne(x_synthobs, x_obs_val, perplex=80)

    print('Analyzing synthetic generated spectra')
    # Compare x_A to x_BA
    A_txc, A_tyc, BA_txc, BA_tybc = run_tsne(x_synth_val, x_obssynth, perplex=80)

    print('Analyzing synthetic cycle-reconstructed spectra')
    # Compare x_A to x_ABA
    A_txd, A_tyd, ABA_txd, ABA_tybd = run_tsne(x_synth_val, x_synthobssynth, perplex=80)

    print('Analyzing observed cycle-reconstructed spectra')
    # Compare x_B to x_BAB
    B_txe, B_tye, BAB_txe, BAB_tybe = run_tsne(x_obs_val, x_obssynthobs, perplex=80)

    # Plot results
    fig = plt.figure(figsize=(10, 15)) 

    gs = gridspec.GridSpec(3, 4)
    ax1 = plt.subplot(gs[0, 0:2])
    ax2 = plt.subplot(gs[0,2:])
    ax3 = plt.subplot(gs[1,0:2])
    ax4 = plt.subplot(gs[1,2:])
    ax5 = plt.subplot(gs[2,0:2])
    ax6 = plt.subplot(gs[2,2:])

    ax_lst = [ax1,ax2,ax3,ax4,ax5, ax6]
    for ax in ax_lst:
        ax.tick_params(
            axis='x',         
            which='both',      
            bottom=False,      
            top=False,         
            labelbottom=False)

        ax.tick_params(
            axis='y',         
            which='both',      
            right=False,      
            left=False,         
            labelleft=False)

    dot_b = ax1.scatter(B_txa, B_tya, marker='o', c='navy', alpha=0.2)
    dot_a = ax1.scatter(A_txa, A_tya, marker='o', c='maroon', alpha=0.2)
    leg1 = ax1.legend((dot_a, dot_b), (r'$\mathbf{\mathcal{X}_{synth}}$', 
                                       r'$\mathbf{\mathcal{X}_{obs}}$'), 
                      fontsize=22, frameon=True, fancybox=True, markerscale=2.)

    dot_b = ax2.scatter(Bz_txa, Bz_tya, marker='o', c='navy', alpha=0.2)
    dot_a = ax2.scatter(Az_txa, Az_tya, marker='o', c='maroon', alpha=0.2)
    leg2 = ax2.legend((dot_a, dot_b), (r'$\mathbf{\mathcal{Z}_{synth}}$', 
                                       r'$\mathbf{\mathcal{Z}_{obs}}$'), 
                      fontsize=22, frameon=True, fancybox=True, markerscale=2.)

    dot_b = ax3.scatter(BA_txc, BA_tybc, marker='o', c='rebeccapurple', alpha=0.2)
    dot_a = ax3.scatter(A_txc, A_tyc, marker='o', c='maroon', alpha=0.2)
    leg3 = ax3.legend((dot_a, dot_b), (r'$\mathbf{\mathcal{X}_{synth}}$', 
                                       r'$\mathbf{\mathcal{X}_{obs\rightarrow synth}}$'), 
                      fontsize=22, frameon=True, fancybox=True, markerscale=2.)

    dot_b = ax4.scatter(B_txb, B_tyb, marker='o', c='navy', alpha=0.2)
    dot_a = ax4.scatter(AB_txb, AB_tyb, marker='o', c='mediumvioletred', alpha=0.2)
    leg4 = ax4.legend((dot_a, dot_b), (r'$\mathbf{\mathcal{X}_{synth\rightarrow obs}}$', 
                                       r'$\mathbf{\mathcal{X}_{obs}}$'), 
                      fontsize=22, frameon=True, fancybox=True, markerscale=2.)

    dot_a = ax5.scatter(A_txd, A_tyd, marker='o', c='maroon', alpha=0.2)
    dot_b = ax5.scatter(ABA_txd, ABA_tybd, marker='o', c='sandybrown', alpha=0.1)
    leg5 = ax5.legend((dot_a, dot_b), (r'$\mathbf{\mathcal{X}_{synth}}$', 
                                       r'$\mathbf{\mathcal{X}_{synth\rightarrow obs\rightarrow synth}}$'), 
                      fontsize=22, frameon=True, fancybox=True, markerscale=2.)

    dot_a = ax6.scatter(B_txe, B_tye, marker='o', c='navy', alpha=0.2)
    dot_b = ax6.scatter(BAB_txe, BAB_tybe, marker='o', c='teal', alpha=0.1)
    leg6 = ax6.legend((dot_a, dot_b), (r'$\mathbf{\mathcal{X}_{obs}}$', 
                                       r'$\mathbf{\mathcal{X}_{obs\rightarrow synth\rightarrow obs}}$'), 
                      fontsize=22, frameon=True, fancybox=True, markerscale=2.)

    leg_lst = [leg1, leg2, leg3, leg4, leg5, leg6]
    for leg in leg_lst:
        leg.get_frame().set_alpha(0.5)
        for lh in leg.legendHandles: 
            lh.set_alpha(1.)

    gs.tight_layout(fig)
    if savename is not None:
        plt.savefig(savename, transparent=True, pad_inches=0.2)
    plt.show()