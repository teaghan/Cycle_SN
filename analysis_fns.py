import numpy as np
from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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

def plot_progress(losses, y_lims=[(0,0.15),(0,0.25),(0,2.),(0,0.3)], savename=None):
    
    fig = plt.figure(figsize=(18,9))

    gs = gridspec.GridSpec(2, 2)

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])

    ax1.set_title('Synthetic Distance Losses', fontsize=30)
    ax2.set_title('Observed Distance Losses', fontsize=30)
    ax3.set_title('Adversarial Losses', fontsize=30)
    ax4.set_title(r'$\mathcal{Z}$ Validation Scores', fontsize=30)

    ax1.plot(losses['batch_iters'], losses['rec_synth'],
             label=r'$\mathcal{X}_{synth\rightarrow synth}$')
    ax1.plot(losses['batch_iters'], losses['cc_synth'],
             label=r'$\mathcal{X}_{synth\rightarrow obs\rightarrow  synth}$')
    ax1.plot(losses['batch_iters'], losses['x_obssynth_val'],
             label=r'$\mathcal{X}_{obs\rightarrow synth}$')
    ax1.set_ylabel('Loss',fontsize=25)
    ax1.set_ylim(*y_lims[0])
    
    ax2.plot(losses['batch_iters'], losses['rec_obs'],
             label=r'$\mathcal{X}_{obs\rightarrow obs}$')
    ax2.plot(losses['batch_iters'], losses['cc_obs'],
             label=r'$\mathcal{X}_{obs\rightarrow synth\rightarrow obs}$')
    ax2.plot(losses['batch_iters'], losses['x_synthobs_val'],
             label=r'$\mathcal{X}_{synth\rightarrow obs}$')
    ax2.set_ylabel('Loss',fontsize=25)
    ax2.set_ylim(*y_lims[1])
    
    ax3.plot(losses['batch_iters'], losses['gen_obs'],
             label=r'$\mathcal{X}_{synth\rightarrow obs}$')
    ax3.plot(losses['batch_iters'], losses['gen_synth'],
             label=r'$\mathcal{X}_{obs\rightarrow synth}$')
    ax3.plot(losses['batch_iters'], losses['dis_real_synth'],label=r'C$_{synth}(\mathcal{X}_{real})$')
    ax3.plot(losses['batch_iters'], losses['dis_fake_synth'],label=r'C$_{synth}(\mathcal{X}_{fake})$')
    ax3.plot(losses['batch_iters'], losses['dis_real_obs'],label=r'C$_{obs}(\mathcal{X}_{real})$')
    ax3.plot(losses['batch_iters'], losses['dis_fake_obs'],label=r'C$_{obs}(\mathcal{X}_{fake})$')
    ax3.set_ylabel('Loss',fontsize=25)
    ax3.set_ylim(*y_lims[2])
    
    ax4.plot(losses['batch_iters'], losses['zsh_synth_val'],label=r'$\mathcal{Z}_{sh,synth}$')
    ax4.plot(losses['batch_iters'], losses['zsh_obs_val'],label=r'$\mathcal{Z}_{sh,obs}$')
    ax4.plot(losses['batch_iters'], losses['zsh_val'],label=r'$\mathcal{Z}_{shared}$')
    if 'zsp_val' in losses.keys():
        ax4.plot(losses['batch_iters'], losses['zsp_val'],label=r'$\mathcal{Z}_{split}$ Score')
    ax4.set_ylabel('Score',fontsize=25)
    ax4.set_ylim(*y_lims[3])

    for i, ax in enumerate([ax1, ax2, ax3, ax4]):
        ax.set_xlabel('Batch Iterations',fontsize=25)
        ax.tick_params(labelsize=20)
        ax.legend(fontsize=22, ncol=2)
        ax.grid(True)

    plt.tight_layout()
    
    if savename is not None:
        plt.savefig(savename)
        
    plt.show()
    
def plot_10_samples(wave_grid, x1, x2, test_indices, 
                    labels=[r'$x_{obs}$', r'$x_{obs\rightarrow synth \rightarrow obs}$']):
    # Calculate residulal
    x_resid = (x1-x2)

    plt.close('all')
    # Plot test results
    fig, axes = plt.subplots(30,1,figsize=(70, 30))
    for i, indx in enumerate(test_indices):
        orig, = axes[i*3].plot(wave_grid, x1[indx],c='r')
        axes[i*3].set_ylim((0.5,1.2))
        pred, = axes[1+3*i].plot(wave_grid, x2[indx],c='b')
        axes[1+3*i].set_ylim((0.5,1.2))
        resid, = axes[2+3*i].plot(wave_grid, x_resid[indx],c='g')
        axes[2+3*i].set_ylim((-0.35,0.35))
    plt.subplots_adjust(right=0.75)
    fig.legend([orig, pred, resid],[labels[0],labels[1],labels[0]+r'$ - $'+labels[1]],
              loc='center right', fontsize=90)  
    plt.show()

def plot_sample(wave_grid, x_obs, x_synth, x_synthobs, x_obs_err, x_obs_msk,
                indx, min_wave=15200, max_wave=15550, savename=None):
    plt.close('all')
    # Calculate residulal
    x_resid = (x_obs-x_synthobs)/x_obs_err

    
    
    # Plot test results
    fig, axes = plt.subplots(4,1,figsize=(14,6), sharex=True)
    orig_synth, = axes[0].plot(wave_grid, 
                               np.ma.masked_array(x_synth[indx], x_obs_msk[indx]==0),
                               c='maroon')
    orig_obs, = axes[1].plot(wave_grid, 
                             np.ma.masked_array(x_obs[indx], x_obs_msk[indx]==0), 
                             c='royalblue')
    pred, = axes[2].plot(wave_grid, 
                         np.ma.masked_array(x_synthobs[indx], x_obs_msk[indx]==0), 
                         c='indianred')
    resid, = axes[3].plot(wave_grid, 
                          np.ma.masked_array(x_resid[indx], x_obs_msk[indx]==0), 
                          c='forestgreen')
    axes[3].plot([wave_grid[0], wave_grid[-1]], [1,1], 'k--', lw=1)
    axes[3].plot([wave_grid[0], wave_grid[-1]], [-1,-1], 'k--', lw=1)

    for i in range(4):
        if i==3:
            axes[3].set_ylim((-5,5))
        else:
            axes[i].set_ylim((0.5,1.2))
            axes[i].plot([wave_grid[0], wave_grid[-1]], [1,1], 'k--', lw=1)
        axes[i].tick_params(labelsize=15)
        axes[i].fill_between(wave_grid, y1=-5, y2=5, 
                             where=x_obs_msk[indx]==0,#np.median(x_obs_msk, 0)<0.7, 
                             color='gray', alpha=0.5)


    axes[0].set_xlim((min_wave,max_wave))
    fig.legend([orig_synth, orig_obs, pred, resid],
               [r'$x_{synth}$',r'$x_{obs}$', 
                r'$x_{synth \rightarrow obs}$', 
                r'$\frac{(x_{obs}-x_{synth \rightarrow obs})}{\sigma_{obs}}$'],
              loc='upper center', fontsize=22, ncol=4)
    plt.xlabel(r'Wavelength (\AA)',fontsize=22)
    fig.subplots_adjust(top=0.83, bottom=0.15)

    if savename is not None:
        plt.savefig(savename, transparent=True, pad_inches=0.05)
    plt.show()
    
def plot_spec_resid_density(wave_grid, resid, mask, labels, ylim, hist=True, kde=True,
                            dist_bins=180, hex_grid=300, bias='med', scatter='std',
                            bias_label=r'$\widetilde{{m}}$ \ ',
                            scatter_label=r'$s$ \ ',
                            cmap="ocean_r", savename=None):
    
    xs = np.repeat(wave_grid.reshape(1,wave_grid.shape[0]), len(resid[0]), axis=0)

    bias_resids = []
    scatter_resids = []
    for i in range(len(resid)):
        if bias=='med':
            bias_resids.append(np.median(resid[i][mask==1.]))
        elif bias=='mean':
            bias_resids.append(np.mean(resid[i][mask==1.]))
            
        if scatter=='std':
            scatter_resids.append(np.std(resid[i][mask==1.]))
        elif scatter=='1sigma':
            scatter_resids.append((np.percentile(resid[i][mask==1.],86)-np.percentile(resid[i][mask==1.],16))/2)
        
    fig = plt.figure(figsize=(17, len(resid)*5)) 
    gs = gridspec.GridSpec(len(resid), 2,  width_ratios=[5., 1])
    for i in range(len(resid)):
        ax0 = plt.subplot(gs[i,0])
        ax1 = plt.subplot(gs[i,1])

        if i == 0:
            a = ax0.hexbin(xs, resid[i], gridsize=hex_grid, cmap=cmap,  bins='log')
            cmax = np.max(a.get_array())
        else:
            a = ax0.hexbin(xs, resid[i], gridsize=hex_grid, cmap=cmap,  bins='log', vmax=cmax)

        ax0.set_xlim(wave_grid[0], wave_grid[-1])
        ax0.tick_params(axis='y',
                        labelsize=25,width=1,length=10)
        ax0.tick_params(axis='x',          
                        which='both',     
                        bottom=False,      
                        top=False,         
                        labelbottom=False, width=1,length=10)
        ax0.set_ylabel(labels[i],
                       fontsize=35)
        ax0.set_ylim(ylim)

        sns.distplot(resid[i].flatten(), vertical=True, hist=hist, ax=ax1, kde=kde,
                 rug=False, bins=dist_bins, kde_kws={"lw": 2., "color": a.cmap(cmax/4.), "gridsize": dist_bins}, 
                 hist_kws={"color": a.cmap(cmax*0.6), "alpha":0.5})
        ax1.set_xticks([])
        ax1.tick_params(axis='x',          
                        which='both',     
                        bottom=False,      
                        top=False,         
                        labelbottom=False)   
        ax1.tick_params(axis='y',          
                        which='both',   
                        left=False,     
                        right=True,        
                        labelleft=False,
                        labelright=True,
                        labelsize=25,width=1,length=10)
        ax1.set_ylim(ylim)

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=1)
        ax0.annotate(r'%s=\ %0.4f \ \ %s=\ %0.4f'%(bias_label, bias_resids[i], 
                                                            scatter_label, scatter_resids[i]),
                     xy=(0.3, 0.87), xycoords='axes fraction', fontsize=25, bbox=bbox_props)


    ax0.tick_params(axis='x',
                    bottom=True,
                    labelbottom=True,
                    labelsize=25,width=1,length=10)
    ax0.set_xlabel(r'Wavelength (\AA)',fontsize=30)

    cax = fig.add_axes([0.86, 0.15, .015, 0.72])
    cb = plt.colorbar(a, cax=cax)
    cb.set_label(r'Count', size=30)
    cb.ax.tick_params(labelsize=25,width=1,length=10) 
    fig.subplots_adjust(wspace=0.01, bottom=0.6*(0.5**len(resid)), right=0.78)
    
    if savename is not None:
        plt.savefig(savename)
        
    plt.show()

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

def tsne_domain_analysis2(x_synth, x_obs, x_synthobs,
                          zsh_synth=None, zsh_obs=None,
                         x_obs_label=r'$\mathbf{\mathcal{X}_{obs}}$',
                         perplex=80, savename=None):
    print('Analyzing original spectra')
    # Compare original spectra
    A_txa, A_tya, B_txa, B_tya = run_tsne(x_synth, x_obs, perplex=perplex)

    print('Analyzing observed generated spectra')
    # Compare x_AB to x_B
    AB_txb, AB_tyb, B_txb, B_tyb = run_tsne(x_synthobs, x_obs, perplex=perplex)
    
    if zsh_synth is not None:
        print('Analyzing shared latent-space')
        # Compare shared latent-space representations
        Az_txa, Az_tya, Bz_txa, Bz_tya = run_tsne(zsh_synth, zsh_obs, perplex=perplex)
        fig = plt.figure(figsize=(15, 5)) 
        gs = gridspec.GridSpec(1, 3)
        ax1 = plt.subplot(gs[0,0])
        ax2 = plt.subplot(gs[0,1])
        ax4 = plt.subplot(gs[0,2])
    else:
        fig = plt.figure(figsize=(10, 5)) 
        gs = gridspec.GridSpec(1, 2)
        ax1 = plt.subplot(gs[0,0])
        ax4 = plt.subplot(gs[0,1])

    if zsh_synth is not None:
        ax_lst = [ax1,ax2,ax4]
    else:
        ax_lst = [ax1,ax4]
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
        
    # Remove outliers
    x_diffx = np.abs(B_txa-A_txa)
    y_diffx = np.abs(B_tya-A_tya)
    x_stdx = np.std(x_diffx)
    y_stdx = np.std(y_diffx)
    if zsh_synth is not None:
        x_diffz = np.abs(Bz_txa-Az_txa)
        y_diffz = np.abs(Bz_tya-Az_tya)
        x_stdz = np.std(x_diffz)
        y_stdz = np.std(y_diffz)
        indices = np.where((y_diffx<5*y_stdx)&(x_diffx<5*x_stdx)&(y_diffz<5*y_stdz)&(x_diffz<5*x_stdz))
    else:
        indices = np.where((y_diffx<5*y_stdx)&(x_diffx<5*x_stdx))
    dot_b = ax1.scatter(B_txa[indices], B_tya[indices], marker='o', c='cornflowerblue', alpha=0.2)
    dot_a = ax1.scatter(A_txa[indices], A_tya[indices], marker='o', c='firebrick', alpha=0.2)
    leg1 = ax1.legend((dot_a, dot_b), (r'$\mathbf{\mathcal{X}_{synth}}$', 
                                       x_obs_label), 
                      fontsize=22, frameon=True, fancybox=True, markerscale=2.)
    
    if zsh_synth is not None:
        dot_b = ax2.scatter(Bz_txa[indices], Bz_tya[indices], marker='o', c='cornflowerblue', alpha=0.2)
        dot_a = ax2.scatter(Az_txa[indices], Az_tya[indices], marker='o', c='firebrick', alpha=0.2)
        leg2 = ax2.legend((dot_a, dot_b), (r'$\mathbf{\mathcal{Z}_{synth}}$', 
                                           r'$\mathbf{\mathcal{Z}_{obs}}$'), 
                          fontsize=22, frameon=True, fancybox=True, markerscale=2.)

    dot_b = ax4.scatter(B_txb[indices], B_tyb[indices], marker='o', c='cornflowerblue', alpha=0.2)
    dot_a = ax4.scatter(AB_txb[indices], AB_tyb[indices], marker='o', c='coral', alpha=0.2)
    leg4 = ax4.legend((dot_a, dot_b), (r'$\mathbf{\mathcal{X}_{synth\rightarrow obs}}$', 
                                       x_obs_label), 
                      fontsize=22, frameon=True, fancybox=True, markerscale=2.)

    if zsh_synth is not None:
        leg_lst = [leg1, leg2, leg4]
    else:
        leg_lst = [leg1, leg4]
    for leg in leg_lst:
        leg.get_frame().set_alpha(0.5)
        for lh in leg.legendHandles: 
            lh.set_alpha(1.)

    gs.tight_layout(fig)
    if savename is not None:
        plt.savefig(savename, transparent=True, pad_inches=0.2)
    plt.show()

def tsne_domain_analysis(x_synth, x_obs, x_synthobs, x_obssynth,
                         zsh_synth=None, zsh_obs=None, 
                         x_obs_label=r'$\mathbf{\mathcal{X}_{obs}}$',
                         perplex=80, savename=None):
    print('Analyzing original spectra')
    # Compare original spectra
    A_txa, A_tya, B_txa, B_tya = run_tsne(x_synth, x_obs, perplex=perplex)

    if zsh_synth is not None:
        print('Analyzing shared latent-space')
        # Compare shared latent-space representations
        Az_txa, Az_tya, Bz_txa, Bz_tya = run_tsne(zsh_synth, zsh_obs, perplex=perplex)

    print('Analyzing observed generated spectra')
    # Compare x_AB to x_B
    AB_txb, AB_tyb, B_txb, B_tyb = run_tsne(x_synthobs, x_obs, perplex=perplex)

    print('Analyzing synthetic generated spectra')
    # Compare x_A to x_BA
    A_txc, A_tyc, BA_txc, BA_tybc = run_tsne(x_synth, x_obssynth, perplex=perplex)

    # Plot results
    fig = plt.figure(figsize=(10, 10)) 

    gs = gridspec.GridSpec(2, 4)
    if zsh_synth is not None:
        ax1 = plt.subplot(gs[0, 0:2])
        ax2 = plt.subplot(gs[0,2:])
    else:
        ax1 = plt.subplot(gs[0, 1:3])
    ax3 = plt.subplot(gs[1,0:2])
    ax4 = plt.subplot(gs[1,2:])

    if zsh_synth is not None:
        ax_lst = [ax1,ax2,ax3,ax4]
    else:
        ax_lst = [ax1,ax3,ax4]
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
        
    # Remove outliers
    x_diff = np.abs(B_txa-A_txa)
    y_diff = np.abs(B_tya-A_tya)
    x_std = np.std(x_diff)
    y_std = np.std(y_diff)
    indices = np.where((y_diff<5*y_std)&(x_diff<5*x_std))

    dot_b = ax1.scatter(B_txa[indices], B_tya[indices], marker='o', c='navy', alpha=0.2)
    dot_a = ax1.scatter(A_txa[indices], A_tya[indices], marker='o', c='maroon', alpha=0.2)
    leg1 = ax1.legend((dot_a, dot_b), (r'$\mathbf{\mathcal{X}_{synth}}$', 
                                       x_obs_label), 
                      fontsize=22, frameon=True, fancybox=True, markerscale=2.)

    if zsh_synth is not None:
        dot_b = ax2.scatter(Bz_txa[indices], Bz_tya[indices], marker='o', c='navy', alpha=0.2)
        dot_a = ax2.scatter(Az_txa[indices], Az_tya[indices], marker='o', c='maroon', alpha=0.2)
        leg2 = ax2.legend((dot_a, dot_b), (r'$\mathbf{\mathcal{Z}_{synth}}$', 
                                           r'$\mathbf{\mathcal{Z}_{obs}}$'), 
                          fontsize=22, frameon=True, fancybox=True, markerscale=2.)

    dot_b = ax3.scatter(BA_txc[indices], BA_tybc[indices], marker='o', c='rebeccapurple', alpha=0.2)
    dot_a = ax3.scatter(A_txc[indices], A_tyc[indices], marker='o', c='maroon', alpha=0.2)
    leg3 = ax3.legend((dot_a, dot_b), (r'$\mathbf{\mathcal{X}_{synth}}$', 
                                       r'$\mathbf{\mathcal{X}_{obs\rightarrow synth}}$'), 
                      fontsize=22, frameon=True, fancybox=True, markerscale=2.)

    dot_b = ax4.scatter(B_txb[indices], B_tyb[indices], marker='o', c='navy', alpha=0.2)
    dot_a = ax4.scatter(AB_txb[indices], AB_tyb[indices], marker='o', c='mediumvioletred', alpha=0.2)
    leg4 = ax4.legend((dot_a, dot_b), (r'$\mathbf{\mathcal{X}_{synth\rightarrow obs}}$', 
                                       x_obs_label), 
                      fontsize=22, frameon=True, fancybox=True, markerscale=2.)

    if zsh_synth is not None:
        leg_lst = [leg1, leg2, leg3, leg4]
    else:
        leg_lst = [leg1, leg3, leg4]
    for leg in leg_lst:
        leg.get_frame().set_alpha(0.5)
        for lh in leg.legendHandles: 
            lh.set_alpha(1.)

    gs.tight_layout(fig)
    if savename is not None:
        plt.savefig(savename, transparent=True, pad_inches=0.2)
    plt.show()