# The Cycle-StarNet New Lines Project

One application of the Cycle-StarNet is to utilize the correlations that the network has found in order to fill in the missing pieces of our theoretical models. To do this, we train the Cycle-StarNet with a _synthetic domain_ and _observed domain_. The network will learn correlations between known information (i.e. the absorption lines in our synthetic spectra) and currently unknown information (for instance, spectral lines in our observed spectra for which we do not know the source). Since our synthetic spectra are generated from stellar labels (including chemical abundances) we form a link between these chemical abundances and the unidentified absorption lines. Next, we investigate the correlations that the network has found to extract intuition from the model and possibly identify these missing pieces in our theoretical models.

### Contents

[Mock Dataset](#mock-dataset)

   - [Getting Started](#code)
   

## Mock Dataset

We first test our method with a "mock dataset" that is meant to mimic a real-world application. For this, we will mask absorption lines in our synthetic domain and try to let the network correlate these lines - that are present in our observed domain - to information found in the synthetic spectra. To create a controlled setting, we will generate our observed spectra using The Payne and add noise to the data. The synthetic spectra will also be generated from The Payne, but about 30% of the lines will be set to the continuum level.

### Code

Before beginning training or utilizing Cycle-StarNet, I recommend reading the [technical write-up](./docs/README.md) on the method. After doing this:
  
  1. First, the line mask is created in [this notebook](./Create_Line_Mask.ipynb).
  
  2. Next, we generate our mock observed dataset [here](./Generate_Observed_Payne_Domain.ipynb).
  
  3. The model architecture and hyper-parameters are set within configuration file in [the config directory](../configs). For instance, I have already created the [paynetopayne_nozsplit_1 configuration file](../configspaynetopayne_nozsplit_1.ini). This model does not utilize the split latent-space method, only shared latent-variables.
  
  2. Using this model as my example, from the main Cycle_SN directory, you can run `python train_network.py configspaynetopayne_nozsplit_1 -v 1000 -ct 15` which will train your model displaying the progress every 1000 batch iterations and saves the model every 15 minutes. This same command will continue training the network if you already have the model saved in the [model directory](../models) from previous training.
  
  3. The [Domain Transfer Analysis notebook](./Domain_Transfer_paynetopayne_nozsplit.ipynb) takes you through the steps of analyzing the StarNet-Cycle to ensure that the model can transfer spectra from one domain to the other.
  
  4. Lastly, the [Tracking New Lines notebook](Track_Lines_paynetopayne_nozsplit.ipynb) shows the method used to identify the missing lines.
