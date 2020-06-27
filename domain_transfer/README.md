# The Cycle-StarNet Domain Adaptation application


### Code

Before beginning training or utilizing Cycle-StarNet, I recommend reading the [technical write-up](../docs/README.md) on the method. After doing this:
  
  1. Download the synthetic (csn_kurucz.h5) and observed (csn_apogee.h5) training sets from [here](https://www.canfar.net/storage/list/starnet/public/Cycle_SN) and place it in the [data directory](../data/).
  
  2. The model architecture and hyper-parameters are set within configuration file in [the config directory](../configs). For instance, I have already created the [kurucz_to_apogee_1 configuration file](../configs/kurucz_to_apogee_1.ini).
  
  3. Using this model as my example, from the main Cycle_SN directory, you can run `python train_network.py kurucz_to_apogee_1 -v 1000 -ct 15` which will train your model displaying the progress every 1000 batch iterations and saves the model every 15 minutes. This same command will continue training the network if you already have the model saved in the [model directory](../models) from previous training. (Note that the training takes approximately 10 hours on GPU). Alternatively, if operating on compute-canada see [this script](../scripts/kur_to_ap_1.sh) for the training. It allows faster data loading throughout training.
  
  4. Lastly, the [Analysis notebook](./Domain_Transfer_DR14.ipynb) shows a few examples of both quantitative and qualitative analysis of the domain transfer.
  
  5. (Work in progress) the [inference notebook](./Estimating_Stellar_Params.ipynb) shows how to use the network to infer stellar labels through a least squares fitting. This currently does not work though..
