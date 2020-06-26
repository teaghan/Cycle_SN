#!/bin/bash

module load python/3.6
source $HOME/torchmed/bin/activate
cp /scratch/obriaint/Cycle_SN/data/csn_kurucz.h5 $SLURM_TMPDIR
cp /scratch/obriaint/Cycle_SN/data/csn_apogee.h5 $SLURM_TMPDIR
cp /scratch/obriaint/Cycle_SN/data/mean_and_std_PAYNE_specs.npy $SLURM_TMPDIR
python /scratch/obriaint/Cycle_SN/train_network.py kurucz_to_apogee_1 -v 1000 -dd $SLURM_TMPDIR/
