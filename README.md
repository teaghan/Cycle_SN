<p align="left">
  <img width="300" height="80" src="./figures/full_logo.png">
</p>

# Cycle-StarNet

## Dependencies

-[PyTorch](http://pytorch.org/): `pip install torch torchvision`

-h5py: `pip install h5py`

-scikit-learn: `pip install -U scikit-learn`

## Generative and Interpretable Deep Learning for Stellar Spectra


This project aims to bridge the gap between two different sets of stellar spectra. Although the underlying physics that produces the two sets may be the same, the data can visually appear very different for a variety of reasons. Cycle-StarNet is meant to determine the commonalities between two sets of spectra (ie. the physical parameters) and learn how to transform one set of data into the other. 

<p align="center">
  <img width="400" height="300" src="./figures/diagram.png">     
</p>      

<p align="center">
  <img width="800" height="550" src="./figures/Architecture.png"> 
</p>      
                                     
    
<p align="center"><b>Figure 1</b>: Overview of the proposed method (top) and a simplified diagram of the Cycle-StarNet Architecture (bottom).<p align="center"> 
                                   

<p align="center">
  <img width="900" height="300" src="./figures/synth_to_obs.png">
</p>

<p align="center"><b>Figure 2</b>: Examples of two spectra (taken from our New Lines Project) from opposite domains that have the same stellar parameters. When mapping the synthetic spectrum (which has an incomplete line list) to the observed domain, the resulting spectrum is a much better fit to the observed spectrum and the missing information is now present in the transferred spectrum.<p align="center"> 


## Getting Started ##

For an in depth explanation on the method and the projects shown here, please checkout our [paper](https://arxiv.org/abs/2007.03109). 

To get started with our application of Domain Adaption with real observed spectra, take a look [here](./domain_transfer/).

To get started with our New Lines Project, take a look [here](./new_lines_project/).

## Citing this work

Here's the BibTeX:

```
@article{OBriain_2021,
	doi = {10.3847/1538-4357/abca96},
	url = {https://doi.org/10.3847/1538-4357/abca96},
	year = 2021,
	month = {jan},
	publisher = {American Astronomical Society},
	volume = {906},
	number = {2},
	pages = {130},
	author = {Teaghan O'Briain and Yuan-Sen Ting and S{\'{e}}bastien Fabbro and Kwang M. Yi and Kim Venn and Spencer Bialek},
	title = {Cycle-{StarNet}: Bridging the Gap between Theory and Data by Leveraging Large Data Sets},
	journal = {The Astrophysical Journal}
}
```
