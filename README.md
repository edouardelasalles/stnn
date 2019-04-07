# Spatio-Temporal Neural Networks for Space-Time Series Forecasting and Relation Discovery

ICDM 2018 - IEEE International Conference on Data Mining series (ICDM)

[Conference Paper](https://ieeexplore.ieee.org/document/8215543/)

[Journal Extension](https://link.springer.com/article/10.1007/s10115-018-1291-x)

Commands for reproducing synthetic experiments:

## Heat Diffusion
### STNN
`python train_stnn.py --dataset heat --outputdir output_heat --manualSeed 2021 --xp stnn`

### STNN-R(efine)
`python train_stnn.py --dataset heat --outputdir output_heat --manualSeed 5718 --xp stnn_r --mode refine --patience 800 --l1_rel 1e-8`

### STNN-D(iscovery)
`python train_stnn.py --dataset heat --outputdir output_heat --manualSeed 9690 --xp stnn_d --mode discover --patience 1000 --l1_rel 3e-6`


## Modulated Heat Diffusion
### STNN
`python train_stnn.py --dataset heat_m --outputdir output_heat_m --manualSeed 679 --xp stnn`

### STNN-R(efine)
`python train_stnn.py --dataset heat_m --outputdir output_heat_m --manualSeed 3488 --xp stnn_r --mode refine --l1_rel 1e-5`

### STNN-D(iscovery)
`python train_stnn_.py --dataset heat_m --outputdir output_m --xp test --manualSeed 7664 --mode discover --patience 500 --l1_rel 3e-6`

## Data format
The file `heat.csv` contains the raw temperature data. The 200 rows correspond to the 200 timestep, and the 41 columns are the 41 space points.
The file `heat_relations.csv` contains the spatial relation between the 41 space points. It is a 41 by 41 adjacency matrix _A_, where _A(i, j)_ = 1 means that series _i_ is a direct neighbor of series _j_ in space, and is 0 otherwise.
