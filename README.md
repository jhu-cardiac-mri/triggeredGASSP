# triggeredGASSP
This repository contains the GPI nodes and network used to reconstruct phase contrast MRI images presented in the manuscript "Phase contrast coronary blood velocity mapping with both high temporal and spatial resolution using triggered Golden Angle rotated Spiral k-t Sparse Parallel imaging (GASSP) with shifted binning".

Example data can be downloaded at

    https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/mschar3_jh_edu/EnPgpmVd7RtKiHAk-eWcRhYB7TLz-mKoSrQ5Yh5MbLzobA?e=wOWo8h
    
and should be copied to the folder called "example data".

# GPI version
Use GPI version 1.1 to run the nodes as they are, or adjust accordingly for any required adaptations.

# Python packages
additionally to what is included with GPI, add the following python packages:
- scikit-image

# some nodes need to be compiled
use

    $ gpi_make -all

in 

    /iterate
    
    /iterate/GPI
    
    /spiral

# GPU
Some nodes have a GPU version that requires CUPY. 
If using the GPU, we recommend a card with >10GB of RAM to run the example data (We are using NVIDIA GeForce GTX 1080 Ti).

# Installation
Clone this repo into your ~/gpi directory by entering:

    $ git clone https://github.com/jhu-cardiac-mri/triggeredGASSP.git triggeredGASSP

Rescan for new nodes or restart GPI to include the new nodes in the GPI library.

