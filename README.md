
   
Continuous Release of Data Streams under both Centralized and Local Differential Privacy
=====

This repository provides code to accompany the paper [Continuous Release of Data Streams under both Centralized and Local Differential Privacy](https://dl.acm.org/doi/pdf/10.1145/3460120.3484750) which was accepted to ACM Conference on Computer and Communications Security (CCS) 2021. (the arxiv version 

Update: 

- the arxiv version https://arxiv.org/abs/2005.11753 incorporates some fixes from the CCS version.
- 'hm' (and other LDP estimators) was mistakenly deleted ([source](https://github.com/dp-cont/dp-cont/blob/5a0e1d6f833ae5527c30f4df49f720667c9cad63/range_estimator/hm.py) here). I may not have the bandwidth to fix it. Feel free to submit a pull request.


Installing
=====

The code should work with Python 3 plus common libraries such as numpy, pandas, pickle, scipy, matplotlib.  

Project Organization
=====

- estimator: the whole estimator (this folder contains different variants); it calls percentile_estimator and hierarchy_estimator 
- percentile_estimator: estimate the percentile using the first $m$ values
- hierarchy_estimator: working on the remaining data  
***
- results: folder for holding the results
- figures: currently the figures are output the overleaf destination directly
- data
- users: for processing the data
***
- main.py: the main file of the project
- evaluator.py: measure the accuracy/utility of the results from main.py  
- recorder.py: writing experimental results to local files within the results folder 

Reproducing Results
=====

1. go into main.py
2. follow the scripts in the comments of main.py: uncomment some to get desired results
3. go into figures/draw_script.py
4. follow the scripts in the end (comments) of figures/draw_script.py to draw figures


Building on this Work
=====

We encourage others to improve our work and to integrate it with their own applications. We provide it under the MIT license.  Contact: Tianhao Wang at tianhao@virginia.edu.


