0. install and use anaconda
    1. create environment in the server: conda env create -f environment.yml; source activate db_code
    2. share the environment: conda env export > environment.yml
    3. or create a new one: conda create -n myenv; conda activate myenv; conda install numpy, pandas, scipy, matplotlib 
1. if using pycharm
    1. create remote environment
    2. upload to server
2. project structure
    - estimator: the whole thing estimator
    - percentile_estimator
    - hierarchy_estimator
    - evo_percentile_estimator: several method to update percentile estimations; deprecated now 
    ***
    - results 
    - figures (currently the figures are output the the overleaf destination directly)
    - data
    - users (for processing the data)
    ***
    - main
    - evaluator
    - recorder (writing to result)
3. how to run the code
    1. entry is main.py
    2. follow the scripts in the end (comments) of main.py
    3. entry to draw figures is figures/draw_script.py
    4. follow the scripts in the end (comments) of figures/draw_script.py to draw figures
