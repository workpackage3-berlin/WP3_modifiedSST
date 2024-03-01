# modified_StopSignal_Task
 A modified version of the Stop-Signal Task, to evaluate both proactive and reactive response inhibition


## Getting Started

These instructions will get you a copy of the project up and running on your local machine. 

#### Repository
* GUI: use a git-manager of preference, and clone: https://github.com/workpackage3-berlin/WP3_modifiedSST.git
* Command line:
    - set working directory to desired folder and run: ```git clone https://github.com/workpackage3-berlin/WP3_modifiedSST.git```
    - to check initiated remote-repo link, and current branch: ```cd WP3_modifiedSST```, ```git init```, ```git remote -v```, ```git branch``` (switch to branch main e.g. with git checkout main)

#### Environment
##### task environment:
* Anaconda prompt: you can easily install the required environment from your Anaconda prompt:
    - navigate to repo directory, e.g.: ```cd Users/USERNAME/Research/WP3_modifiedSST```
    - ```conda env create -n psychopy -f psychopy-env.yml``` (Confirm Proceed? with ```y```)
    - ```conda activate psychopy```
    - ```git init```

##### tmsi environment:
    - navigate to repo directory, e.g.: ```cd Users/USERNAME/Research/WP3_modifiedSST```
    - ```conda create --name tmsi python==3.9.18 pyqtgraph==0.12.3 pandas==2.1.4``` (Confirm Proceed? with ```y```)
    - ```conda activate tmsi```
    - ```pip install edflib-Python==1.0.8 mne==1.6.0 pylsl==1.16.2 pyside2==5.15.2.1 pyxdf==1.16.5```

