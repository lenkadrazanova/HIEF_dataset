# HIEF_dataset

This script creates the Historical Index of Ethnic Fractionalization from the XXXXX dataset

## Pre-requisites (Windows)

Download and install python for windows (https://www.python.org/downloads/windows/)

Download and install git (for instance from https://gitforwindows.org/)

Open a command prompt (Start > type "cmd" > run command prompt) and type:

```
py -m pip install --upgrade pip
```

Install python virtual environment:

```
py -m pip install --user virtualenv  
```

Create a new python virtual environment and activate it

```
py -m venv python_env  
.\python_env\Scripts\activate 
```

Install additional packages
```
pip install matplotlib pandas scikit-learn xlrd h5py tables openpyxl scipy
```

Download the repository and enter the folder:
```
git clone https://github.com/lenkadrazanova/HIEF_dataset.git  
cd HIEF_dataset
```

Create a folder containing the original dataset
```
mkdir data
```

Copy inside this folder the original xlsx dataset file. The dataset file is available from XXXXXXXXXXXXXX
Name this file as "ethnic fractionalization - original.xlsx" or, otherwise, rename the "filename" variable in the script main.py accordingly. 
Make sure this file is inside the data folder you just created
```
data\ethnic fractionalization - original.xlsx
```

Run the script by typing:
```
python main.py
```

The script creates a new folder "results" containing the HIEF dataset, three additional noisy HIEF dataset (with three types of noise, for data robustness check, see accompanying paper) and three plots comparing the HIEF dataset with each of the three additional noisy HIEF dataset. In  the console, you will read also some statistics about, for instance:

```
Pearsons correlation between original and sigma_1: 0.982
Pearsons correlation between original and sigma_2: 0.974
Pearsons correlation between original and smaller: 1.000
T-test original vs sigma1:  Ttest_indResult(statistic=-0.18606227308299908, pvalue=0.8523980751762262)
T-test original vs sigma2:  Ttest_indResult(statistic=-1.41149232199664, pvalue=0.1581172077171192)
T-test original vs smaller:  Ttest_indResult(statistic=-0.06286910218215389, pvalue=0.9498714542734937)
```

# Citations and contacts

Please cite this work as:

Please forward your comments and questions to lenka.drazanova@eui.eu

# Disclaimer
The scripts in this repository have been created by Lenka Drazanova and are licensed under a GNU GPL v3 License.

In no case shall the authors of this work be liable for any injury, loss, claim, or any direct, indirect, incidental, punitive, special, or consequential damages of any kind, including, without limitation lost profits, lost revenue, lost savings, loss of data, replacement costs, or any similar damages, whether based in contract, tort (including negligence), strict liability or otherwise, arising from your use of any of the models and the material provided here, or for any other claim related in any way to your use of the models and the material provided here, including, but not limited to, any errors or omissions in any content, or any loss or damage of any kind incurred as a result of the use of the models and the material provided. Because some states or jurisdictions do not allow the exclusion or the limitation of liability for consequential or incidental damages, in such states or jurisdictions, our liability shall be limited to the maximum extent permitted by law.
