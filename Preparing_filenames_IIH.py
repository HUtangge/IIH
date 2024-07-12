"""
This is a file to connect the filenames with the information of cosmonauts
You need to prepare the two parts
1. The information xlsx file which contains the information of cosmonauts/astronauts/controls
2. The parameterization file which contains the raw filenames

parameter to configure:
1. project_path: the path of the project
2. info_file: the name of the xlsx file which contains the information of cosmonauts/astronauts/controls
3. cohort: 'Cosmonauts' or 'Astronauts', for the parameter_file and newfile_name 
    parameter_file: the name of the parameterization file which contains the raw filenames
    newfile_name: the name of the new file

"""

#%%
project_path = r'/Users/getang/Documents/EarthResearch/IIH'
# project_path = r'/Users/getang/Documents/SpaceResearch/spaceflight_and_aging/SpaceAging'
import sys
import os
sys.path.append(os.path.join(project_path, 'src'))

import file_search_tool as fs
import numpy as np
import pandas as pd

#%% Configuration
save = False
test = False
info_file = 'IIH_with_BMI.xlsx'
parameter_file = 'IIH_Metrics.csv'
newfile_name = 'withinfo_IIH_Metrics_20231127'
colnames_info = ['id', 'group', 'birthdate', 'Gender', 'Height', 'Weight', 'age']

#%% Get the file infomation based on the xslx sheets
info = pd.read_excel(fs.osnj(project_path, 'info', info_file), sheet_name=None)
info_all = pd.concat(info.values(), ignore_index=True)   
info_all = info_all[colnames_info]

#%% Get the fileinformation based on the file names
df = pd.read_csv(fs.osnj(project_path, 'data', parameter_file))
df_with_info = pd.merge(left = df, right = info_all, how = 'left', on = 'id')        
df_with_info = df_with_info.replace(0, np.nan)
df_with_info = df_with_info.replace('?', np.nan)

if test:
    df_with_info = pd.merge(left = df, right = info_all, how = 'left', on = 'id', indicator=True)        

if save:   
    print(f'Saving the Metrics to the {project_path}')
    df_with_info.to_csv(os.path.join(project_path, 'data', f'{newfile_name}.csv'),index = False)
    df_with_info.to_excel(os.path.join(project_path, 'data', f'{newfile_name}.xlsx'), index = False)

#%%
if __name__ == "__main__":
    print('Done')
    pass

