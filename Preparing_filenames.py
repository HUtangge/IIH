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
# project_path = r'/Users/getang/Documents/SpaceResearch/spaceflight_associated_neuroocular_syndrome/SANS'
project_path = r'/Users/getang/Documents/EarthResearch/IIH'
import sys
import os
sys.path.append(os.path.join(project_path, 'src'))

import file_search_tool as fs
import csv
import pandas as pd
import numpy as np
import warnings
from datetime import date, timedelta
import re

#%%
info_file = 'IIH_healthy_controls_with BMI.xlsx'
info = pd.read_excel(fs.osnj(project_path, 'info', info_file), sheet_name='data')
info['id'].to_csv(fs.osnj(project_path, 'info', 'controls_names_list.csv'), index=False)

#%% Configuration
info_file = 'samples_idiopathic_intracranial_hypertension_300623.xlsx'
# Parameter_file name should contain cohort information
parameter_file = 'Pseudonyms_MRI_IIH-study.xlsx'
newfile_name = cohort + '_Polar_projection_info.csv' # cohort + '_withinfo_02mmT1_morphometrics.csv'

parameter_file = cohort + '_cnn_predictions.csv'
newfile_name = cohort + '_cnn_predictions_with_info.csv'


#%% Get the file infomation based on the xslx sheets
info = pd.read_excel(fs.osnj(project_path, 'data', info_file), sheet_name=None)
info_all = pd.concat(info.values(), ignore_index=True)   
info_all = info_all.rename({'Age_at_preflight_scan': 'Age-preflight', 'Age_at_return_post1': 'Age-postflight1', 
           'Age_at_return_post2': 'Age-postflight2', 'Age_at_return_post3': 'Age-postflight3',
           'Age_at_return_post4': 'Age-followup'}, axis=1) 

merge_id_vars = ['Age-preflight', 'Age-postflight1', 'Age-postflight2', 'Age-postflight3', 'Age-followup']   

id_vars = list(info_all.columns)

Age_info_all = info_all.melt(
    id_vars = [ele for ele in id_vars if ele not in merge_id_vars],
    value_vars = merge_id_vars,
    value_name = 'Age',
    var_name = 'session'
)

Age_info_all.loc[Age_info_all['subject_ID'].str.count('-') + 1 == 2, 'subject_ID'] = Age_info_all.loc[Age_info_all['subject_ID'].str.count('-') + 1 == 2, 'subject_ID'] + '-f1'
Age_info_all['identity'] = Age_info_all['session'].str.lstrip('Age-') + '-' + Age_info_all['subject_ID'].str.lstrip('sub-') 
Age_info_all['session'] = Age_info_all['session'].str.lstrip('Age-')
Age_info_all = pd.concat([Age_info_all, Age_info_all['identity'].str.split('-', expand=True)], axis = 1)
Age_info_all.rename({1:'group', 2:'mission'}, axis = 1, inplace = True)
Age_info_all.drop([0], axis = 1, inplace = True)
Age_info_all['mission'] = Age_info_all['mission'].fillna('f1')
Age_info_all = pd.concat([Age_info_all, Age_info_all['group'].str.extract(r'(cosmonaut|astronaut|control)(\d+|\w)')], axis = 1)
Age_info_all.drop(['group'], axis = 1, inplace = True)
Age_info_all.rename({0:'group', 1:'number'}, axis = 1, inplace = True)

# Create a Subject variable to identify groups
Age_info_all['subject'] = Age_info_all['group'] + Age_info_all['number']
Age_info_all.drop(['number'], axis = 1, inplace = True)

#%% Get the fileinformation based on the file names
df = pd.read_csv(fs.osnj(project_path, 'data', parameter_file))
df.reset_index(inplace=True)
fl = df['filename'] 
df.drop(['filename'], axis = 1, inplace = True)
if not os.path.exists(fs.osnj(project_path, 'data', newfile_name)):
    print('Creating file ...')
    print(fs.osnj(project_path, 'data', newfile_name))
    with open(fs.osnj(project_path, 'data', newfile_name), 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        all = []
        head = list([fl.name])
        fl = list(fl)
        head.extend(['identity','sequence'])
        sequence = ''
        if re.search('Cosmonaut', parameter_file) != None:
            print('SANS ' + cohort + ' dataset')
            for row in fl:
                row = list([row])
                # Get the session info and the subject info
                identity = re.search(r'_ses-(.*?)\d', row[0]).group(1) + '-' + re.search(r'cosmonaut\d+|astronaut\d+|control[a-zA-Z]|control\d+', row[0]).group(0) 

                # Recoding the followup and control to be same as astronaut coding
                if not re.search('postflight', identity) == None:
                    identity = identity.replace(re.search(r'postflight', identity).group(0), 'postflight1')
                                
                # Add a new parameter for later inter sequence comparison
                if not re.search('fmri', row[0]) == None:
                    sequence = sequence + 'fmri'            
                if not re.search('dti', row[0]) == None:
                    sequence = sequence + 'dti'                
                if len(sequence) > 5:
                    sequence = np.NaN
                    
                # Get the first mission or second mission info
                row.extend([identity + '-f' + re.search(r'(\d+)(fmri|dti)', row[0]).group(1), sequence])
                all.append(row)
                sequence = ''

        else:
            print('SANS ' + cohort + ' dataset')
            for row in fl:
                row = list([row])
                # Get the session info and the subject info
                if len(re.search(r'pre\w+|post\w+|follow\w+|ses-\d+', row[0]).group(0)) > 11:
                    identity = re.split('_', re.search(r'pre\w+|post\w+|follow\w+|ses-\d+', row[0]).group(0))[0] + '-' + re.search(r'cosmonaut\d+|astronaut\d+|control[a-zA-Z]|control\d+', row[0]).group(0)                
                else:
                    identity = re.search(r'pre\w+|post\w+|follow\w+|ses-\d+', row[0]).group(0) + '-' + re.search(r'cosmonaut\d+|astronaut\d+|control[a-zA-Z]|control\d+', row[0]).group(0) 

                # Recoding the followup and control to be same as astronaut coding
                if not re.search('postflight', identity) == None and re.search('postflight[\d]+', identity) == None:
                    identity = identity.replace(re.search(r'postflight', identity).group(0), 'postflight1')
                elif not re.search(r'ses-1', identity) == None:
                    identity = identity.replace(re.search(r'ses-1', identity).group(0), 'preflight')
                elif not re.search(r'ses-2', identity) == None:
                    identity = identity.replace(re.search(r'ses-2', identity).group(0), 'postflight1')
                elif not re.search(r'ses-3', identity) == None:
                    identity = identity.replace(re.search(r'ses-3', identity).group(0), 'postflight3')
                elif not re.search(r'ses-4', identity) == None:
                    identity = identity.replace(re.search(r'ses-4', identity).group(0), 'followup')

                # Add a new parameter for later inter sequence comparison
                if not re.search('fmri', row[0]) == None:
                    sequence = sequence + 'fmri'            
                if not re.search('dti', row[0]) == None:
                    sequence = sequence + 'dti'                
                if len(sequence) > 5:
                    sequence = np.NaN
                    
                # Get the first mission or second mission info
                row.extend([identity, sequence])
                all.append(row)
                sequence = ''
  
        # List to the dataframe
        all = pd.DataFrame(all, columns=head)

        if cohort == 'Cosmonauts':
            # Add all the interested paramters to the filenames_info
            identity = all.identity.str.split(r"\-", expand=True).set_axis(['session', 'subject', 'mission'], axis=1, inplace=False)
            group = identity.subject.str.extract(r'(cosmonaut|astronaut|control)(\d+|\w)').set_axis(['group', 'number'], axis=1, inplace=False)
            days_of_scan = all.filename.str.extract(r'(L\d+)').set_axis(['scandays'], axis=1, inplace=False)
            all = pd.concat([all, identity, group, days_of_scan, df], axis = 1)
            all['mission'] = all['mission'].replace(np.NaN, 'f1')
            all['scandays'] = all['scandays'].replace('N.A.', np.NaN)
        else:
            # Add all the interested paramters to the filenames_info
            all['identity'] = all['identity'] + '-f1'
            identity = all.identity.str.split(r"\-", expand=True).set_axis(['session', 'subject', 'mission'], axis=1, inplace=False)
            group = identity.subject.str.extract(r'(cosmonaut|astronaut|control)(\d+|\w)').set_axis(['group', 'number'], axis=1, inplace=False)
            days_of_scan = all.filename.str.extract(r'(L\d+)').set_axis(['scandays'], axis=1, inplace=False)
            all = pd.concat([all, identity, group, days_of_scan, df], axis = 1)
            all['mission'] = all['mission'].replace(np.NaN, 'f1')
            all['scandays'] = all['scandays'].replace('N.A.', np.NaN)

        all.drop([150, 151], inplace = True)
        all = pd.merge(left = all, right = Age_info_all, how = 'left', on = 'identity')        

        # Remove duplicated values
        if all.T.duplicated().any():
            warnings.warn('Duplicated values')
            duplicated_values = all.columns[np.where(all.T.duplicated())].to_list()
            print("The duplicated columns are ... %s" % duplicated_values)
            all.drop(duplicated_values, axis=1, inplace = True)

        duplicated_columns = [item for item in all.columns if '_x' in item]
        if duplicated_columns and any(item in [s.replace('_x', '_y') for s in duplicated_columns] for item in list(all.columns)):
            # This is for sanity check to get the index of removing rows
            duplicated_columns_filenames = pd.DataFrame()
            for pairs in duplicated_columns:
                different_pairs = all[all[pairs].ne(all[pairs.replace('x', 'y')])]
                duplicated_columns_filenames = pd.concat([duplicated_columns_filenames, different_pairs['filename']], axis = 1)
                duplicated_columns_filenames = duplicated_columns_filenames.rename(columns={'filename': pairs})
            print(duplicated_columns_filenames.T.duplicated())
            print(different_pairs)   
        if any(item in [s.replace('_x', '_y') for s in duplicated_columns] for item in list(all.columns)):
            all.drop([s.replace('x', 'y') for s in duplicated_columns], axis = 1, inplace = True)
        rename_list = {k: v for k, v in zip(duplicated_columns, [s.replace('_x', '') for s in duplicated_columns])}
        all.rename(columns=rename_list, inplace=True)

        # Write the filenames_info.csv
        head = list(all.columns)
        all = all.values.tolist()
        writer.writerow(head)
        writer.writerows(all)
else:
    print('File already exists')
    df_info = pd.read_csv(fs.osnj(project_path, 'data', newfile_name))
    pass


#%% Merge Cosmonauts and Astronauts if neccessary
# I do it separately because the file names are not consistent in the SANS project
if os.path.exists(fs.osnj(project_path, 'data', 'Cosmonauts_withinfo_02mmT1_morphometrics.csv')) and os.path.exists(fs.osnj(project_path, 'data', 'Astronauts_withinfo_02mmT1_morphometrics.csv')):
    print('Combining data sets from Cosmonauts and Astronauts')
    df_cosmonauts = pd.read_csv(fs.osnj(project_path, 'data', 'Cosmonauts_withinfo_02mmT1_morphometrics.csv'))
    df_astronauts = pd.read_csv(fs.osnj(project_path, 'data', 'Astronauts_withinfo_02mmT1_morphometrics.csv'))
    df_pooling = pd.concat([df_cosmonauts, df_astronauts], ignore_index=True)
    df_pooling.to_csv(fs.osnj(project_path, 'data', 'Pooling_withinfo_02mmT1_morphometrics.csv'), index=False)
elif os.path.exists(fs.osnj(project_path, 'data', 'Pooling_withinfo_02mmT1_morphometrics.csv')):
    print('Combined data sets already exist')
    df_pooling = pd.read_csv(fs.osnj(project_path, 'data', 'Pooling_withinfo_02mmT1_morphometrics.csv'))
else:
    print('Please create the files ...')
    print('Cosmonaus_withinfo_02mmT1_morphometrics.csv')
    print('Astronaus_withinfo_02mmT1_morphometrics.csv')
    print('Then run this code again')
    
#%%
if __name__ == "__main__":
    print('Done')
    pass

