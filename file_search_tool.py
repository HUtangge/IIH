# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 15:19:18 2021

@author: getang
"""
import numpy as np
import os
import os
from os.path import join as osj
from os.path import normpath as osn
import fnmatch
import re

def locateFiles(pattern, root_path, level=0, tailOnly=False, sort=True):
    fl = np.array([])
    #for root, dirs, files in os.walk(root_path):
    for root, dirs, files in walklevel(root_path, level):
        for filename in fnmatch.filter(files, pattern):
            # print(os.path.join(root_path, filename))
            if tailOnly:
                fl = np.append(fl,filename)
            else:
                fl = np.append(fl,osnj(root, filename))
    if sort:
        fl = natural_sort(fl)
    return fl

# depends on pandas
# depends on pandas
def locateFilesDf(pattern, root_path, level=0, tailOnly=False, sorted=True):
    import pandas as pd
    ffl = locateFiles(pattern, root_path, level=level, tailOnly=tailOnly)
    #for root, dirs, files in os.walk(root_path):
    # create a pandas data frame with columns ff, root_path, subpath, filename
    df = pd.DataFrame(data=ffl,columns=['ff'])
    listSubpath = []
    listFn = []
    listFnRoot = []
    listFnExt = []
    for ff in ffl:
        sp,fn = os.path.split(ff.replace(root_path,''))
        # clean sp (subpath) from remaining path literals ('/','\\'), especially necessary in Windows
        sp = sp.replace('/','')
        sp = sp.replace('\\','')
        # store to list for df
        listSubpath.append(sp)
        listFn.append(fn)
        root,ext = splitext(fn)
        listFnRoot.append(root)
        listFnExt.append(ext)
    df = df.assign( subpath=listSubpath )
    df = df.assign( fn=listFn )
    df = df.assign( fn_root=listFnRoot )
    df = df.assign( fn_ext=listFnExt )
    df['pn'] = [os.path.split(x)[0] for x in df.ff]
    df['root_path'] = [root_path for x in df.ff]
    if sorted:
        df = df.sort_values(['ff'])
        df = df.reset_index(drop=True)
    return df
# regex tutorial:
# http://regextutorials.com/intro.html

def locateDirs(pattern, root_path, level=0):
    if not os.path.exists(root_path):
        raise ValueError('Directory ("%s") does not exist!'%root_path)
    pl = np.array([])
    for root, dirs, files in walklevel(root_path, level): #os.walk(root_path):
        for pathname in fnmatch.filter(dirs, pattern):
            #print( os.path.join(root, pathname))
            pl = np.append(pl,os.path.join(root, pathname))
    return pl

def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        if root.count(os.path.sep) == num_sep + level:            
            yield root, dirs, files
        # num_sep_this = root.count(os.path.sep)
        # if num_sep + level <= num_sep_this:
        #     print(dirs)
        #     del dirs[:]
        
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
    
def osnj(*args):
    return osn(osj(*args)).replace('\\','/')
    
def splitext(fn):
    if fn.endswith('.nii.gz'):
        root = fn.replace('.nii.gz','')
        ext = '.nii.gz'
        return (root, ext)
    else:
        return os.path.splitext(fn)
    
def remove_file(pattern, root_path, level=0):
    fl = locateFiles(pattern, root_path, level)
    for idx, ff in enumerate(fl):
        print(ff)
        os.remove(ff)
        
def convert_cmd_to_linux(cmd):
    return cmd.replace('D:', '/mnt/d').replace('\\','/')                       

def convert_linux_to_win(filename:str):
    return filename.replace('/mnt/d', 'D:').replace('/','\\')

def save_pickle(data:dict, filename:str):
    outfile = open(filename,'wb')
    pickle.dump(data,outfile)
    outfile.close()
    return
    
def open_pickle(filename:str):
    infile = open(filename,'rb')
    temp_data = pickle.load(infile)
    infile.close()
    return temp_data
    