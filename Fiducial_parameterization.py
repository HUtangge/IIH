"""
Extract The fiducial parameters from Fiducial list
"""
#%% Prepare the file names 
pnMRIptn = r'C:\users\tangge\SANS\Raw_data\%s'
ptnMRI = 'Denoised_*.nii'
pnout = r'C:\users\tangge\SANS\Raw_data\Summary'
ffout = r'allFidDistances_%s'
file_extension = r'.csv'

#%% Some help functions for this part
def readSlicerAnnotationFiducials(ff):
    fids = pd.read_csv(ff,
                       comment='#',
                       header=None,
                       names=['id','x','y','z','ow','ox','oy','oz','vis','sel','lock','label','desc','associatedNodeID'],
                       engine='python')
    return fids

def df_dist(df,pt1,pt2):
    p1 = df.loc[pt1,['x','y','z']].values.astype(float)
    p2 = df.loc[pt2,['x','y','z']].values.astype(float)
    return np.linalg.norm(p1-p2)

def fid_measures(df, withEyeOrbDist=True):
    #df = df.set_index('label')
    d = dict()
    for side in ['L','R']: 
        d['d1_%s'%side] = df_dist(df,'individualized_center_%s_lens'%side,'individualized_center_%s_eyeball'%side)
        d['d2_%s'%side] = df_dist(df,'individualized_center_%s_eyeball'%side,'nerve_tip_%s'%side)
        d['d3_%s'%side] = df_dist(df,'individualized_center_%s_lens'%side,'eyeball_back_%s'%side)
        d['w1_%s'%side] = df_dist(df,'eyeball_midline_%s_lat'%side,'eyeball_midline_%s_med'%side)
        d['w2_%s'%side] = df_dist(df,'nerve_baseline_muscle_%s_lat'%side,'nerve_baseline_muscle_%s_med'%side)
        d['w3_%s'%side] = df_dist(df,'nerve_baseline_bone_%s_lat'%side,'nerve_baseline_bone_%s_med'%side)
        d['n1_%s'%side] = df_dist(df,'nerve_width_%s_lat'%side,'nerve_width_%s_med'%side)
        d['h1_%s'%side] = df_dist(df,'optcanal_height_%s_inf'%side,'optcanal_height_%s_sup'%side)
        d['w4_%s'%side] = df_dist(df,'optcanal_width_%s_lat'%side,'optcanal_width_%s_med'%side)
        
        # estimate lsq-plane of orbital rim
        if withEyeOrbDist:
            ptsOrb = df.loc[['orbital_rim_%s_lat'%side,
                              'orbital_rim_%s_med'%side,
                              'orbital_rim_%s_sup'%side,
                              'orbital_rim_%s_inf'%side],['x','y','z']].values.astype(float)
            normal,offset,R = lstsqPlaneEstimation(ptsOrb)
            ptsOrbMean = np.mean(ptsOrb,axis=0)
            # Eyecenter to the plane
            ptsEyeCtr  = df.loc['individualized_center_%s_eyeball'%side,['x','y','z']].values.astype(float)
            disteye = np.dot(normal,ptsEyeCtr-ptsOrbMean)
            d['d4_%s'%side] = disteye
            # Lens center to the plane
            ptsLensCtr  = df.loc['individualized_center_%s_lens'%side,['x','y','z']].values.astype(float)
            distlens = np.dot(normal,ptsLensCtr-ptsOrbMean)
            d['d5_%s'%side] = distlens
    return d

def vol_nerve():
    pass

def vol_sheath():
    pass

def lstsqPlaneEstimation(pts):
    # pts need to be of dimension (Nx3)
    # pts need to be centered before estimation of normal!!
    ptsCentered = pts-np.mean(pts,axis=0)
    # do fit via SVD
    u, s, vh = np.linalg.svd(ptsCentered[:,0:3].T, full_matrices=True)
    #print(u)
    #print(s)
    #print(vh)
    normal = u[:,-1] 
    # the normal should point towards the world z-direction
    if normal[-1]<0:
        normal *= -1.0
    #R = np.zeros((3,3))
    #for i in range(3):
    #    R[:,i] = u[:,i] / np.linalg.norm(u[:,i])
    R = u.copy() # u is already orthonormal
    #normal = u[:,-1] / np.linalg.norm(u[:,-1])
    offset = np.mean(pts[:,0:3],axis=0)
    return (normal,offset,R)

def change_columnname_in_dictionary(dataframe, nameSeg: str):
    column_with_segment = {}
    for idx, name in enumerate(df.columns):
        column_with_segment[name] = f'{nameSeg}_{name}'
    updated_dataframe = df.rename(columns = column_with_segment, inplace = False)
    return updated_dataframe


#%% Prepare the file names 
pnMRIptn = r'C:\users\tangge\SANS\Raw_data\%s'
ptnMRI = 'Denoised_*.nii'
pnout = r'C:\users\tangge\SANS\Raw_data\Summary'
ffout = r'allFidDistances_%s'
file_extension = r'.csv'

df_fls = []
flagDemo = False
save = True
# For each cohort (Cosmonauts and Astronauts)
list_tags_cohort = []
for iterc, tag_cohort in enumerate([['Cosmo02mm', 'Cosmonaut_BIDS']]): 
    print(iterc, tag_cohort)    
    if flagDemo:
        if iterc>0:
            break
    pnMRI  = pnMRIptn%tag_cohort[1]
    if iterc == 0:
        ffout = ffout%tag_cohort[0]
    else:
        ffout = ffout + '%s' 
        ffout = ffout%tag_cohort[0]            
    fl = fs.locateFilesDf(ptnMRI, pnMRI, level=3)
    tempTagsCohort = [tag_cohort[0] for x in fl.fn]
    fl['tag_cohort'] = tempTagsCohort
    df_fls.append(fl)

fl = pd.concat(df_fls)
fl = fl.reset_index(drop=True)
fn_splits = [x.split('_') for x in fl.fn]
tagsID  = [x[0] for x in fn_splits]
tagsSes = [x[1][4:] for x in fn_splits]
fl['id'] = tagsID
fl['ses'] = tagsSes

#%% Extract the distances
fnFIDSptn = 'fids*.fcsv' # Astronauts / Cosmonauts
dfsFIDS = []
list_fid_measures = []
for idx in range(fl.shape[0]):
    #pnFIDS = r'D:\Dropbox\Projects\AstronautT1\data\results_fidsFiles_AstronautT1'
    pn, fn = os.path.split(fl.ff[idx].replace('/','\\'))
    pn = r'{}'.format(pn)    
    ffFIDS = fs.locateFiles(fnFIDSptn, os.path.join(pn,'SANS'), level=0)[0].replace('/','\\')
    if not os.path.exists(ffFIDS):
        # dummy dataframe
        dfFIDS = pd.DataFrame()
        dfsFIDS.append(dfFIDS)
        # dummy distances
        dists = {'d1_L': np.nan,
                 'd1_R': np.nan,
                 'd2_L': np.nan,
                 'd2_R': np.nan,
                 'd3_L': np.nan,
                 'd3_R': np.nan,
                 'd4_L': np.nan,
                 'd4_R': np.nan,
                 'd5_L': np.nan,
                 'd5_R': np.nan,
                 'd6_L': np.nan,
                 'd6_R': np.nan,
                 'fn_root': fl.fn_root[idx],
                 'n1_L': np.nan,
                 'n1_R': np.nan,
                 'w1_L': np.nan,
                 'w1_R': np.nan,
                 'w2_L': np.nan,
                 'w2_R': np.nan,
                 'w3_L': np.nan,
                 'w3_R': np.nan,
                 'w4_L': np.nan,
                 'w4_R': np.nan,
                 'h1_L': np.nan,
                 'h1_R': np.nan,}
        list_fid_measures.append(dists)
    else:    
        dfFIDS = readSlicerAnnotationFiducials(ffFIDS)
        dfFIDS = dfFIDS.set_index('label')
        dists = fid_measures(dfFIDS)
        dists['fn_root'] = fl.fn_root[idx][9:]
        dfsFIDS.append(dfFIDS)
        list_fid_measures.append(dists)
    print('Read fids file %d of %d'%(idx+1,fl.shape[0]))

fl['dfFIDS'] = dfsFIDS
fl['fid_measures'] = list_fid_measures
#%
dfDists = pd.DataFrame(list_fid_measures)
cols = ['fn_root', 'd1_L', 'd1_R', 'd2_L', 'd2_R', 'd3_L', 'd3_R', 'd4_L', 'd4_R', 
        'd5_L', 'd5_R', 'n1_L', 'n1_R', 
        'w1_L', 'w1_R', 'w2_L', 'w2_R', 'w3_L', 'w3_R',
        'w4_L', 'w4_R', 'h1_L', 'h1_R']

dfDists = dfDists[cols]

#%% Save the files
if save:
    ffout = (fs.osnj(pnout, ffout) + file_extension)
    print(f'Save the file to {ffout}')
    if file_extension == '.xlsx':
        dfDists.to_excel(ffout, index = False)
    elif file_extension == '.csv':
        dfDists.to_csv(ffout, index = False)

