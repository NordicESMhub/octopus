###### This python file collects all the functions needed to solve the eScience 2021 project of Group 3:
#      "OCTOPUS - ExplOring aerosol-Cloud inTeractiOns in CMIP6 models Using joint-hiStograms"

###### Note that this file merges what was contained previously in different files and gathered in the package 'eclimate'.
#      The files were:
#      - analysis.py: core functions to work and plot during the 'climate analysis', such as evaluating the climatological mean
#      - regrid.py: functions to regrid
#      - misc.py: all support functions for the core analysis

###### This file is organised with first the specific functions, created as support of the report, and then the functions of the package 'eclimate'.

# Import packages
import numpy as np
import xarray as xr; xr.set_options(display_style='html')
import s3fs
import intake
import cftime
from datetime import datetime
import nc_time_axis
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cmaps
import pandas as pd
from pandas.plotting import table
import xesmf as xe
import xskillscore as xs
import os
import seaborn as sns; sns.set()



global str

################ Part 1 : EVALUATE THE BEST MODEL ################################################

def read_process_aod_data(col, years, verbose = True):

    ############## MODIS: connect to bucket and select AOD dataset
    fs = s3fs.S3FileSystem(anon=True, client_kwargs={'endpoint_url': 'https://forces2021.uiogeo-apps.sigma2.no/'})
    # specify file path on remote
    fobj = fs.open("s3://data/MODIS/MOD08_M3_SUB_20000201-20210901.nc")
    # load dataset
    dset = xr.open_dataset(fobj)
    # select AOD variable
    aod_obs = dset['AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean'].sel(time = dset.time.dt.year.isin(years)).squeeze()
    aod_obs = aod_obs.rename({'longitude': 'lon','latitude': 'lat'})
    if verbose:
        print("Imported and processed MODIS data...")

    ############## CMIP6: open online catalog
    if not col:
        cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
        col = intake.open_esm_datastore(cat_url)
    # Select just series of needed data
    cat = col.search(experiment_id='historical', variable_id='od550aer', member_id="r1i1p1f1") #col.df.member_id[0])
    # Create dictionary from the list of datasets we found
    dset_dict = cat.to_dataset_dict(progressbar = verbose, zarr_kwargs={'use_cftime':True})
    if verbose:
        print("Imported CMIP6 data...")

    # Create the equivalent dictionary for aod, in shared time range
    aod_dict = {}
    models_prop = {} # change aod_dict.keys() to source_id, but record the whole name in here
    for model_p in list(dset_dict.keys()):
        dset = dset_dict[model_p]
        aod = dset['od550aer'].sel(time = dset.time.dt.year.isin(years)).squeeze() #<class 'xarray.core.dataarray.DataArray'>
        if len(aod.time.values) != 0: # clean the datasets with no 2000-2014 time series
            model_name = model_name_of(model_p)
            models_prop[model_name] = model_p
            aod_dict[model_name] = aod
    if verbose:
        print("Processed CMIP6 data.")
        
    print("Models with AOD in the right time range:", list(aod_dict.keys()))
    print("A DataArray for observed AOD and a dictionary of DataArray for modelled AOD are created.\n")
            
    return aod_obs, aod_dict
    
def overview_aod_data(aod_obs, aod_model_dict, model_name, years, save_fig = True):

    fig=plt.figure(figsize=(15, 15))
    plt.rcParams.update({'font.size': 13})

    ax = plt.subplot(211, projection=ccrs.PlateCarree())
    
    obs_clim = annual_climatology(aod_obs)
    fig, ax = plot_climatology_ax(obs_clim, fig, ax, title = 'MODIS: climatological mean ('+str(years[0])+'-'+str(years[-1])+') of AOD', clabel="Aerosol Optical Depth at 550nm", vmin=0., vmax= 1., clabelsize = 13)

    ax = plt.subplot(212, projection=ccrs.PlateCarree())
    model_clim = annual_climatology(aod_model_dict[model_name])
    fig, ax = plot_climatology_ax(model_clim, fig, ax, title = model_name+': climatological mean ('+str(years[0])+'-'+str(years[-1])+') of AOD', clabel="Aerosol Optical Depth at 550nm", vmin=0., vmax= 1., clabelsize = 13)

    plt.suptitle('Overview of the AOD data', y=.93, weight = 'semibold', size = 17)
    if save_fig:
        check_dirs(["output/Figures"])
        plt.savefig("output/Figures/overview_AOD_data.png", dpi=100)
    plt.show()
    
    
def models_evaluation(aod_obs, aod_dict, common_models, verbose = True, save_fig = True):
    df = pd.DataFrame(columns=['MAE', 'RMSE', 'R2', r'$\Delta_{max, pos}$',r'$\Delta_{max, neg}$'], index = list(aod_dict.keys()))

    # Check on folders for storing figures
    dirp = 'output/Figures/Performance_each_model'
    path =dirp+"/"
    dirs = ('output/Figures', dirp, path+"AOD_Maps", path+"AOD_Bias")
    check_dirs(dirs)
        
    ############## 1) Map comparison
    print("Evaluating models' perfomance...")
    for m, model_name in enumerate(list(aod_dict.keys())):
        if verbose:
            print(m+1,"/",len(list(aod_dict.keys())), ":",model_name)
        aod_cmip = aod_dict[model_name]

        ####### Regrid to coarser grid
        aod_cmip_regrid, aod_obs_regrid , grid = regrid_upscale(aod_cmip, aod_obs)

        ####### Evaluate climatology
        clim_cmip = annual_climatology(aod_cmip_regrid)
        clim_obs = annual_climatology(aod_obs_regrid)
        diff = clim_cmip - clim_obs
        
        ####### Error analysis
        df.loc[model_name]['MAE'] = mean_absolute_error(clim_cmip.values, clim_obs.values)
        df.loc[model_name]['RMSE'] = sqrt(mean_squared_error(clim_cmip.values, clim_obs.values))
        df.loc[model_name]['R2'] = np.float64(xs.pearson_r(clim_cmip, clim_obs, dim=["lat", "lon"]).values)**2
        df.loc[model_name][r'$\Delta_{max, neg}$'] = np.min(diff.values)
        df.loc[model_name][r'$\Delta_{max, pos}$'] = np.max(diff.values)

        ####### Plot climatology
            
        aod_name = 'Aerosol Optical Depth at 550 nm'

        vmin = 0.#min(np.min(clim_cmip.values), np.min(clim_obs.values))
        vmax = 1. #max(np.max(clim_cmip.values), np.max(clim_obs.values))

        title=model_name+": climatological mean (2000-2014)"
        filename = path+'AOD_Maps/clim_mean_'+model_name+'.png'
        plot_climatology(clim_cmip, title = title, clabel=aod_name, cmap = "YlOrBr", vmin=vmin, vmax=vmax, filename = filename if save_fig else None)
        title = "MODIS: climatological mean (2000-2014)"
        filename=path+'AOD_Maps/clim_mean_MODIS_'+model_name+'.png'
        plot_climatology(clim_obs, title=title, clabel=aod_name, cmap = "YlOrBr", vmin=vmin, vmax=vmax, filename = filename if save_fig else None)

        annot = r"MAE: "+str(np.round(df.loc[model_name]['MAE'],2))+"\nRMSE :"+str(np.round(df.loc[model_name]['RMSE'],2))+"\nR2: "+str(np.round(df.loc[model_name]['R2'],2))+"\n$\Delta_{max, pos}$: "+str(np.round(df.loc[model_name][r'$\Delta_{max, pos}$'],2))+"\n$\Delta_{max,neg}$: "+str(np.round(df.loc[model_name][r'$\Delta_{max, neg}$'],2))
        title = "Bias ["+model_name+"]-MODIS in climatological mean (2000-2014)"
        filename = path+'AOD_Bias/clim_bias_'+model_name+'.png'
        clabel = r"$\Delta$ AOD"
        plot_climatology(diff, title=title, clabel=clabel, cmap = "RdBu_r", filename = filename if save_fig else None, vmin=-1., vmax=1., annotate = annot)

    print("Figures are in the directory 'output/Figures/Performance_each_model'")

    ############## 2) Error analysis
    print("Error analysis...")
    ### Evaluate absolute difference in between positive and negative ones and the total effect of errors
    dforig = df
    df['$\Delta_{max, abs}$'] = abs(df[r'$\Delta_{max, neg}$'].values)
    df['$\Delta_{max, abs}$'] = df[[r'$\Delta_{max, pos}$', r'$\Delta_{max, abs}$']].max(axis = 1)
    df['Total'] = tot_error_df(df[['MAE','RMSE','R2',r'$\Delta_{max, abs}$']])
    #display(df)
    
    ### Create table for the ranking scale
    best = {}
    best['MAE'] = list(df.sort_values('MAE').index.values)
    best['RMSE'] = list(df.sort_values('RMSE').index.values)
    best['R2'] =list(df.sort_values('R2', ascending=False).index.values)
    best[r'$\Delta_{max, abs}$'] = list(df.sort_values(r'$\Delta_{max, abs}$').index.values)
    best['Total'] = list(df.sort_values('Total').index.values)
    #display(pd.DataFrame.from_dict(best))
    
    
    ### Filter ranking table with just the models that have the aod and cdnc
    bestnp = list(best.values())
    errors = list(best.keys())

    best_filt = {}
    for e in range(len(bestnp)):
        all_elem = bestnp[e] #array for each 'MAE', RMSE' etc
        best_filt[errors[e]] = selection_array(all_elem,common_models)
    #display(pd.DataFrame.from_dict(best_filt))
    
    if save_fig:
        check_dirs([path+"Errors"])
        save_df_as_fig(df.astype(float).round(decimals = 3), filename=path+"Errors/Errors_table_tot.png", figsize=(11,7), colwidth=0.07, displayfig=False)
        save_df_as_fig(pd.DataFrame.from_dict(best), filename=path+"Errors/Models_ranking.png", figsize=(15,7), colwidth=0.13, displayfig=False)
        save_df_as_fig(pd.DataFrame.from_dict(best_filt), filename=path+"Errors/Models_ranking_filtered.png", figsize=(10,3), colwidth=0.15, displayfig=False)
    
    return df, best, best_filt
    
def plot_errorbar(error_df, common_models, save_fig = True):
    df_sel = error_df[['MAE', 'RMSE', 'R2','$\Delta_{max, abs}$', 'Total']]
    df_sel = df_sel.astype(float).round(decimals = 3)

    # Select models with 4 variables
    index = []
    for i in common_models:
        index.append(int(np.where(df_sel.index.values == i)[0]))

    #for sel in range(2): #loop for plotting 'selection' highlighting or not
    plt.figure()
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['figure.facecolor'] = 'white'

    axes = df_sel.plot(kind ="barh",subplots =True,figsize=(15,10),title = "Statistical analysis of the comparison CMIP6-MODIS",layout=(1,5),sharey =True,sharex=False,legend=False)
    for i, ax in enumerate(axes.ravel()):
        ax.bar_label(ax.containers[0])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for n in index:
                ax.axhline(y=n, xmin = 0, xmax=1.2,linewidth=20, alpha = 0.5, color="gold")

    #plt.tight_layout()
    if save_fig:
        check_dirs(["output/Figures/Performance_each_model/Errors"])
        filename = "output/Figures/Performance_each_model/Errors/Errors_tot"
        filename += "_sel"
        plt.savefig(filename+".png", transparent=True)
    plt.show()
    
def plot_best_worst_maps(bests, worsts, title):
    maps = bests
    maps.extend(worsts)

    path = "output/Figures/Performance_each_model/AOD_Bias/clim_bias_"

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    for i, ax in enumerate(axes.ravel()):
        
        if maps[i]:
            ax.imshow(mpimg.imread(path+maps[i]+'.png'))

    [ax.set_axis_off() for ax in axes.ravel()]

    plt.tight_layout()
    plt.suptitle(title, y=.97, size=20)
    plt.show()
    plt.close(fig)

################ Part 2 : SELECTION OF MODELS WITH ALL THE VARIABLES  ######################################

def selection_of_models(col, years, var_search = ["lwp", "cdnc", "od550aer", "clt", "clivi"], verbose = True):

    #Variables:
    #- 'lwp': liquid water path
    #- 'cdnc': cloud droplet number concentration
    #- "od550aer": aerosol optical depth
    #- "clt": cloud fraction
    #- "clivi":cloud ice content
    
    # Open CMIP6 online catalog
    if not col:
        cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
        col = intake.open_esm_datastore(cat_url)

    # Create dictionaries to work on for the selection of variables and time range
    result_tot =  {}
    result =  {}
    models = {}
    models_all = []

    for ivar in var_search:
        
        # filter the raw file and save it to dictionary
        iresult = col.search(activity_id='CMIP', variable_id=ivar, experiment_id='historical', member_id = "r1i1p1f1")
        result_tot[ivar] = iresult.to_dataset_dict(progressbar = verbose, zarr_kwargs={'use_cftime':True})
        if verbose:
            print("Models with {} = {}".format(ivar, len(iresult)))
        
        # create a dict filtered in time range (2000-2014)
        dic = {}
        res_tot = result_tot[ivar]
        for model_long_name in list(res_tot.keys()):
            ires_tot = res_tot[model_long_name]
            iresult = ires_tot.sel(time = ires_tot.time.dt.year.isin(years)).squeeze()
            if len(iresult.time.values) != 0: # clean the datasets with no 2000-2014 time series
                dic[model_long_name] = iresult
        result[ivar] = dic
        
        # record model names in a dict
        models[ivar] = model_name_of_dict(result[ivar])
        models_all.extend(models[ivar])
        
    # This was used to check to code, but very useful to see which model is daily VS monthly
    """t = 0
    res_tot = result_tot['clivi']
    for model_long_name in list(res_tot.keys()):
        t+= 1
        print(model_long_name)
        ires_tot = res_tot[model_long_name]
        iresult = ires_tot.sel(time = ires_tot.time.dt.year.isin(years)).squeeze()
        print(t,len(iresult.time.values))"""

    # Print the matching of variables-models
    df = pd.DataFrame(False, index = np.unique(models_all), columns = var_search)

    for ivar in var_search:
        for imod in df.index:
            if imod in models[ivar]:
                df.loc[imod, ivar] = True
    #display(df)

    # Models in common with AOD and CDNC, and models with all the variables in common
    models_with_aod_cdnc = [i for i in list(df[['cdnc','od550aer']].index.where(df.sum(axis = 1) == 5)) if type(i) == str]
    models_with_all_vars = [i for i in list(df.index.where(df.sum(axis = 1) == 5)) if type(i) == str]

    print("Models with AOD and CDNC", models_with_aod_cdnc)
    print("Models with all variables", models_with_all_vars)
    print()
    return models_with_aod_cdnc, models_with_all_vars
    
################ Part 3 : JOINT HISTOGRAMS ################################################

def read_process_allvar_data(col, years, model_names, variables = ["lwp","cdnc", "od550aer", "clt", "clivi"], verbose = True):

    if not col:
        col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")

    dict_tot = {}

    for m, model_name in enumerate(model_names):
        dict_tot[model_name] = {}
        
        for ivar in variables:
            # filter the raw file and save it to dictionary
            iresult = col.search(activity_id='CMIP', source_id = model_name, variable_id=ivar, experiment_id='historical', member_id = "r1i1p1f1")
            dset = iresult.to_dataset_dict(progressbar = verbose, zarr_kwargs={'use_cftime':True})
            dset = dset[list(dset.keys())[0]]
            # create a dict filtered in time range (2000-2014)
            dset = dset.sel(time = dset.time.dt.year.isin(years)).squeeze()
            #if ivar == 'od550aer':
            #    ivar = 'aod'
            if len(dset.time.values) != 0: # clean the datasets with no 2000-2014 time series
                dict_tot[model_name][ivar] = dset[ivar]
                if ivar == 'cdnc': #select the maximum value in the level scale (i.e. proxy of the cndn at TOA)
                    dict_tot[model_name][ivar] = dset[ivar].max('lev') *1e-6 # convert to [cm-3]
    
    print("Dictionary of DataArrays for models", model_names, "and variables", variables, "created.")
        
    return dict_tot
    
def plot_study_areas(model_dict_tot, model_name, zones_lat, zones_lon, years, save_fig = True):

    fig=plt.figure(figsize=(15,15))
    plt.rcParams.update({'font.size': 13})

    ax = plt.subplot(211, projection=ccrs.PlateCarree())
    
    zones_letters = ['A','B','C','D']

    # Evaluate climatology of AOD
    aod_clim = annual_climatology(model_dict_tot[model_name]['od550aer'])
    # Draw plot of AOD
    fig, ax = plot_climatology_ax(aod_clim, fig, ax, title = 'Climatological mean ('+str(years[0])+'-'+str(years[-1])+') of AOD', clabel="Aerosol Optical Depth at 550nm", clabelsize = 13)
    # Add rectagles to the ax
    for i in range(4):
        xy = (zones_lon[i][0]-180.,zones_lat[i][0])
        width = zones_lon[i][1] - zones_lon[i][0]
        height = zones_lat[i][1] - zones_lat[i][0]
        ax.add_patch(Rectangle(xy,width,height,fc ='none', ec = 'black',lw =3))
        ax.text(zones_lon[i][0]-180+width/2.,zones_lat[i][0]+height/2., zones_letters[i],
             horizontalalignment='center',verticalalignment='center',size=20)#transform=ccrs.Geodetic())

    ax = plt.subplot(212, projection=ccrs.PlateCarree())

    # Evaluate climatology of CDNC
    cdnc_clim = annual_climatology(model_dict_tot[model_name]['cdnc'])
    # Draw plot of CDNC
    fig, ax = plot_climatology_ax(cdnc_clim, fig, ax, cmap="YlGnBu", title = 'Climatological mean ('+str(years[0])+'-'+str(years[-1])+') of CDNC', clabel=r"Cloud Droplet Number Concentration [$cm^{-3}$]", clabelsize = 13)
    
    # Add rectagles to the ax
    for i in range(4):
        xy = (zones_lon[i][0]-180.,zones_lat[i][0])
        width = zones_lon[i][1] - zones_lon[i][0]
        height = zones_lat[i][1] - zones_lat[i][0]
        ax.add_patch( Rectangle(xy,width,height,fc ='none', ec = 'black',lw =3))
        ax.text(zones_lon[i][0]-180+width/2.,zones_lat[i][0]+height/2., zones_letters[i],
             horizontalalignment='center',verticalalignment='center',size=20) #transform=ccrs.PlateCarree())#ccrs.Geodetic())
    
    plt.suptitle('Overview of the study areas \n(data of '+str(model_name)+')', y=.95)
    if save_fig:
        check_dirs(["output/Figures"])
        plt.savefig("output/Figures/study_areas.png", dpi=100)
    plt.show()
    
    
def prepare_global_df(dict_tot, model_names, verbose = True):
    """Ravel all the data into a dataframe as input of the joint histograms"""
    global_df_tot = {}
    for m, model_name in enumerate(model_names):
        if verbose:
            print(model_name, '...')
        cdnc = dict_tot[model_name]['cdnc'].to_series()
        aod = dict_tot[model_name]['od550aer'].to_series()
        frame = { 'aod': aod, 'cdnc': cdnc }
        global_df_tot[model_name] = pd.DataFrame(frame)
    return global_df_tot
        
        
def plot_aod_cdnc(df, title ='', vmax =None, yrange=None):
    im = sns.jointplot(data=df, x="aod", y="cdnc", kind="hist", height = 6, vmax =vmax,cmap="Spectral_r", cbar = True, stat = 'probability', marginal_kws=dict(stat="probability", fill=False), marginal_ticks=True, cbar_kws={'label': 'Probability', 'location':'top'})
    im.ax_marg_x.set_xlim(0.01, 1)
    if yrange:
        im.ax_marg_y.set_ylim(yrange[0],yrange[1])
    im.ax_joint.set_xscale('log')
    im.ax_joint.set_yscale('log')
    im.set_axis_labels(r'AOD ($log_{10}$)', r'CDNC [$cm^{-3}$] ($log_{10}$)')
    im.ax_joint.set_yticks((10,30,100))
    im.ax_joint.set_yticklabels(['10','30','100'])
    im.fig.suptitle(title, fontsize=16)
    im.fig.tight_layout()

    return im#.fig
    
    
def joint_histograms(model, global_df_tot, zones_lat, zones_lon, zones_titles, zones_filenames, displayfig=True):
    path = "output/Figures/JointHistograms"
    check_dirs([path])
    path+= "/"
    global_df = global_df_tot[model]
    no_polar_df = select_multindex_lat_lon(global_df,lat=(-60,60))
    global_fig = plot_aod_cdnc(global_df, "Global", yrange=(5,200), vmax =6.*1e-5)
    global_fig.fig.tight_layout()
    global_fig.fig.savefig(path+"Global_"+model+".png", transparent=True)
    if not displayfig:
        plt.close(global_fig.fig)
    #global_fig.fig.show()
    no_polar_fig = plot_aod_cdnc(no_polar_df, "No polar", yrange=(5,200), vmax =6.*1e-5)
    no_polar_fig.fig.tight_layout()
    no_polar_fig.fig.savefig(path+"No_polar_"+model+".png", transparent=True)
    if not displayfig:
        plt.close(no_polar_fig.fig)
    #no_polar_fig.fig.show()

    for i in range(4):
        zone_df = select_multindex_lat_lon(global_df,lat=zones_lat[i], lon=zones_lon[i])
        c = plot_aod_cdnc(zone_df, zones_titles[i], yrange=(5,100), vmax=0.0025)
        c.fig.tight_layout()
        c.fig.savefig(path+zones_filenames[i]+"_"+model+".png", transparent=True, dpi = 150)
        if not displayfig:
            plt.close(c.fig)
        #c.fig.show()
        
def merge_jointhist(model, zones_filenames):

    path = "output/Figures/JointHistograms/"

    title_list = ['Global', 'No_polar']
    title_list.extend(zones_filenames)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 24))

    for i, ax in enumerate(axes.ravel()):
        ax.imshow(mpimg.imread(path+title_list[i]+'_'+model+'.png'))

    [ax.set_axis_off() for ax in axes.ravel()]

    plt.tight_layout()
    plt.suptitle("Joint histograms AOD-CDNC of '"+model+"' model", y=1., size=20)
    plt.show()
    plt.close(fig)



############################### analysis.py:

def climatology_mean(ds, time_res="month"):  # test for season too
    """Evaluate the 'time_res'-ly (i.e. monthly) mean, weighted on the days"""
    # Make a DataArray with the number of days in each month, size = len(time)
    attrs = ds.attrs
    month_length = ds.time.dt.days_in_month

    # Calculate the weights by grouping by 'time.season'
    weights = month_length.groupby("time."+time_res) / month_length.groupby("time."+time_res).sum()

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby("time."+time_res).sum().values, np.ones(len(month_length.groupby("time."+time_res).sum().values)))

    # Calculate the weighted average
    wm = (ds * weights).groupby("time."+time_res).sum(dim="time")
    wm.attrs = attrs
    return wm

def annual_climatology(ds):
    """Evaluate the annual climatological mean, through evaluating the annual cycle first.
    Return an array"""
    attrs = ds.attrs
    ds_clim = climatology_mean(ds, "month")
    m = ds_clim.mean(dim = "month")
    m.attrs = attrs
    return m
    
def plot_climatology(clim, title=None, clabel=None, cmap = "YlOrBr", vmin=None, vmax=None, robust =False, filename=None, annotate = None, displayfig=False):
    """Plot suitable for climatological mean. It shows a map with PlateCaree projection with the colorbar indicating the value of the climatological mean."
        
        Args:
          - clim (xarray.Dataset/DataArray): with dim=['lon','lat']
          - title (str): title of the plot
          - cmap (str): colormap, palette for the colorbar
          - vmin (float): min value for the colobar
          - vmax (float): max value for the colorbar
          - filename (str): name of the saved file. If None no file will be saved. Path needs to be included. Format in vector file ('svg').
          - annotate (str): if not None it will create an annotation box on the plot.
          - displayfig (bool): if False the figure display is silenced."""
    
    clim = clim.sortby(clim['lon'])
    
    fig = plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 10})
    ax = plt.subplot(projection=ccrs.PlateCarree())
    
    fig, ax = plot_climatology_ax(clim, fig, ax, title = title, clabel=clabel, cmap = cmap, vmin=vmin, vmax=vmax, robust = robust)

    if annotate: #If "bias" plot, this box of annotations is useful to plot the errors
        ax.annotate(annotate, xy = (0.15,0.4),xycoords = 'figure fraction', bbox=dict(boxstyle="round", fc="w", alpha=0.7))
    if not displayfig:
        plt.close(fig)
    
    if filename:
        fig.savefig(filename, dpi=100)
        
def plot_climatology_ax(clim, fig, ax, title = None, clabel=None, clabelsize = 10, cmap = "YlOrBr", vmin=None, vmax=None, robust =False):
    """Core function for drawing plots of climatological means.
    It decorates the given ax and returns it in order to plot in different subplot layouts"""
    
    im = clim.plot(ax=ax, vmin = vmin, vmax = vmax, cmap = cmap, robust = robust, add_colorbar = False)
    cbar = add_colorbar(im, fig, ax)
    cbar.set_label(clabel, size = clabelsize)
    #cbar.ax.tick_params(labelsize=clabelsize)
    ax.set_title(title, size = clabelsize+2)
    ax.coastlines(resolution='110m')
    
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    ax.text(-0.07, 0.55, "Latitudine", va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(0.5, -0.1, "Longitude", va='bottom', ha='center',
            rotation='horizontal', rotation_mode='anchor',
            transform=ax.transAxes)
    return fig, ax
        
def add_colorbar(im, fig, ax):
    """Add colorbar of same size of the plot (helpful if subplot).
      Needs to be accompany by 'add_colorbar = False' in ds.plot().
    
      Args:
        - im : output of ds.plot(). """
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    return plt.colorbar(im, cax=cax)
    
def tot_error_df(df):
    """Evaluate the cumulative effect of multiple errors, rescaling each error type in the range [0,1]
    It is an ad hoc function for the error df used in this analysis.
    Returns the total sum of the rescaled errors"""
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=df.columns, index=df.index)
    df['R2'] = 1. - df['R2']
    return df.sum(axis=1)
    
def save_df_as_fig(df, filename, figsize=(15,7), colwidth=0.1, scale=(1.5,1.5), displayfig=True):
    """Save pandas dataframe as a figure. Be careful on setting the right parameters to obtain the figure you want. It's not automatical.
        
        Args:
        - df (pandas dataframe): df to save
        - filename (str): name of the saving file with the path
        - figsize (tuple): size of the figure
        - colwidth (float): fraction of the columns width
        - scale (tuple): scaling of df into the set image frame
        - displayfig (bool): if True, the table is display interactively, otherwise just saving.
    """
    
    fig, ax = plt.subplots(figsize=figsize) # set size frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
    tabla = table(ax, df, loc='center', colWidths=[colwidth]*len(df.columns))  # where df is your data frame
    tabla.auto_set_font_size(False) # Activate set fontsize manually
    tabla.set_fontsize(12)
    tabla.scale(scale[0], scale[1]) # change size table
    plt.savefig(filename, transparent=True)
    if not displayfig:
        plt.close(fig)




############################### regrid.py:

def convert360_180(ds):
    _ds = ds
    """
    Convert the longitude of the given xr:Dataset from [0-360] to [-180-180] deg
    """

    if _ds['lon'].min() >= 0:
        with xr.set_options(keep_attrs=True): 
            _ds.coords['lon'] = (_ds['lon'] + 180) % 360 - 180
        _ds = _ds.sortby(_ds.lon)
    return _ds

def regrid_upscale(ds_model, ds_obs, method='bilinear'):
    """If the model and observation grids are different, it upscales the finer grid to the coarser one using the passed 'method' (default: bilinear).
    In case longitudine is measured in [0,360], it converts it in [-180,180].
    It uses the xesmf.Regridder, that works with "lon" and "lat" dimension names, pay attention to rename them.
    In case lat_model<lat_obs and lon_model>lon_obs (viceversa) I just take the model grid.
    
    Args:
        - ds_model (xarray.Dataset/xarray.Dataarray): model dataset
        - ds_obs (xarray.Dataset/xarray.Dataarray): observation dataset
        - method (string): regridding method (default: bilinear)
        
    Returns:
        - ds_model(_regrid) (xarray.Dataset/xarray.Dataarray): model dataset, eventually regrided
        - ds_obs(_regrid) (xarray.Dataset/xarray.Dataarray): observation dataset, eventually regrided
        - grid (string): "model"/"obs" to keep record of the grid that has been regrided
        """

    # I can trhow en exception here for the .lat and .lon
    #ds_obs = convert360_180(ds_obs)
    grid_obs = xr.Dataset({
         "lat": (["lat"], np.arange(-89.5, 90.5, 1.)),
         "lon": (["lon"], np.arange(-179.5, 180.5, 1.)),})
    ds_obs = ds_obs.sortby(ds_obs.lat)
    ds_model = convert360_180(ds_model)
    step_model = abs(ds_model.lat.values[0] - ds_model.lat.values[1])
    step_obs = abs(grid_obs.lat.values[0] - grid_obs.lat.values[1])
    if step_model < step_obs:
        step_model = abs(ds_model.lon.values[0] - ds_model.lon.values[1])
        step_obs = abs(grid_obs.lon.values[0] - grid_obs.lon.values[1])
        if step_model > step_obs:
            pass
        else:
            # if upscaling:
            # input grid = original grid = coarser grid  (2.5 x 2.5) -> i.e. obs
            # output grid = edited grid = finer grid (1 x 1.5) -> i.e. model
            # xe.Regridder(grid_in, grid_out, method)
            regridder = xe.Regridder(ds_model, grid_obs, method)
            ds_model_regrid = regridder(ds_model, keep_attrs=True)
            ds_obs = ds_obs.sortby(ds_obs.lon)
            grid = "model"
            return ds_model_regrid, ds_obs, grid
        
    regridder = xe.Regridder(grid_obs, ds_model, method)
    ds_obs_regrid = regridder(ds_obs, keep_attrs=True)
    grid = "obs"
    return ds_model, ds_obs_regrid, grid
    




############################### misc.py:

def model_name_support(model_prop):
    """ Support function for model_name_of() in order to make it versatile for different input type """
    # From 'activity_id.institution_id.source_id.etc' select just 'source_id'
    model_prop = str(model_prop)
    start = model_prop.find(".")+1
    substring = model_prop[start:-1]
    start = substring.find(".")+1
    substring = substring[start:-1]
    end = substring.find(".")
    return substring[0:end]
    
def model_name_of(model_prop):
    """Select from the whole model properties name, just the one which refers to the source_id
    
        Args:
            - model_prop (string or list): activity_id.institution_id.source_id.etc
        Returns:
            - model name (string or list): source_id """

    if type(model_prop) == list:
        model_name = []
        for m in model_prop:
            model_name.append(model_name_support(m))
        return model_name
        
    else:
        return model_name_support(model_prop)
        
def model_name_of_dict(dict):
    "Same duty as model_name_of() module but with input as dictionary"
    return model_name_of(list(dict.keys()))
    
def selection_array(vect, selection):
    """Selection of the given array/list with repetition, keeping the elements listed in 'selection' in the right order.
        Returns an array or list"""
    array = np.array(vect)
    indx = []
    for i in range(len(array)):
        if array[i] in selection:
            indx.append(i)
    if type(vect) == list:
        return list(array[indx])
    else:
        return array[indx]

def select_multindex_lat_lon(big_df,lat=(-90,90),lon=(-180,360)):
    """Select range of 'lat' and 'lon' in a given MultiIndex dataframe 'df'.
    The args 'lat' and 'lon' are tuples with the range values of selection (ex: (lat1, lat2))"""
    df = big_df
    lat = sorted(lat)
    lon = sorted(lon)
    return df.loc[(df.index.get_level_values('lat') > lat[0]) & (df.index.get_level_values('lat') < lat[1]) & (df.index.get_level_values('lon') > lon[0]) & (df.index.get_level_values('lon') < lon[1])]

def check_dirs(list):
    """Check if the list of directories already exist, otherwise it creates them"""
    for i in list:
        if not os.path.exists(i):
            os.makedirs(i)

def debug_print():
    """Debug function with an italian flavour ;) """
    print("Ciao amici!")


