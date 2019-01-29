#!/usr/bin/env python

# inst: university of bristol
# auth: jeison sosa
# mail: sosa.jeison@gmail.com / j.sosa@bristol.ac.uk

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import gdalutils as gu
import geopandas as gpd
import lfptools.utils as lfp
import hydroutils.core as hu
from glob import glob
from scipy import stats
from subprocess import call
from shapely.geometry import Point
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from lfptools.buildmodel import write_bci
from lfptools.buildmodel import write_bdy
from lfptools.buildmodel import write_evap
from lfptools.buildmodel import write_par
from lfptools.buildmodel import write_gauge_stage_all_cells


def banks(dirtif, widthtif, bedtif, runoffcsv, reccsv, date1, date2, banktif, return_per, layer, lisfloodfp):

    # Create a temp temporal work folder
    outfolder = os.path.dirname(banktif) + '/banks-temp/'
    try:
        os.makedirs(outfolder + 'lfp/')
    except FileExistsError:
        pass

    run_simulation(reccsv=reccsv,
                   dirtif=dirtif, 
                   widthtif=widthtif,
                   bedtif=bedtif,
                   runoffcsv=runoffcsv,
                   date1=date1,
                   date2=date2,
                   lisfloodfp=lisfloodfp,
                   outfolder=outfolder)

    dischf = outfolder + 'lfp/lfp.discharge'
    stagef = outfolder + 'lfp/lfp.stage'

    calc_banks(banktif=banktif,
               bedtif=bedtif,
               fname_disch=dischf,
               fname_stage=stagef,
               reccsv=reccsv,
               return_per=return_per,
               layer=layer,
               outfolder=outfolder)


def run_simulation(reccsv, dirtif, widthtif, bedtif, runoffcsv, date1, date2, lisfloodfp, outfolder):

    # Determine end of the simulation, how many days
    t = (pd.to_datetime(date2, format='%Y-%m-%d') -
         pd.to_datetime(date1, format='%Y-%m-%d')).days + 1

    # Create 1D DEM, synthetic
    demtif = outfolder + 'dem1d.tif'
    wdt = gu.get_data(widthtif)
    geo = gu.get_geo(widthtif)
    dem = np.where(wdt > 0, 10000, 0)
    gu.write_raster(dem, demtif, geo, 'Int16', 0)

    # Convert input files to ASCII
    widthasc = outfolder + 'width.asc'
    call(['gdal_translate',
          '-of', 'AAIGRID',
          widthtif, widthasc])

    demasc = outfolder + 'dem.asc'
    call(['gdal_translate',
          '-of', 'AAIGRID',
          demtif, demasc])

    bedasc = outfolder + 'bed.asc'
    call(['gdal_translate',
          '-of', 'AAIGRID',
          bedtif, bedasc])

    # Write LISFLOOD-FP files
    bcilfp = outfolder + 'lfp.bci'
    write_bci(bcilfp, runoffcsv)

    bdylfp = outfolder + 'lfp.bdy'
    write_bdy(bdylfp, runoffcsv, t)

    evaplfp = outfolder + 'lfp.evap'
    write_evap(evaplfp, t)

    gaugelfp = outfolder + 'lfp.gauge'
    stagelfp = outfolder + 'lfp.stage'
    write_gauge_stage_all_cells(reccsv, dirtif, widthtif, gaugelfp, stagelfp)

    parlfp = outfolder + 'lfp.par'
    write_par(parlfp=parlfp,
              bcilfp=bcilfp,
              bdylfp=bdylfp,
              evaplfp=evaplfp,
              gaugelfp=gaugelfp,
              stagelfp=stagelfp,
              dembnktif=demasc,
              wdttif=widthasc,
              bedtif=bedasc,
              t=t)

    # Run simulation
    call([lisfloodfp, '-v', 'lfp.par'], cwd=outfolder)


def calc_banks(banktif, bedtif, fname_disch, fname_stage, reccsv, return_per, layer, outfolder):

    # Loading stage and discharge files
    # Try statement added since some discharge and stage files are empty, exit program
    try:
        stage = lfp.read_stage(fname_stage)
        df_locs = lfp.read_stage_locs(fname_stage)
        df_locs.index = range(len(stage.columns))
        discharge = lfp.read_discharge(fname_disch)
        stage.columns = range(len(discharge.columns))
        discharge.columns = range(len(stage.columns))
    except ValueError:
        sys.exit('ERROR: Probably stage or discharge file is empty')

    # Loading Return Perid database (eg. FLOPROS)
    gdf_defenses = gpd.read_file(return_per)

    # Getting protection level from Return Period dataset at every cell
    # River points have been buffered to allow disaggrement between geolocations
    # By buffering some points get more than one value, maximum flood protection is selected
    mygeom = [Point(x, y) for x, y in zip(df_locs['x'], df_locs['y'])]
    gdf_locs = gpd.GeoDataFrame(crs={'init': 'epsg:4326'}, geometry=mygeom)
    gdf_locs_buf = gpd.GeoDataFrame(
        crs={'init': 'epsg:4326'}, geometry=gdf_locs.buffer(0.1))
    gdf_locs_ret = gpd.sjoin(gdf_locs_buf, gdf_defenses, op='intersects')
    gdf_locs_ret['index'] = gdf_locs_ret.index
    gdf_locs_ret = gdf_locs_ret.sort_values(
        layer, ascending=False).drop_duplicates('index').sort_values('index')

    # Estimating error in discharge fitting
    dis_err = []
    for i in range(discharge.shape[1]):
        try:
            dis_err.append(get_discharge_error(discharge[i]))
        except (KeyError,np.core._internal.AxisError):
            dis_err.append(0)

    # Estimating a defenses-related discharge
    dis_df = []
    for i in range(discharge.shape[1]):
        ret_pe = gdf_locs_ret['MerL_Riv'][i]
        try:
            dis_df.append(get_discharge_returnperiod(discharge[i], ret_pe))
        except (KeyError,np.core._internal.AxisError):
            dis_df.append(np.nan)

    # Estimating error in stage fitting
    stg_err = []
    for i in range(discharge.shape[1]):
        try:
            stg_err.append(get_stage_error(discharge[i], stage[i]))
        except (RuntimeError, TypeError):
            stg_err.append(0)

    # Estimating a defenses-related stage
    stg_df = []
    for i in range(discharge.shape[1]):
        try:
            stg_df.append(get_stage_discharge(
                discharge[i], stage[i], dis_df[i]))
        except (RuntimeError, TypeError):
            stg_df.append(np.nan)

    # Preparing a summary with variables retrived
    df_locs['dis_df'] = dis_df
    df_locs['stg_df'] = stg_df
    df_locs['dis_err'] = dis_err
    df_locs['stg_err'] = stg_err

    # Read REC file
    rec = pd.read_csv(reccsv)

    # Convert dataframe to geodataframe, join with rec
    gdf_sum = gpd.GeoDataFrame(df_locs, crs={'init': 'epsg:4326'}, geometry=[
        Point(x, y) for x, y in zip(df_locs['x'], df_locs['y'])])
    gdf_rec = gpd.GeoDataFrame(rec, crs={'init': 'epsg:4326'}, geometry=[
        Point(x, y) for x, y in zip(rec['lon'], rec['lat'])])
    gdf_rec_buf = gpd.GeoDataFrame(
        rec, crs={'init': 'epsg:4326'}, geometry=gdf_rec.buffer(0.001))
    gdf_sum_rec = gpd.sjoin(gdf_sum, gdf_rec_buf,
                            how='inner', op='intersects')
    gdf_sum_rec.sort_values('index_right', inplace=True)

    # Save errors in a GeoJSON file
    try:
        gdf_sum_rec.to_file(outfolder + 'bnk_err.geojson', driver='GeoJSON')
    except:
        os.remove(outfolder + 'bnk_err.geojson')
        gdf_sum_rec.to_file(outfolder + 'bnk_err.geojson', driver='GeoJSON')

    # Score should greater than 0.85 for both Discharge and Stage to be accepted, otherwise NaN
    gdf_err = gdf_sum_rec['stg_df'].where(
        (gdf_sum_rec['dis_err'] > 0.85) & (gdf_sum_rec['stg_err'] > 0.85))

    # Fill with NaN stg_df not filling that condition
    gdf_sum_rec['stg_df'] = gdf_err

    # NaNs are filled repating last/first number per link
    gdf_sum_rec_fillna = gdf_sum_rec.groupby('link').fillna(
        method='bfill').fillna(method='ffill')
    gdf_sum_rec_fillna['link'] = gdf_sum_rec['link']

    # Read data and geo for bedtif
    bed = gu.get_data(bedtif)
    geo = gu.get_geo(bedtif)

    # Convert dataframes to arrays
    df_locs_stgdf = gdf_sum_rec_fillna[['x', 'y', 'stg_df']]
    df_locs_stgdf.columns = ['x', 'y', 'z']
    arr_stgdf = gu.pandas_to_array(df_locs_stgdf, geo, 0)

    # Sum bankfull stage and defenses-related stage to bed
    arr_bnkdf = (bed + arr_stgdf)

    # Write burned banks in ASC and TIF files
    gu.write_raster(arr_bnkdf, banktif, geo, 'Float64', 0)


def get_discharge_error(tserie):
    serie = tserie.to_frame()
    amax = hu.find_events_amax(serie)
    serie.columns = ['series']
    amax.columns = ['amax']
    amax_vals = np.sort(amax.values.squeeze())

    # Select distribution
    cdf = "genextreme"

    # Fit our data set against every probability distribution
    parameters = eval("stats."+cdf+".fit(amax_vals)")

    # Applying the Kolmogorov-Smirnof one sided test
    D, p = stats.kstest(amax_vals, cdf, args=parameters)

    return p


def get_discharge_returnperiod(tserie, x):
    serie = tserie.to_frame()
    amax = hu.find_events_amax(serie)
    serie.columns = ['series']
    amax.columns = ['amax']
    amax_vals = np.sort(amax.values.squeeze())
    return hu.get_dis_rp(amax_vals, hu.stats.genextreme, x, 'mle')


def plot_rel_retperiod_discharge(tserie):
    serie = tserie.to_frame()
    amax = hu.find_events_amax(serie)
    serie.columns = ['series']
    amax.columns = ['amax']
    ax = serie.plot()
    amax.plot(ax=ax, style='o', c='Red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Discharge (m^3/s)')
    amax_vals = np.sort(amax.values.squeeze())
    hu.plot_return(amax_vals, hu.stats.genextreme, 'mle')


def _fit_func_discharge_stage(func_discharge_stage, df_dis, df_sta):
    popt, pcov = curve_fit(func_discharge_stage, df_dis, df_sta)
    return popt


def _func_discharge_stage(x, a, b, c):
    return a * (x + b) ** c


def get_stage_error(discharge, stage):
    popt = _fit_func_discharge_stage(_func_discharge_stage, discharge, stage)
    mystage = _func_discharge_stage(discharge, *popt)

    # To catch NaN, infinity or a value too large for dtype('float64')
    # Better to set 0 the r2_score
    try:
        return r2_score(stage, mystage)
    except ValueError:
        return 0


def get_stage_discharge(discharge, stage, x):
    df = pd.DataFrame([discharge, stage, ]).T
    df.columns = ['discharge', 'stage']
    popt = _fit_func_discharge_stage(
        _func_discharge_stage, df['discharge'], df['stage'])
    return _func_discharge_stage(x, *popt)


def plot_rel_discharge_stage(tdischarge, tstage):
    df = pd.DataFrame([tdischarge, tstage, ]).T
    df.columns = ['discharge', 'stage']
    df['func_x'] = np.linspace(
        df.discharge.min(), df.discharge.max(), len(df['stage']))
    df['func'] = get_stage_discharge(tdischarge, tstage, df['func_x'])
    ax = df.plot(kind='scatter', x='discharge', y='stage')
    df.plot(kind='line', x='func_x', y='func', style='-r', ax=ax)
    ax.set_xlabel('discharge')
    ax.set_ylabel('stage')
