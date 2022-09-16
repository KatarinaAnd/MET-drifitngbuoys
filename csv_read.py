import pandas as pd
from requests.utils import to_key_val_list
import xarray as xr
import yaml
import datetime
import time
from itertools import dropwhile
import sys 
import os
import requests
import argparse
from io import StringIO
import numpy as np
import traceback
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta, date

def parse_arguments():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-s","--startday",dest="startday",
            help="Start day in the form YYYY-MM-DD", required=False)
    parser.add_argument("-e","--endday",dest="endday",
            help="End day in the form YYYY-MM-DD", required=False)
    parser.add_argument("-st","--station",dest="station",
            help="simba10/simba80/aws", required=True)
    parser.add_argument("-c", "--cfg", dest="cfgfile",
            help="Configuration file", required=True)
    parser.add_argument("-u","--upt",dest="update",
            help="To update data", required=False, action='store_true')
    args = parser.parse_args()


    if args.startday is None:
        pass
    else:
        try:
            datetime.strptime(args.startday,'%Y-%m-%d')
        except ValueError:
            raise ValueError
        
    if args.endday is None:
            pass
    else:
        try:
            datetime.strptime(args.endday,'%Y-%m-%d')
        except ValueError:
            raise ValueError
        
    if args.station is None:
        parser.print_help()
        parser.exit()
    
    if args.cfgfile is None:
        parser.print_help()
        parser.exit()
        
    return args

print(parse_arguments().update)



def parse_cfg(cfgfile):

    with open(cfgfile, 'r') as ymlfile:
        cfgstr = yaml.full_load(ymlfile)
    return cfgstr

def update_func(cfgfile):
    cfg = parse_cfg(cfgfile)
    dir_list = os.listdir(cfg['output']['destdir'])
    blob = []
    for i in dir_list:
        if i.split('_')[1].endswith('765510'):
            blob.append(datetime.strptime(i.split('_')[-1][:-3], "%Y%m"))
    dates_sorted = sorted(blob)
    return datetime.strftime(dates_sorted[-1], "%Y-%m-%d")

def simba10():
    path = parse_cfg(parse_arguments().cfgfile)['input']['path']
    simba80 = ['simba_300234065765510_gps_full.csv','simba_300234065765510_temper.csv','simba_300234065765510_status.csv','lt-ito_300234065765510.csv']
    gps = pd.read_csv(path + simba80[0])
    temp = pd.read_csv(path + simba80[1])
    status = pd.read_csv(path + simba80[2])
    light = pd.read_csv(path + simba80[3])

    gps_df = gps[['GPS_Time_dtm', 'Lat','Long','BaroTemp','Pressure','Mag_X','Mag_Y', 'Mag_Z', 'Acc_X', 'Acc_Y', 'Acc_Z']].copy()
    gps_df['GPS_Time_dtm'] = pd.to_datetime(gps_df['GPS_Time_dtm'])
    gps_df = gps_df.rename(columns={'GPS_Time_dtm': 'time1', 'Lat': 'latitude', 'Long': 'longitude',
                                    'BaroTemp': 'baro_temp', 'Pressure': 'pressure', 'Mag_X':'mag_x','Mag_Y':'mag_y',
                                    'Mag_Z': 'mag_z', 'Acc_X':'acc_x', 'Acc_Y': 'acc_y', 'Acc_Z':'acc_z'})
    gps_df = gps_df.set_index('time1')
    gps_ds = gps_df.to_xarray()

    gps_ds = gps_ds.assign_coords({'latitude': gps_ds['latitude'].values, 'longitude': gps_ds['longitude'].values})

    gps_ds['time1'].attrs['long_name'] = 'time'
    gps_ds['time1'].attrs['standard_name'] = 'time'
    gps_ds['time1'].attrs['coverage_content_type'] = 'referenceInformation'

    gps_ds['longitude'].attrs['long_name'] = 'longitude'
    gps_ds['longitude'].attrs['standard_name'] = 'longitude'
    gps_ds['longitude'].attrs['units'] = 'degrees_east'
    gps_ds['longitude'].attrs['coverage_content_type'] = 'coordinate'

    gps_ds['latitude'].attrs['long_name'] = 'latitude'
    gps_ds['latitude'].attrs['standard_name'] = 'latitude'
    gps_ds['latitude'].attrs['units'] = 'degrees_north'
    gps_ds['latitude'].attrs['coverage_content_type'] = 'coordinate'


    time = pd.to_datetime(temp["RTC_Time_DTM"])

    air_temp = temp['temp240']
    air_temp_da = xr.DataArray(air_temp, coords=[time], dims = ['time2'])

    gps_ds['air_temp'] = air_temp_da
    ds = gps_ds
    
    ds['baro_temp'] = ds['baro_temp']*16
    ds['baro_temp'].attrs['long_name'] = 'temperature of barometer'
    ds['baro_temp'].attrs['units'] = 'DegC'
    ds['baro_temp'].attrs['coverage_content_type'] = 'physicalMeasurement'
    
    #filtering values to stay below 30 degrees
    ds['baro_temp'] = ds.air_temp.where(ds['baro_temp'] < 30)  

    ds['pressure'].attrs['long_name'] = 'atmospheric air pressure'
    ds['pressure'].attrs['standard_name'] = 'air_pressure'
    ds['pressure'].attrs['units'] = 'Pa'
    ds['pressure'].attrs['coverage_content_type'] = 'physicalMeasurement'
    
    #filtering values to stay below 1050 hPa
    ds['pressure'] = ds.air_temp.where(ds['pressure'] < 1050)  


    # magnometer is given in 1/100 th of mikro-T = 1e-9 T

    ds['mag_x'].attrs['long_name'] = 'Magnetometer X-axis'
    ds['mag_x'].attrs['units'] = r'1e-9 T'
    ds['mag_x'].attrs['coverage_content_type'] = 'physicalMeasurement'

    ds['mag_y'].attrs['long_name'] = 'Magnetometer Y-axis'
    ds['mag_y'].attrs['units'] = r'1e-9 T'
    ds['mag_y'].attrs['coverage_content_type'] = 'physicalMeasurement'

    ds['mag_z'].attrs['long_name'] = 'Magnetometer Z-axis'
    ds['mag_z'].attrs['units'] = r'1e-9 T'
    ds['mag_z'].attrs['coverage_content_type'] = 'physicalMeasurement'

    # accelerometer is given in milli-g. Standard unit is standard_free_fall. Multiply with 1000?
    ds['acc_x'] = ds['acc_x'] * 1e3
    ds['acc_x'].attrs['long_name'] = 'Accelerometer X-axis'
    ds['acc_x'].attrs['units'] = r'standard_free_fall'
    ds['acc_x'].attrs['coverage_content_type'] = 'physicalMeasurement'

    ds['acc_y'] = ds['acc_y'] * 1e3
    ds['acc_y'].attrs['long_name'] = 'Accelerometer Y-axis'
    ds['acc_y'].attrs['units'] = r'standard_free_fall'
    ds['acc_y'].attrs['coverage_content_type'] = 'physicalMeasurement'

    ds['acc_z'] = ds['acc_z'] * 1e3
    ds['acc_z'].attrs['long_name'] = 'Accelerometer Z-axis'
    ds['acc_z'].attrs['units'] = r'standard_free_fall'
    ds['acc_z'].attrs['coverage_content_type'] = 'physicalMeasurement'

    ds['air_temp'].attrs['long_name'] = 'Air temperature from in-air sensor 240'
    ds['air_temp'].attrs['standard_name'] = 'air_temperature'
    ds['air_temp'].attrs['units'] = 'degC'
    ds['air_temp'].attrs['coverage_content_type'] = 'physicalMeasurement'
    
    #filtering values to stay below 30 degrees
    ds['air_temp'] = ds.air_temp.where(ds['air_temp'] < 30)  
    
    ds['time2'].attrs['long_name'] = 'time'
    ds['time2'].attrs['standard_name'] = 'time'
    ds['time2'].attrs['coverage_content_type'] = 'referenceInformation'

    # global attributes
    
    cfg = parse_cfg(parse_arguments().cfgfile)

    ds.attrs['featureType'] = 'trajectory'
    ds.attrs['naming_authority'] = 'met.no'
    ds.attrs['summary'] = 'Data from drifting buoys deployed on the 29th of June by UiT, using Kronprins Haakon'
    ds.attrs['history'] = '30.08.2022: Data converted from BUFR to NetCDF-CF'
    ds.attrs['date_created'] = date.today().strftime("%B %d, %Y")
    ds.attrs['geospatial_lat_min'] = '{:.3f}'.format(ds['latitude'].values.min())
    ds.attrs['geospatial_lat_max'] = '{:.3f}'.format(ds['latitude'].values.max())
    ds.attrs['geospatial_lon_min'] = '{:.3f}'.format(ds['longitude'].values.min())
    ds.attrs['geospatial_lon_max'] = '{:.3f}'.format(ds['longitude'].values.max())   
    ds.attrs['keywords'] = (','.join(['GCMDSK: EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC TEMPERATURE > SURFACE TEMPERATURE > AIR TEMPERATURE',
                            'GCMDSK: EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC PRESSURE',
                            'GCMDSK: In Situ/Laboratory Instruments > Magnetic/Motion Sensor > Accelerometers',
                            'GCMDSK: In Situ/Laboratory Instruments > Magnetic/Motion Sensor > Accelerometers']))
    ds.attrs['keywords_vocabulary'] = (','.join(['GCMDSK:GCMD Science Keywords:https://gcmd.earthdata.nasa.gov/kms/concepts/concept_scheme/sciencekeywords',
                                       'GCMDSK:GCMD Instruments:https://gcmd.earthdata.nasa.gov/kms/concepts/concept_scheme/instruments']))
    ds.attrs['standard_name_vocabulary'] = 'CF Standard Name V79'
    ds.attrs['Conventions'] = 'ACDD-1.3, CF-1.8'
    ds.attrs['creator_type'] = 'None'
    ds.attrs['institution'] = cfg['author']['PrincipalInvestigatorOrganisation']
    ds.attrs['creator_name'] = cfg['author']['PrincipalInvestigator']
    ds.attrs['creator_email'] = cfg['author']['PrincipalInvestigatorEmail']
    ds.attrs['creator_url'] = cfg['author']['PrincipalInvestigatorOrganisationURL']
    ds.attrs['publisher_name'] = cfg['author']['Publisher']
    ds.attrs['publisher_email'] = cfg['author']['PublisherEmail']
    ds.attrs['publisher_url'] = cfg['author']['PublisherURL']
    ds.attrs['project'] = cfg['author']['Project']
    ds.attrs['license'] = cfg['author']['License']
    ds.attrs['processing_level'] = 'None'
    ds.attrs['references'] = 'None'

    for var in ds.keys():
        if ds[var].dtype == 'int64':
            ds[var] = ds[var].astype(np.int32) 
        if ds[var].dtype == 'float64':
            ds[var] = ds[var].astype(np.float32) 
        else:
            continue
        
    ds = ds.assign({'sensor': (('name_strlen'),np.array(['simba_300234065765510']))})
    ds['sensor'].attrs['long_name'] = 'Sensor identifier'
    ds['sensor'].attrs['units'] = '1'
    ds['sensor'].attrs['coverage_content_type'] = 'referenceInformation'
    ds = ds.fillna(-9999)    
    
    encoding = {i : {'_FillValue': -9999} for i in ds.keys() if isinstance(ds[i].values[0], str) != True}
    encoding['time1'] = {'units': 'seconds since 1970-01-01 00:00:00+0','dtype': 'int32',}
    encoding['time2'] = {'units': 'seconds since 1970-01-01 00:00:00+0','dtype': 'int32',}
    encoding['longitude'] = {'_FillValue': -9999,'dtype': 'float32',}
    encoding['latitude'] = {'_FillValue': -9999,'dtype': 'float32',}

    if parse_arguments().update:
        startdate = update_func(parse_arguments().cfgfile)

        get_index1 = list(ds.time1.values).index(ds.sel(time1=startdate, method='bfill').time1)
        first_slice = ds.isel(time1=slice(get_index1,-1))
        try:
            get_index2 = list(first_slice.time2.values).index(first_slice.sel(time2=startdate, method='bfill').time2)
            second_slice = first_slice.isel(time2=slice(get_index2,-1))
        except:
            second_slice = first_slice.drop_dims('time2')
        return second_slice
    if parse_arguments().startday and parse_arguments().endday is not None:
        try:
            startday = parse_arguments().startday
            endday = parse_arguments().endday
            get_index1 = list(ds.time1.values).index(ds.sel(time1=startday, method='bfill').time1)
            get_index2 = list(ds.time1.values).index(ds.sel(time1=endday, method='ffill').time1)
            first_slice = ds.isel(time1=slice(get_index1,get_index2))
            try:
                get_index3 = list(first_slice.time2.values).index(first_slice.sel(time2=startday, method='bfill').time2)
                get_index4 = list(first_slice.time2.values).index(first_slice.sel(time2=endday, method='ffill').time2)
                second_slice = first_slice.isel(time2=slice(get_index3,get_index4))
            except:
                second_slice = first_slice.drop_dims("time2")
            return second_slice
        except:
            return ds
    else:
        return ds

def simba80():
    path = parse_cfg(parse_arguments().cfgfile)['input']['path']
    simba80 = ['simba_300234065863280_gps_full.csv','simba_300234065863280_temper.csv','simba_300234065863280_status.csv','lt-ito_300234065863280.csv']
    gps = pd.read_csv(path + simba80[0])
    temp = pd.read_csv(path + simba80[1])
    status = pd.read_csv(path + simba80[2])
    light = pd.read_csv(path + simba80[3])

    gps_df = gps[['GPS_Time_dtm', 'Lat','Long','BaroTemp','Pressure','Mag_X','Mag_Y', 'Mag_Z', 'Acc_X', 'Acc_Y', 'Acc_Z']].copy()
    gps_df['GPS_Time_dtm'] = pd.to_datetime(gps_df['GPS_Time_dtm'])
    gps_df = gps_df.rename(columns={'GPS_Time_dtm': 'time1', 'Lat': 'latitude', 'Long': 'longitude',
                                    'BaroTemp': 'baro_temp', 'Pressure': 'pressure', 'Mag_X':'mag_x','Mag_Y':'mag_y',
                                    'Mag_Z': 'mag_z', 'Acc_X':'acc_x', 'Acc_Y': 'acc_y', 'Acc_Z':'acc_z'})
    gps_df = gps_df.set_index('time1')
    gps_ds = gps_df.to_xarray()

    gps_ds = gps_ds.assign_coords({'latitude': gps_ds['latitude'].values, 'longitude': gps_ds['longitude'].values})

    gps_ds['time1'].attrs['long_name'] = 'time'
    gps_ds['time1'].attrs['standard_name'] = 'time'
    gps_ds['time1'].attrs['coverage_content_type'] = 'referenceInformation'

    gps_ds['longitude'].attrs['long_name'] = 'longitude'
    gps_ds['longitude'].attrs['standard_name'] = 'longitude'
    gps_ds['longitude'].attrs['units'] = 'degrees_east'
    gps_ds['longitude'].attrs['coverage_content_type'] = 'coordinate'

    gps_ds['latitude'].attrs['long_name'] = 'latitude'
    gps_ds['latitude'].attrs['standard_name'] = 'latitude'
    gps_ds['latitude'].attrs['units'] = 'degrees_north'
    gps_ds['latitude'].attrs['coverage_content_type'] = 'coordinate'

    time = pd.to_datetime(temp["RTC_Time_DTM"])

    air_temp = temp['temp240']
    air_temp_da = xr.DataArray(air_temp, coords=[time], dims = ['time2'])
    
    gps_ds['air_temp'] = air_temp_da
    ds = gps_ds

    ds['baro_temp'].attrs['long_name'] = 'temperature of barometer'
    ds['baro_temp'].attrs['units'] = '1/16th DegC'
    ds['baro_temp'].attrs['coverage_content_type'] = 'physicalMeasurement'
    
    #filtering values to stay below 30 degrees
    ds['baro_temp'] = ds.air_temp.where(ds['baro_temp'] < 30)  

    ds['pressure'].attrs['long_name'] = 'atmospheric air pressure'
    ds['pressure'].attrs['standard_name'] = 'air_pressure'
    ds['pressure'].attrs['units'] = 'Pa'
    ds['pressure'].attrs['coverage_content_type'] = 'physicalMeasurement'
    
    #filtering values to stay below 1050 hPa
    ds['pressure'] = ds.air_temp.where(ds['pressure'] < 1050)  

    ds['mag_x'].attrs['long_name'] = 'Magnetometer X-axis'
    ds['mag_x'].attrs['units'] = r'1/100th $\mu$T'
    ds['mag_x'].attrs['coverage_content_type'] = 'physicalMeasurement'

    ds['mag_y'].attrs['long_name'] = 'Magnetometer Y-axis'
    ds['mag_y'].attrs['units'] = r'1/100th $\mu$T'
    ds['mag_y'].attrs['coverage_content_type'] = 'physicalMeasurement'

    ds['mag_z'].attrs['long_name'] = 'Magnetometer Z-axis'
    ds['mag_z'].attrs['units'] = r'1/100th $\mu$T'
    ds['mag_z'].attrs['coverage_content_type'] = 'physicalMeasurement'

    ds['acc_x'].attrs['long_name'] = 'Accelerometer X-axis'
    ds['acc_x'].attrs['units'] = r'milli-g'
    ds['acc_x'].attrs['coverage_content_type'] = 'physicalMeasurement'

    ds['acc_y'].attrs['long_name'] = 'Accelerometer Y-axis'
    ds['acc_y'].attrs['units'] = r'milli-g'
    ds['acc_y'].attrs['coverage_content_type'] = 'physicalMeasurement'

    ds['acc_z'].attrs['long_name'] = 'Accelerometer Z-axis'
    ds['acc_z'].attrs['units'] = r'milli-g'
    ds['acc_z'].attrs['coverage_content_type'] = 'physicalMeasurement'

    ds['air_temp'].attrs['long_name'] = 'Air temperature from in-air sensor 240'
    ds['air_temp'].attrs['standard_name'] = 'air_temperature'
    ds['air_temp'].attrs['units'] = 'degC'
    ds['air_temp'].attrs['coverage_content_type'] = 'physicalMeasurement'
    
    #filtering values to stay below 30 degrees
    ds['air_temp'] = ds.air_temp.where(ds['air_temp'] < 30)  

    ds['time2'].attrs['long_name'] = 'time'
    ds['time2'].attrs['standard_name'] = 'time'
    ds['time2'].attrs['coverage_content_type'] = 'referenceInformation'

    
    # global attributes
    cfg = parse_cfg(parse_arguments().cfgfile)
    
    ds.attrs['featureType'] = 'trajectory'
    ds.attrs['naming_authority'] = 'met.no'
    ds.attrs['summary'] = cfg['output']['abstract']
    ds.attrs['history'] = date.today().strftime("%B %d, %Y") + ": Data converted from CSV to netCDF"
    ds.attrs['date_created'] = date.today().strftime("%B %d, %Y") 
    ds.attrs['geospatial_lat_min'] = '{:.3f}'.format(ds['latitude'].values.min())
    ds.attrs['geospatial_lat_max'] = '{:.3f}'.format(ds['latitude'].values.max())
    ds.attrs['geospatial_lon_min'] = '{:.3f}'.format(ds['longitude'].values.min())
    ds.attrs['geospatial_lon_max'] = '{:.3f}'.format(ds['longitude'].values.max())
    ds.attrs['time_coverage_start'] = min(ds['time1'].values[0].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S'),
                                        ds['time2'].values[0].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S'))# note that the datetime is changed to microsecond precision from nanosecon precision
    ds.attrs['time_coverage_end'] = max(ds['time1'].values[-1].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S'),
                                        ds['time2'].values[-1].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S'))      
        
    duration_years = str(relativedelta(max(ds['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                        ds['time2'].values[-1].astype('datetime64[s]').astype(datetime)),
                                    min(ds['time1'].values[0].astype('datetime64[s]').astype(datetime),
                                        ds['time2'].values[0].astype('datetime64[s]').astype(datetime))).years)
    duration_months = str(relativedelta(max(ds['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                        ds['time2'].values[-1].astype('datetime64[s]').astype(datetime)),
                                    min(ds['time1'].values[0].astype('datetime64[s]').astype(datetime),
                                        ds['time2'].values[0].astype('datetime64[s]').astype(datetime))).months)
    duration_days = str(relativedelta(max(ds['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                        ds['time2'].values[-1].astype('datetime64[s]').astype(datetime)),
                                    min(ds['time1'].values[0].astype('datetime64[s]').astype(datetime),
                                        ds['time2'].values[0].astype('datetime64[s]').astype(datetime))).days)
    duration_hours = str(relativedelta(max(ds['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                        ds['time2'].values[-1].astype('datetime64[s]').astype(datetime)),
                                    min(ds['time1'].values[0].astype('datetime64[s]').astype(datetime),
                                        ds['time2'].values[0].astype('datetime64[s]').astype(datetime))).hours)
    duration_minutes = str(relativedelta(max(ds['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                        ds['time2'].values[-1].astype('datetime64[s]').astype(datetime)),
                                    min(ds['time1'].values[0].astype('datetime64[s]').astype(datetime),
                                        ds['time2'].values[0].astype('datetime64[s]').astype(datetime))).minutes)
    duration_seconds = str(relativedelta(max(ds['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                        ds['time2'].values[-1].astype('datetime64[s]').astype(datetime)),
                                    min(ds['time1'].values[0].astype('datetime64[s]').astype(datetime),
                                        ds['time2'].values[0].astype('datetime64[s]').astype(datetime))).seconds)
    ds.attrs['time_coverage_duration'] = ('P' + duration_years + 'Y' + duration_months +
                                                'M' + duration_days + 'DT' + duration_hours + 
                                                'H' + duration_minutes + 'M' + duration_seconds + 'S')    
        
    ds.attrs['keywords'] = (','.join(['GCMDSK: EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC TEMPERATURE > SURFACE TEMPERATURE > AIR TEMPERATURE',
                            'GCMDSK: EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC PRESSURE',
                            'GCMDSK: In Situ/Laboratory Instruments > Magnetic/Motion Sensor > Accelerometers',
                            'GCMDSK: In Situ/Laboratory Instruments > Magnetic/Motion Sensor > Accelerometers']))
    ds.attrs['keywords_vocabulary'] = (','.join(['GCMDSK:GCMD Science Keywords:https://gcmd.earthdata.nasa.gov/kms/concepts/concept_scheme/sciencekeywords',
                                       'GCMDSK:GCMD Instruments:https://gcmd.earthdata.nasa.gov/kms/concepts/concept_scheme/instruments']))
    ds.attrs['standard_name_vocabulary'] = 'CF Standard Name V79'
    ds.attrs['Conventions'] = 'ACDD-1.3, CF-1.8'
    ds.attrs['creator_type'] = 'None'
    ds.attrs['institution'] = cfg['author']['PrincipalInvestigatorOrganisation']
    ds.attrs['creator_name'] = cfg['author']['PrincipalInvestigator']
    ds.attrs['creator_email'] = cfg['author']['PrincipalInvestigatorEmail']
    ds.attrs['creator_url'] = cfg['author']['PrincipalInvestigatorOrganisationURL']
    ds.attrs['publisher_name'] = cfg['author']['Publisher']
    ds.attrs['publisher_email'] = cfg['author']['PublisherEmail']
    ds.attrs['publisher_url'] = cfg['author']['PublisherURL']
    ds.attrs['project'] = cfg['author']['Project']
    ds.attrs['license'] = cfg['author']['License']
    ds.attrs['processing_level'] = 'None'
    ds.attrs['references'] = 'None'

    for var in ds.keys():
        if ds[var].dtype == 'int64':
            ds[var] = ds[var].astype(np.int32) 
        if ds[var].dtype == 'float64':
            ds[var] = ds[var].astype(np.float32) 
        else:
            continue
        
    ds = ds.assign({'sensor': (('name_strlen'),np.array(['simba_300234065863280']))})
    ds['sensor'].attrs['long_name'] = 'Sensor identifier'
    ds['sensor'].attrs['units'] = '1'
    ds['sensor'].attrs['coverage_content_type'] = 'referenceInformation'
    ds = ds.fillna(-9999)  
        
    encoding = {i : {'_FillValue': -9999} for i in ds.keys() if isinstance(ds[i].values[0], str) != True}
    encoding['time1'] = {'units': 'seconds since 1970-01-01 00:00:00+0','dtype': 'int32',}
    encoding['time2'] = {'units': 'seconds since 1970-01-01 00:00:00+0','dtype': 'int32',}
    encoding['longitude'] = {'_FillValue': -9999,'dtype': 'float32',}
    encoding['latitude'] = {'_FillValue': -9999,'dtype': 'float32',}

    if parse_arguments().update:
        startdate = update_func(parse_arguments().cfgfile)
        get_index1 = list(ds.time1.values).index(ds.sel(time1=startdate, method='bfill').time1)
        first_slice = ds.isel(time1=slice(get_index1,-1))
        try:
            get_index2 = list(first_slice.time2.values).index(first_slice.sel(time2=startdate, method='bfill').time2)
            second_slice = first_slice.isel(time2=slice(get_index2,-1))
        except:
            second_slice = first_slice.drop_dims('time2')
        return second_slice
    if parse_arguments().startday and parse_arguments().endday is not None:
        try:
            startday = parse_arguments().startday
            endday = parse_arguments().endday
            get_index1 = list(ds.time1.values).index(ds.sel(time1=startday, method='bfill').time1)
            get_index2 = list(ds.time1.values).index(ds.sel(time1=endday, method='ffill').time1)
            first_slice = ds.isel(time1=slice(get_index1,get_index2))
            try:
                get_index3 = list(first_slice.time2.values).index(first_slice.sel(time2=startday, method='bfill').time2)
                get_index4 = list(first_slice.time2.values).index(first_slice.sel(time2=endday, method='ffill').time2)
                second_slice = first_slice.isel(time2=slice(get_index3,get_index4))
            except:
                second_slice = first_slice.drop_dims("time2")
            return second_slice
        except:
            return ds
    else:
        return ds

def aws():
    path = parse_cfg(parse_arguments().cfgfile)['input']['path']
    file = 'aws_300234068660540.csv'
    weather = pd.read_csv(path + file)
    weather = weather[['wind_u', 'wind_w', 'temper', 'humidity', 'pressure', 'dtm','latitude','longitude']].copy()
    weather = weather.rename(columns={'dtm': 'time','temper':'temperature'})
    weather['time'] = pd.to_datetime(weather['time'])
    weather = weather.set_index('time')
    weather = weather.rename(columns={'temper':'temperature'})

    weather_ds = weather.to_xarray()
    weather_ds = weather_ds.assign_coords({'latitude': weather_ds['latitude'].values, 'longitude': weather_ds['longitude'].values})

    weather_ds['time'].attrs['long_name'] = 'time'
    weather_ds['time'].attrs['standard_name'] = 'time'
    weather_ds['time'].attrs['coverage_content_type'] = 'referenceInformation'

    weather_ds['longitude'].attrs['long_name'] = 'longitude'
    weather_ds['longitude'].attrs['standard_name'] = 'longitude'
    weather_ds['longitude'].attrs['units'] = 'degrees_east'
    weather_ds['longitude'].attrs['coverage_content_type'] = 'coordinate'

    weather_ds['latitude'].attrs['long_name'] = 'latitude'
    weather_ds['latitude'].attrs['standard_name'] = 'latitude'
    weather_ds['latitude'].attrs['units'] = 'degrees_north'
    weather_ds['latitude'].attrs['coverage_content_type'] = 'coordinate'

    weather_ds['wind_u'].attrs['long_name'] = 'wind in the west-east direction'
    weather_ds['wind_u'].attrs['standard_name'] = 'eastward_wind'
    weather_ds['wind_u'].attrs['units'] = 'm s-1'
    weather_ds['wind_u'].attrs['coverage_content_type'] = 'physicalMeasurement'
    
    #filtering wind_u between -50 and 50 m/s
    weather_ds['wind_u'] = weather_ds.wind_u.where(weather_ds['wind_u'] < 50)  
    weather_ds['wind_u'] = weather_ds.wind_u.where(weather_ds['wind_u'] > -50)  

    weather_ds['wind_w'].attrs['long_name'] = ''
    weather_ds['wind_w'].attrs['standard_name'] = ''
    weather_ds['wind_w'].attrs['units'] = 'm s-1'
    weather_ds['wind_w'].attrs['coverage_content_type'] = 'physicalMeasurement'
    
    #filtering wind_w between -50 and 50 m/s
    weather_ds['wind_w'] = weather_ds.wind_w.where(weather_ds['wind_w'] < 50)  
    weather_ds['wind_w'] = weather_ds.wind_w.where(weather_ds['wind_w'] > -50)  

    weather_ds['temperature'].attrs['long_name'] = 'air temperature'
    weather_ds['temperature'].attrs['standard_name'] = 'air_temperature'
    weather_ds['temperature'].attrs['units'] = 'degC'
    weather_ds['temperature'].attrs['coverage_content_type'] = 'physicalMeasurement'
    
    #filtering temperature to be below 30 degrees
    weather_ds['temperature'] = weather_ds.temperature.where(weather_ds['temperature'] < 30) 

    weather_ds['humidity'].attrs['long_name'] = 'relative humidity'
    weather_ds['humidity'].attrs['standard_name'] = 'relative_humidity'
    weather_ds['humidity'].attrs['units'] = '%'
    weather_ds['humidity'].attrs['coverage_content_type'] = 'physicalMeasurement'
    
    weather_ds['humidity'] = weather_ds.humidity.where(weather_ds['humidity'] < 101)  
    weather_ds['humidity'] = weather_ds.humidity.where(weather_ds['humidity'] > 0)  

    weather_ds['pressure'].attrs['long_name'] = 'air pressure'
    weather_ds['pressure'].attrs['standard_name'] = 'air_pressure'
    weather_ds['pressure'].attrs['units'] = 'Pa'
    weather_ds['pressure'].attrs['coverage_content_type'] = 'physicalMeasurement'
    
    weather_ds['pressure'] = weather_ds.pressure.where(weather_ds['pressure'] < 1050)    

    # global attributes

    weather_ds.attrs['featureType'] = 'trajectory'
    weather_ds.attrs['naming_authority'] = 'met.no'
    weather_ds.attrs['summary'] = 'Data from drifting buoys deployed on the 29th of June by UiT, using Kronprins Haakon'
    weather_ds.attrs['history'] = '30.08.2022: Data converted from BUFR to NetCDF-CF'
    weather_ds.attrs['date_created'] = '30.08.2022'
    weather_ds.attrs['geospatial_lat_min'] = '{:.3f}'.format(weather_ds['latitude'].values.min())
    weather_ds.attrs['geospatial_lat_max'] = '{:.3f}'.format(weather_ds['latitude'].values.max())
    weather_ds.attrs['geospatial_lon_min'] = '{:.3f}'.format(weather_ds['longitude'].values.min())
    weather_ds.attrs['geospatial_lon_max'] = '{:.3f}'.format(weather_ds['longitude'].values.max())
    weather_ds.attrs['keywords'] = 'keywords_here'
    weather_ds.attrs['keywords_vocabulary'] = 'keywords_voc_here'
    weather_ds.attrs['standard_name_vocabulary'] = 'CF Standard Name V79'
    weather_ds.attrs['Conventions'] = 'ACDD-1.3, CF-1.8'
    weather_ds.attrs['creator_type'] = 'None'
    weather_ds.attrs['institution'] = 'None'
    weather_ds.attrs['creator_name'] = 'MET'
    weather_ds.attrs['creator_email'] = 'met@met.no'
    weather_ds.attrs['creator_url'] = 'met.no'

    for var in weather_ds.keys():
        if weather_ds[var].dtype == 'int64':
            weather_ds[var] = weather_ds[var].astype(np.int32) 
        if weather_ds[var].dtype == 'float64':
            weather_ds[var] = weather_ds[var].astype(np.float32) 
        else:
            continue
        
    weather_ds = weather_ds.assign({'sensor': (('name_strlen'),np.array(['aws_300234068660540']))})
    weather_ds['sensor'].attrs['long_name'] = 'Sensor identifier'
    weather_ds['sensor'].attrs['units'] = '1'
    weather_ds['sensor'].attrs['coverage_content_type'] = 'referenceInformation'
    weather_ds = weather_ds.fillna(-9999)
        
    encoding = {i : {'_FillValue': -9999} for i in weather_ds.keys() if isinstance(weather_ds[i].values[0], str) != True}
    encoding['time'] = {'units': 'seconds since 1970-01-01 00:00:00+0','dtype': 'int32',}
    encoding['longitude'] = {'_FillValue': -9999,'dtype': 'float32',}
    encoding['latitude'] = {'_FillValue': -9999,'dtype': 'float32',}

    if parse_arguments().update:
        startdate = update_func(parse_arguments().cfgfile)
        weather_ds = weather_ds.drop_duplicates('time')
        weather_ds = weather_ds.sortby('time')
        get_index1 = list(weather_ds.time.values).index(weather_ds.sel(time=startdate, method='bfill').time)
        first_slice = weather_ds.isel(time=slice(get_index1,-1))
        return first_slice
    if parse_arguments().startday and parse_arguments().endday is not None:
        try:
            startday = parse_arguments().startday
            endday = parse_arguments().endday
            weather_ds = weather_ds.drop_duplicates('time')
            weather_ds = weather_ds.sortby('time')
            get_index1 = list(weather_ds.time1.values).index(weather_ds.sel(time1=startday, method='bfill').time1)
            get_index2 = list(weather_ds.time1.values).index(weather_ds.sel(time1=endday, method='ffill').time1)
            first_slice = weather_ds.isel(time1=slice(get_index1,get_index2))
            return first_slice
        except:
            return weather_ds
    else:
        return weather_ds

if __name__ == "__main__":
    
    saveto = parse_cfg(parse_arguments().cfgfile)['output']['destdir']

    if parse_arguments().station == 'aws':
        ds = aws()
        
        encoding = {i : {'_FillValue': -9999} for i in ds.keys() if isinstance(ds[i].values[0], str) != True}
        encoding['time'] = {'units': 'seconds since 1970-01-01 00:00:00+0','dtype': 'int32',}
        encoding['longitude'] = {'_FillValue': -9999,'dtype': 'float32',}
        encoding['latitude'] = {'_FillValue': -9999,'dtype': 'float32',}
        
        gb = ds.groupby('time.month')
        datasets = []
        for group_name, group_da in gb:
            #datasets.append(group_da)
            t = group_da.time.isel(time=-1).values.astype('datetime64[s]')
            t = t.astype(datetime)
            timestring = t.strftime('%Y%m')
            if parse_arguments().startday and parse_arguments().endday is not None:
                timestring1 = parse_arguments().startday.replace('-','')
                timestring2 = parse_arguments().endday.replace('-','')
                timestring = timestring1 + '-' + timestring2
            filename = 'aws_300234068660540_{}.nc'.format(timestring)
            if filename in os.listdir(saveto):
                os.remove("{}/{}".format(saveto, filename))
            group_da.attrs['time_coverage_start'] = group_da['time'].values[0].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S')# note that the datetime is changed to microsecond precision from nanosecon precision
            group_da.attrs['time_coverage_end'] = group_da['time'].values[-1].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S')     

            duration_years = str(relativedelta(group_da['time'].values[-1].astype('datetime64[s]').astype(datetime),
                                            group_da['time'].values[0].astype('datetime64[s]').astype(datetime)).years)
            duration_months = str(relativedelta(group_da['time'].values[-1].astype('datetime64[s]').astype(datetime),
                                            group_da['time'].values[0].astype('datetime64[s]').astype(datetime)).months)
            duration_days = str(relativedelta(group_da['time'].values[-1].astype('datetime64[s]').astype(datetime),
                                            group_da['time'].values[0].astype('datetime64[s]').astype(datetime)).days)
            duration_hours = str(relativedelta(group_da['time'].values[-1].astype('datetime64[s]').astype(datetime),
                                            group_da['time'].values[0].astype('datetime64[s]').astype(datetime)).hours)
            duration_minutes = str(relativedelta(group_da['time'].values[-1].astype('datetime64[s]').astype(datetime),
                                            group_da['time'].values[0].astype('datetime64[s]').astype(datetime)).minutes)
            duration_seconds = str(relativedelta(group_da['time'].values[-1].astype('datetime64[s]').astype(datetime),
                                            group_da['time'].values[0].astype('datetime64[s]').astype(datetime)).seconds)
            group_da.attrs['time_coverage_duration'] = ('P' + duration_years + 'Y' + duration_months +
                                                        'M' + duration_days + 'DT' + duration_hours + 
                                                        'H' + duration_minutes + 'M' + duration_seconds + 'S')                    
            group_da.attrs['title'] = ('Buoy aws_300234068660540, from ' + group_da.attrs['time_coverage_start'] +\
                                            ' to ' + group_da.attrs['time_coverage_end'])            
            group_da.to_netcdf('{}/{}'.format(saveto, filename), engine='netcdf4', encoding=encoding)
            #group_da.to_netcdf('{}/aws_300234068660540_{}.nc'.format(saveto,timestring), engine='netcdf4', encoding=encoding)
            
    if parse_arguments().station == 'simba10':
        ds = simba10()

        encoding = {i : {'_FillValue': -9999} for i in ds.keys() if isinstance(ds[i].values[0], str) != True}
        encoding['time1'] = {'units': 'seconds since 1970-01-01 00:00:00+0','dtype': 'int32',}
        encoding['longitude'] = {'_FillValue': -9999,'dtype': 'float32',}
        encoding['latitude'] = {'_FillValue': -9999,'dtype': 'float32',}
    

        if 'time2' in ds.variables:
            
            gb1 = ds.groupby('time1.month')
            sets = []
            for group_name, group_da in gb1:
                sets.append(group_da)
            
            datasets = []
            
            for i in sets:
                finished = i.groupby("time2.month")
                for group_name, group_da in finished:
                    datasets.append(group_da)
                    t = group_da.time1.isel(time1=-1).values.astype('datetime64[s]')
                    t = t.astype(datetime)
                    timestring = t.strftime('%Y%m')
                    if parse_arguments().startday and parse_arguments().endday is not None:
                        timestring1 = parse_arguments().startday.replace('-','')
                        timestring2 = parse_arguments().endday.replace('-','')
                        timestring = timestring1 + '-' + timestring2
                    filename = 'simba_300234065765510_{}.nc'.format(timestring)
                    if filename in os.listdir(saveto):
                        os.remove("{}/{}".format(saveto, filename))          
                    group_da.attrs['time_coverage_start'] = min(group_da['time1'].values[0].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S'),
                                            group_da['time2'].values[0].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S'))# note that the datetime is changed to microsecond precision from nanosecon precision
                    group_da.attrs['time_coverage_end'] = max(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S'),
                                            group_da['time2'].values[-1].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S'))      
            
                    duration_years = str(relativedelta(max(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[-1].astype('datetime64[s]').astype(datetime)),
                                        min(group_da['time1'].values[0].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[0].astype('datetime64[s]').astype(datetime))).years)
                    duration_months = str(relativedelta(max(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[-1].astype('datetime64[s]').astype(datetime)),
                                        min(group_da['time1'].values[0].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[0].astype('datetime64[s]').astype(datetime))).months)
                    duration_days = str(relativedelta(max(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[-1].astype('datetime64[s]').astype(datetime)),
                                        min(group_da['time1'].values[0].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[0].astype('datetime64[s]').astype(datetime))).days)
                    duration_hours = str(relativedelta(max(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[-1].astype('datetime64[s]').astype(datetime)),
                                        min(group_da['time1'].values[0].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[0].astype('datetime64[s]').astype(datetime))).hours)
                    duration_minutes = str(relativedelta(max(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[-1].astype('datetime64[s]').astype(datetime)),
                                        min(group_da['time1'].values[0].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[0].astype('datetime64[s]').astype(datetime))).minutes)
                    duration_seconds = str(relativedelta(max(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[-1].astype('datetime64[s]').astype(datetime)),
                                        min(group_da['time1'].values[0].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[0].astype('datetime64[s]').astype(datetime))).seconds)
                    group_da.attrs['time_coverage_duration'] = ('P' + duration_years + 'Y' + duration_months +
                                                    'M' + duration_days + 'DT' + duration_hours + 
                                                    'H' + duration_minutes + 'M' + duration_seconds + 'S') 
                    group_da.attrs['title'] = ('Buoy simba_300234065765510, from ' + group_da.attrs['time_coverage_start']+ \
                                                ' to ' + group_da.attrs['time_coverage_end'])  
                    encoding['time2'] = {'units': 'seconds since 1970-01-01 00:00:00+0','dtype': 'int32',}
                    group_da.to_netcdf('{}/{}'.format(saveto, filename), engine='netcdf4', encoding=encoding)
        else:
            gb = ds.groupby('time1.month')
            datasets = []
            for group_name, group_da in gb:
                #datasets.append(group_da)
                t = group_da.time1.isel(time1=-1).values.astype('datetime64[s]')
                t = t.astype(datetime)
                timestring = t.strftime('%Y%m')
                if parse_arguments().startday and parse_arguments().endday is not None:
                    timestring1 = parse_arguments().startday.replace('-','')
                    timestring2 = parse_arguments().endday.replace('-','')
                    timestring = timestring1 + '-' + timestring2
                filename = 'simba_300234065765510_{}.nc'.format(timestring)
                if filename in os.listdir(saveto):
                    os.remove("{}/{}".format(saveto, filename))
                group_da.attrs['time_coverage_start'] = group_da['time1'].values[0].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S')# note that the datetime is changed to microsecond precision from nanosecon precision
                group_da.attrs['time_coverage_end'] = group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S')     

                duration_years = str(relativedelta(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                                group_da['time1'].values[0].astype('datetime64[s]').astype(datetime)).years)
                duration_months = str(relativedelta(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                                group_da['time1'].values[0].astype('datetime64[s]').astype(datetime)).months)
                duration_days = str(relativedelta(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                                group_da['time1'].values[0].astype('datetime64[s]').astype(datetime)).days)
                duration_hours = str(relativedelta(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                                group_da['time1'].values[0].astype('datetime64[s]').astype(datetime)).hours)
                duration_minutes = str(relativedelta(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                                group_da['time1'].values[0].astype('datetime64[s]').astype(datetime)).minutes)
                duration_seconds = str(relativedelta(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                                group_da['time1'].values[0].astype('datetime64[s]').astype(datetime)).seconds)
                group_da.attrs['time_coverage_duration'] = ('P' + duration_years + 'Y' + duration_months +
                                                            'M' + duration_days + 'DT' + duration_hours + 
                                                            'H' + duration_minutes + 'M' + duration_seconds + 'S')                    
                group_da.attrs['title'] = ('Buoy simba_300234065765510_{}.nc, from ' + group_da.attrs['time_coverage_start'] +\
                                                ' to ' + group_da.attrs['time_coverage_end'])            
                group_da.to_netcdf('{}/{}'.format(saveto, filename), engine='netcdf4', encoding=encoding)
                #group_da.to_netcdf('{}/simba_300234065765510_{}.nc'.format(saveto,timestring), engine='netcdf4', encoding=encoding)
    
    if parse_arguments().station == 'simba80':
        ds = simba80()
        encoding = {i : {'_FillValue': -9999} for i in ds.keys() if isinstance(ds[i].values[0], str) != True}
        encoding['time1'] = {'units': 'seconds since 1970-01-01 00:00:00+0','dtype': 'int32',}
        encoding['longitude'] = {'_FillValue': -9999,'dtype': 'float32',}
        encoding['latitude'] = {'_FillValue': -9999,'dtype': 'float32',}
        
        if 'time2' in ds.variables:        
            gb1 = ds.groupby('time1.month')
            sets = []
            for group_name, group_da in gb1:
                sets.append(group_da)

            datasets = []
            for i in sets:
                finished = i.groupby("time2.month")
                for group_name, group_da in finished:
                    datasets.append(group_da)
                    t = group_da.time1.isel(time1=-1).values.astype('datetime64[s]')
                    t = t.astype(datetime)
                    timestring = t.strftime('%Y%m')
                    if parse_arguments().startday and parse_arguments().endday is not None:
                        timestring1 = parse_arguments().startday.replace('-','')
                        timestring2 = parse_arguments().endday.replace('-','')
                        timestring = timestring1 + '-' + timestring2        
                    filename = 'simba_300234065863280_{}.nc'.format(timestring)
                    if filename in os.listdir(saveto):
                        os.remove("{}/{}".format(saveto, filename))
                    group_da.attrs['time_coverage_start'] = min(group_da['time1'].values[0].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S'),
                                            group_da['time2'].values[0].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S'))# note that the datetime is changed to microsecond precision from nanosecon precision
                    group_da.attrs['time_coverage_end'] = max(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S'),
                                            group_da['time2'].values[-1].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S'))      
            
                    duration_years = str(relativedelta(max(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[-1].astype('datetime64[s]').astype(datetime)),
                                        min(group_da['time1'].values[0].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[0].astype('datetime64[s]').astype(datetime))).years)
                    duration_months = str(relativedelta(max(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[-1].astype('datetime64[s]').astype(datetime)),
                                        min(group_da['time1'].values[0].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[0].astype('datetime64[s]').astype(datetime))).months)
                    duration_days = str(relativedelta(max(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[-1].astype('datetime64[s]').astype(datetime)),
                                        min(group_da['time1'].values[0].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[0].astype('datetime64[s]').astype(datetime))).days)
                    duration_hours = str(relativedelta(max(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[-1].astype('datetime64[s]').astype(datetime)),
                                        min(group_da['time1'].values[0].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[0].astype('datetime64[s]').astype(datetime))).hours)
                    duration_minutes = str(relativedelta(max(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[-1].astype('datetime64[s]').astype(datetime)),
                                        min(group_da['time1'].values[0].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[0].astype('datetime64[s]').astype(datetime))).minutes)
                    duration_seconds = str(relativedelta(max(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[-1].astype('datetime64[s]').astype(datetime)),
                                        min(group_da['time1'].values[0].astype('datetime64[s]').astype(datetime),
                                            group_da['time2'].values[0].astype('datetime64[s]').astype(datetime))).seconds)
                    group_da.attrs['time_coverage_duration'] = ('P' + duration_years + 'Y' + duration_months +
                                                    'M' + duration_days + 'DT' + duration_hours + 
                                                    'H' + duration_minutes + 'M' + duration_seconds + 'S')
                    group_da.attrs['title'] = ('Buoy simba_300234065863280, from ' + group_da.attrs['time_coverage_start'] +\
                                                ' to ' + group_da.attrs['time_coverage_end'])
                    encoding['time2'] = {'units': 'seconds since 1970-01-01 00:00:00+0','dtype': 'int32',}
                    group_da.to_netcdf('{}/{}'.format(saveto, filename), engine='netcdf4', encoding=encoding)
        else:
            gb = ds.groupby('time1.month')
            datasets = []
            for group_name, group_da in gb:
                #datasets.append(group_da)
                t = group_da.time1.isel(time1=-1).values.astype('datetime64[s]')
                t = t.astype(datetime)
                timestring = t.strftime('%Y%m')
                if parse_arguments().startday and parse_arguments().endday is not None:
                    timestring1 = parse_arguments().startday.replace('-','')
                    timestring2 = parse_arguments().endday.replace('-','')
                    timestring = timestring1 + '-' + timestring2
                filename = 'simba_300234065863280_{}.nc'.format(timestring)
                if filename in os.listdir(saveto):
                    os.remove("{}/{}".format(saveto, filename))
                group_da.attrs['time_coverage_start'] = group_da['time1'].values[0].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S')# note that the datetime is changed to microsecond precision from nanosecon precision
                group_da.attrs['time_coverage_end'] = group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d %H:%M:%S')     

                duration_years = str(relativedelta(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                                group_da['time1'].values[0].astype('datetime64[s]').astype(datetime)).years)
                duration_months = str(relativedelta(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                                group_da['time1'].values[0].astype('datetime64[s]').astype(datetime)).months)
                duration_days = str(relativedelta(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                                group_da['time1'].values[0].astype('datetime64[s]').astype(datetime)).days)
                duration_hours = str(relativedelta(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                                group_da['time1'].values[0].astype('datetime64[s]').astype(datetime)).hours)
                duration_minutes = str(relativedelta(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                                group_da['time1'].values[0].astype('datetime64[s]').astype(datetime)).minutes)
                duration_seconds = str(relativedelta(group_da['time1'].values[-1].astype('datetime64[s]').astype(datetime),
                                                group_da['time1'].values[0].astype('datetime64[s]').astype(datetime)).seconds)
                group_da.attrs['time_coverage_duration'] = ('P' + duration_years + 'Y' + duration_months +
                                                            'M' + duration_days + 'DT' + duration_hours + 
                                                            'H' + duration_minutes + 'M' + duration_seconds + 'S')                    
                group_da.attrs['title'] = ('Buoy simba_300234065863280_{}.nc, from ' + group_da.attrs['time_coverage_start'] +\
                                                ' to ' + group_da.attrs['time_coverage_end'])            
                group_da.to_netcdf('{}/{}'.format(saveto, filename), engine='netcdf4', encoding=encoding)
                #group_da.to_netcdf('{}/simba_300234065765510_{}.nc'.format(saveto,timestring), engine='netcdf4', encoding=encoding)
