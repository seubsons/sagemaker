import xarray as xr
import pandas as pd
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import numpy as np
from prophet import Prophet


def plot(latitude, longitude):

    filename = 'heatidx_012020_072023_06.nc'
    ds = xr.open_dataset(filename)

    start_date = '2020-01-01'
    end_date = '2023-07-01'


    # Convert date strings to datetime.date objects
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()

    # Create a boolean mask for the date range
    mask = (ds['date'].dt.date >= start_date_obj) & (ds['date'].dt.date <= end_date_obj)

    filtered_ds = ds.sel(time=mask)

    hi_data = filtered_ds['heat_index'].squeeze()
    hi_time = filtered_ds['date']


    # INPUT LATITUDE, LONGITUDE OF INTEREST
    # latitude = 16.43
    # longitude = 100.12

    lat_str = str(latitude)
    lon_str = str(longitude)

    # Assuming hi_data is a NumPy array of shape (num_samples, 65, 37)
    num_samples = hi_data.shape[0]
    # hi_loc = np.zeros(num_samples)
    # timestamp_array = np.zeros(num_samples)

    heat_index_array = []
    timestamp_array = []

    for i in range(num_samples):
        fdata = hi_data[i, :, :].squeeze()
        fdata_loc = bilinear_interpolation(latitude, longitude, fdata)
        fdata_loc_c = 5.0/9.0*(fdata_loc - 32.0)

        heat_index_array.append(round(fdata_loc_c,2))

        timestamp = hi_time[i].dt.date
        timestamp_array.append(timestamp.values)
        # hi_loc[i] = fdata_loc
        # timestamp_array[i] = hi_time[i]

    data = {'date': timestamp_array, 'heat_index': heat_index_array}
    df = pd.DataFrame(data)

    df.set_index('date', inplace=True)


    # Assuming df is your DataFrame with 'heat_index' column and datetime index
    plt.figure(figsize=(10, 6))

    # Convert datetime index to numerical values
    x_values = mdates.date2num(df.index)

    plt.plot(x_values, df['heat_index'], marker='.')
    plt.title(f'Heat Index at ({lat_str}, {lon_str}), 06:00 UTC',fontsize=20)
    plt.xlabel('Date')
    plt.ylabel('Heat Index')
    plt.grid(True)

    # Format x-axis labels as dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Rotate x-axis labels for better readability
    plt.gcf().autofmt_xdate()

    #plt.savefig('test.png', dpi=200, bbox_inches='tight')
    #plt.show()

    return timestamp_array, heat_index_array


def predict(latitude, longitude, start_date_training, end_date_training, start_date_testing, end_date_testing, fdays):
    
    filename = 'heatidx_012020_072023_06.nc'
    ds = xr.open_dataset(filename)

    start_date = '2020-01-01'
    end_date = '2023-07-01'


    # Convert date strings to datetime.date objects
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
    

    # Create a boolean mask for the date range
    mask = (ds['date'].dt.date >= start_date_obj) & (ds['date'].dt.date <= end_date_obj)    

    filtered_ds = ds.sel(time=mask)
    
    hi_data = filtered_ds['heat_index'].squeeze()
    hi_time = filtered_ds['date']


    # INPUT LATITUDE, LONGITUDE OF INTEREST
    # latitude = 16.43
    # longitude = 100.12

    lat_str = str(latitude)
    lon_str = str(longitude)

    # Assuming hi_data is a NumPy array of shape (num_samples, 65, 37)
    num_samples = hi_data.shape[0]
    # hi_loc = np.zeros(num_samples)
    # timestamp_array = np.zeros(num_samples)

    heat_index_array = []
    timestamp_array = []

    for i in range(num_samples):
        fdata = hi_data[i, :, :].squeeze()
        fdata_loc = bilinear_interpolation(latitude, longitude, fdata)
        fdata_loc_c = 5.0/9.0*(fdata_loc - 32.0)

        heat_index_array.append(round(fdata_loc_c,2))

        timestamp = hi_time[i].dt.date
        timestamp_array.append(timestamp.values)
        # hi_loc[i] = fdata_loc
        # timestamp_array[i] = hi_time[i]

    #prediction start here
    ds, y = timestamp_array, heat_index_array

    date_list = [dt.item() for dt in ds]
    y_list = y

    data = {'ds': date_list, 'y': y_list}
    df2 = pd.DataFrame(data)

    #####################################
    df3 = df2.copy()
    df3['ds'] = pd.to_datetime(df2['ds'])
    
    # start_date_training = '2021-02-01'
    # end_date_training = '2022-11-30'
    # Filter the DataFrame based on the time period
    training_df = df3[(df3['ds'] >= start_date_training) & (df3['ds'] <= end_date_training)]
    testing_df = df3[(df3['ds'] >= start_date_testing) & (df3['ds'] <= end_date_testing)]

    
    
    m = Prophet()
    #m.fit(df2)
    m.fit(training_df)
    
    future = m.make_future_dataframe(periods=fdays)
    forecast = m.predict(future)
    
    # Plot forecast
    fig = m.plot(forecast)
    plt.xlabel('Date')
    plt.ylabel('Heat Index (°C)')
    plt.ylim([0,120])
    #plt.title('Heat Index, '+city_name)
    plt.title(f'Heat Index at ({lat_str}, {lon_str}), 06:00 UTC',fontsize=20)

    # Format x-axis ticks to show year and specific months (January and July)
    date_format = mdates.DateFormatter('%Y-%b')
    plt.gca().xaxis.set_major_formatter(date_format)

    # Set the locator to show ticks only for January and July
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))

    plt.gcf().autofmt_xdate()

    #plt.savefig('test_predict.png', dpi=200, bbox_inches='tight')
    plt.show()

    return forecast, training_df, testing_df
    
    
def bilinear_interpolation(lat, lon, data_array_orig):
    
    #flip image upside down before interpolation
    data_array = np.flipud(data_array_orig)

    latitudes = np.arange(65) * 0.25 + 5
    longitudes = np.arange(37) * 0.25 + 97

    lat_index = int((lat - 5.0) / 0.25)
    lon_index = int((lon - 97.0) / 0.25)

    #print(lat_index,lon_index)
    lat_index0 = lat_index
    lat_index1 = lat_index + 1
    if lat_index1 > 64:
        lat_index1= 64
        
    lon_index0 = lon_index
    lon_index1 = lon_index + 1
    if lon_index1 > 36:
        lon_index1 = 36
    
    
    lat_fraction = (lat - latitudes[lat_index]) / 0.25
    lon_fraction = (lon - longitudes[lon_index]) / 0.25

    #print(lat_fraction,lon_fraction)
    
    
    # top_left = data_array[lat_index, lon_index]
    # top_right = data_array[lat_index, lon_index + 1]
    top_left = data_array[lat_index0, lon_index0]
    top_right = data_array[lat_index0, lon_index1]

    #print(top_left,top_right)
    
    # bottom_left = data_array[lat_index + 1, lon_index]
    # bottom_right = data_array[lat_index + 1, lon_index + 1]
    bottom_left = data_array[lat_index1, lon_index0]
    bottom_right = data_array[lat_index1, lon_index1]

    #print(bottom_left, bottom_right)
    
    interpolated_value = (1 - lon_fraction) * (1 - lat_fraction) * top_left \
                         + lon_fraction * (1 - lat_fraction) * top_right \
                         + (1 - lon_fraction) * lat_fraction * bottom_left \
                         + lon_fraction * lat_fraction * bottom_right
    
    return round(interpolated_value,2)

