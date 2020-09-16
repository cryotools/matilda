# Install cdsapi if not done already. (e.g pip install cdsapi)
# Write your UID and API-Keay from CDS in the config file:

# ~/.cdsapirc
# url: https://cds.climate.copernicus.eu/api/v2
# key: <UID>:<API key>
# verify: 0

import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-land',
    {
        'format': 'netcdf',
        'variable': [
            '2m_temperature', 'surface_pressure',
        ],
        'year': [
            "2017", "2018", "2019",
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'area': ['40.7/75.7/41.2/76.2'
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
    },
    '20200916_era5_t2_sp_2017-2019.nc')
