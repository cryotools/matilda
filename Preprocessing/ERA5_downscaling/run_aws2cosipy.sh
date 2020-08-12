#!/usr/bin/env bash
# Run aws2python routine on input_cosipy.csv and static.nc to produce input_cosipy.nc
sudo python3 /home/ana/Seafile/SHK/Scripts/cosipy/utilities/aws2cosipy/aws2cosipy.py -c /home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/20200810_Umrumqi_ERA5_2000_2019.csv -o /home/ana/Seafile/Ana-Lena_Phillip/input_output/input/20200810_Umrumqi_ERA5_2000_2019_cosipy.nc -s /home/ana/Seafile/Ana-Lena_Phillip/data/input_output/static/Urumqi_static.nc


