#!/usr/bin/env bash
# Run aws2python routine on input_cosipy.csv and static.nc to produce input_cosipy.nc
sudo python3 /home/phillip/PycharmProjects/cosipy_Jan2020/utilities/aws2cosipy/aws2cosipy.py -c /home/phillip/Seafile/Phillip_Anselm/input_output/input/20200129_Umrumqi_ERA5_2016_2018_cosipy.csv -o /home/phillip/Seafile/Phillip_Anselm/input_output/input/20200129_Umrumqi_ERA5_2016_2018_cosipy.nc -s /home/phillip/Seafile/Phillip_Anselm/input_output/static/Urumqi_static.nc


