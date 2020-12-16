for field in \
"d2m-2m_dewpoint_temperature" \
"sf-snowfall" \
"sp-surface_pressure" \
"ssrd-surface_solar_radiation_downwards" \
"strd-surface_thermal_radiation_downwards" \
"t2m-2m_temperature" \
"tcc-total_cloud_cover" \
"tp-total_precipitation" \
"u10-10m_u_component_of_wind" \
"v10-10m_v_component_of_wind" \
; do
    PART=(${field//-/ })
    mkdir "/data/projects/prime-SG/io/Halji/ERA5/grib/${PART[0]}"
    for year in `seq 1979 2019`; do
        echo "Downloading field $field year $year"
        sed -e "s/YYYY/$year/g" -e "s/LONG_NAME/${PART[1]}/g" -e "s/NAME/${PART[0]}/g" Halji_template.py > era5-${PART[0]}.py
        python era5-${PART[0]}.py
        done
    done
