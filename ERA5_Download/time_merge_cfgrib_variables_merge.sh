working_folder=/data/projects/prime-SG/io/Halji/ERA5_land/
grib_folder=$working_folder/grib/
nc_folder=$working_folder/nc/
dataset="_Halji_ERA5_1980_2019.nc"
output_file="20200401_Halji_ERA5_1980_2019.nc"

for field in "d2m" "sf" "sp" "ssrd" "strd" "t2m" "tp" "u10" "v10" ; do
    echo $field
    mkdir $nc_folder$field
    echo $nc_folder$field
    input_folder=$grib_folder$field/
    cd $input_folder
    for file in *.grib ; do
        PART=(${file//./ })
        export PATH="/nfsdata/programs/anaconda3_201910/bin:$PATH"
        output_cfgrib=$nc_folder$field/${PART[0]}.nc
        echo $file
        echo $output_cfgrib
        cfgrib to_netcdf -c CDS $file -o $output_cfgrib
    done
done

for field in "d2m" "sf" "sp" "ssrd" "strd" "t2m" "tp" "u10" "v10" ; do
    input_folder=$nc_folder$field/
    cd $input_folder
    output_timemerge=$nc_folder$field$dataset
    ls
    echo $output_timemerge
    cdo -b F64 mergetime "*.nc" $output_timemerge
done

cd $nc_folder
output=$working_folder$output_file
cdo merge *.nc $output
