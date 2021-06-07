#!/bin/bash
longitude_range=75.80,76.40
latitude_range=41.00,41.40
# longitude_range=78.00,78.30
# latitude_range=42.00,42.40

ncdf_folder=/data/projects/ensembles/era5_land/ERA5-Land_HMA/nc/
dataset_name=ERA5L_1982_2020.nc
underscore=_

destination_folder=/data/projects/ebaca/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/new_grib_conversion
destination_name=no182
# destination_folder=/data/projects/ebaca/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/Kysylsuu
# destination_name=kysylsuu


# "d2m" "sf" "sp" "ssrd" "strd" "t2m" "tp" "u10" "v10"

mkdir $destination_folder/variables
for field in "t2m" "tp" ; do
    echo $field
    output_folder=$destination_folder/variables/$field
    mkdir $output_folder
    input_folder=$ncdf_folder$field
    cd $input_folder
    for file in *.nc ; do
        output_file=$output_folder/$destination_name_$file
        echo $file
        echo $output_file
        module load cdo
        command="cdo sellonlatbox,$longitude_range,$latitude_range $file $output_file"
        echo $command
        $command
    done
done

for field in "t2m" "tp" ; do
    input_folder=$destination_folder/variables/$field
    cd $input_folder
    output_timemerge=$destination_folder/variables/$field$underscore$dataset_name
    pwd
    ls -l
    echo $output_timemerge
    module load cdo
    cdo -b F64 mergetime "*.nc" $output_timemerge
done

cd $destination_folder/variables
output_file=$destination_folder$dataset_name
pwd
echo $output_file
module load cdo
pwd
ls -l
cdo merge *.nc $output_file


