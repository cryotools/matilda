#!/bin/bash
longitude_range=75.85,76.06
latitude_range=40.94,41.17

ncdf_folder=/data/projects/ensembles/era5_land/ERA5Land_HighMountainAsia/nc/
destination_folder=/data/projects/ebaca/data/input_output/input/ERA5/no182
destination_name=no182
dataset_name=ERA5_Land_1981_2019.nc
underscore=_

mkdir $destination_folder/variables
for field in "d2m" "sf" "sp" "ssrd" "strd" "t2m" "tp" "u10" "v10" ; do
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

for field in "d2m" "sf" "sp" "ssrd" "strd" "t2m" "tp" "u10" "v10" ; do
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
