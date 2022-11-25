## Setup script to initiate statistical parameter optimization runs on an HPC using slurm. To change forcing data edit input paths in spotpy_matilda_template.py

jobname=Test
reps=2100
node_count=10		# Max. 10 on computehm + max. 7 on compute
core_count=20		# Max. 20
algorithm=fast	# mc, lhs, fast, rope, sceua or demcz

df=df_era	# df_har, df_mswx, df_df_era, df
ele_dat=3273	# ERA5L/MSWX: 3273, HARv2: 3172, era_adj: 2550

set_up_start=1997
set_up_end=1999
sim_start=2000
sim_end=2020
freq=D			# D, W, M or Y
partition=compute	# compute or computehm

dbformat=csv		# csv, sql or None
save_sim=True
output_path=/data/projects/ebaca/Ana-Lena_Phillip/data/matilda/Cirrus

# Create standard filenames
timestamp=`date +"%Y-%m-%d_%H-%M-%S"`
error_log=${jobname}_${timestamp}.err
out_log=${jobname}_${timestamp}.out

# Create output directories
working_dir=$output_path/${jobname}_${timestamp}
log_dir=$working_dir/logs
mspot_script=${log_dir}/spotpy_matilda_${jobname}.py
mkdir "$working_dir"
mkdir "$log_dir"

# Construct mspot python script. Use + (or any other separator) instead of / when paths are involved.
sed -e "s/REPETITIONS/$reps/" -e "s/ALGORITHM/$algorithm/" -e "s/CORES/$core_count/" -e "s/DB_NAME/$jobname/" -e "s+OUTPATH+$working_dir+" -e "s/SETUPSTART/$set_up_start/" -e "s/SETUPEND/$set_up_end/" -e "s/SIMSTART/$sim_start/" -e "s/SIMEND/$sim_end/" -e "s/FREQ/$freq/" -e "s/DBFORMAT/$dbformat/" -e "s/SAVESIM/$save_sim/" -e "s/DATAFRAME/$df/" -e "s/ELE_DAT/$ele_dat/" spotpy_matilda_template.py > $mspot_script

# Construct sbatch script
sed -e "s/jobname/$jobname/g" -e "s/node_count/$node_count/g" -e "s/core_count/$core_count/g" -e "s/error_log/$error_log/g" -e "s/out_log/$out_log/g" -e "s+working_dir+$working_dir+g" -e "s+mspot_script+$mspot_script+g" -e "s/PARTITION/$partition/g"  spotpy_mpirun_template.bat > $log_dir/spotpy_mpirun_${jobname}.bat

# Start slurm job
chmod +x $log_dir/spotpy_mpirun_${jobname}.bat
sbatch -W $log_dir/spotpy_mpirun_${jobname}.bat
wait

# Run FAST analyzer
if [ $algorithm==fast ]
then
fast_script=${log_dir}/fast_analyzer_${jobname}.py
sed -e "s+OUTPATH+$working_dir+" -e "s/DB_NAME/$jobname/" FAST_analyzer.py > $fast_script
module load anaconda
/home/susterph/env/bin/python $fast_script
fi


