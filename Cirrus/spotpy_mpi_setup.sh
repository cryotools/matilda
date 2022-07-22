## Setup script to initiate statistical parameter optimization runs on an HPC using slurm. To change forcing data edit input paths in spotpy_matilda_template.py

jobname=MATILDA_BENCHMARK
reps=500
node_count=7		# Max. 10 on computehm + max. 7 on compute
core_count=20		# Max. 20
algorithm=demcz	# mc, lhs, fast, rope, sceua or demcz

set_up_start=1982
set_up_end=1984
sim_start=1985
sim_end=1989
freq=D			# D, W, M or Y
partition=compute	# compute or computehm

dbformat=csv		# csv, sql or None
save_sim=False
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
sed -e "s/REPETITIONS/$reps/" -e "s/ALGORITHM/$algorithm/" -e "s/CORES/$core_count/" -e "s/DB_NAME/$jobname/" -e "s+OUTPATH+$working_dir+" -e "s/SETUPSTART/$set_up_start/" -e "s/SETUPEND/$set_up_end/" -e "s/SIMSTART/$sim_start/" -e "s/SIMEND/$sim_end/" -e "s/FREQ/$freq/" -e "s/DBFORMAT/$dbformat/" -e "s/SAVESIM/$save_sim/" spotpy_matilda_template.py > $mspot_script

# Construct sbatch script
sed -e "s/jobname/$jobname/g" -e "s/node_count/$node_count/g" -e "s/core_count/$core_count/g" -e "s/error_log/$error_log/g" -e "s/out_log/$out_log/g" -e "s+working_dir+$working_dir+g" -e "s+mspot_script+$mspot_script+g" -e "s/PARTITION/$partition/g"  spotpy_mpirun_template.bat > $log_dir/spotpy_mpirun_${jobname}.bat

# Start slurm job
chmod +x $log_dir/spotpy_mpirun_${jobname}.bat
sbatch $log_dir/spotpy_mpirun_${jobname}.bat

