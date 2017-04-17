echo "---start running ----"

# ,owners,normal

snakemake --keep-going --configfile cfDNA_config_run_V15.json -j 500 --latency-wait 60 --cluster "sbatch -p pritch,hbfraser --job-name {params.job_name} -o {params.job_out_dir}/{params.job_out_file}.out -e {params.job_out_dir}/{params.job_out_file}.error --time {params.run_time} --cpus-per-task {params.cores} --mem {params.memory}000"

# --keep-going 

echo "---end running ----"
