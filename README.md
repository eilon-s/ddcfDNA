# cfDNA1G

## installing cfDNA1G

* install miniconda3 (https://conda.io/miniconda.html)
* ~/miniconda3/bin/conda config --add channels r
* ~/miniconda3/bin/conda config --add channels bioconda
* ~/miniconda3/bin/conda create --name cfDNA -c bioconda python=3.5 --file workflow/requirements3.txt       
* install autograd using pip: 
source ~/miniconda3/bin/activate cfDNA; 
pip install --user autograd;

*clone this github.

## Running inference executable (the executable will run only on 64bit linux machines):

Before running the executable, you will need to set LD_LIBRARY_PATH to point to your miniconda installation using:
export LD_LIBRARY_PATH=/home/<YOUR USER NAME>/miniconda3/envs/cfDNA/lib
export PYTHONHOME=/home/<YOUR USER NAME>/miniconda3/envs/cfDNA
You need to run these lines before using the executable.
After running the executable return these variable values to their original values
(usually using 
export LD_LIBRARY_PATH=;
and 
export PYTHONHOME=;)

so a run shell may look like that:

source ~/miniconda3/bin/activate cfDNA;
export LD_LIBRARY_PATH=/home/<YOUR USER NAME>/miniconda3/envs/cfDNA/lib
export PYTHONHOME=/home/<YOUR USER NAME>/miniconda3/envs/cfDNA
cfDNA1G/python/cfDNA_infer_donor_fraction.exe --help;
export LD_LIBRARY_PATH=;
export PYTHONHOME=;

## installing WASP

* download the scripts in the mapping directory from WASP github https://github.com/gmcvicker/WASP
* copy the scripts from WASP mapping to {cfDNA scripts path}/WASP 
* install anaconda2 from https://www.continuum.io/downloads
* create anaconda environment for wasp (this is a different environment for cfDNA since it is python 2.7)
  * ~/anaconda2/bin/conda config --add channels r
  * ~/anaconda2/bin/conda config --add channels bioconda
  * ~/anaconda2/bin/conda create -n wasp python=2.7 --file workflow/requirements2.txt

## Full workflow: creates input file for the inference algorithm, runs the inference for each sample and collect the results
See README under the workflow directory

## Running only the inference step
1. Filter genotype file (python3 script. Dependencies: pandas,scipy):
  python cfDNA_filter_SNPs.py input_filename output_filename -f 1e-5 -s 100 -r 100 -g 0.7
2. Infer donor-cfDNA fraction (python3 script. Dependencies: pandas,scipy and autograd): 
  python cfDNA_infer_donor_fraction.py input_filt_filename output_filename -s sample_id
  For more details see: 
  python ./code/cfDNA_infer_donor_fraction.py --help
3. Collect the results:
  cfDNA_collect_inference_results.py




