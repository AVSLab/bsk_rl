#!/bin/bash

# Allocation account name
#SBATCH --account=ucb550_asc2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=47:59:59
# Jobs to run, inclusive
#SBATCH --array=0-0
#SBATCH --partition=amilan
#SBATCH --mem=100G
#SBATCH --constraint=epyc-7713
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --output=/scratch/alpine/%u/job_output/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_100d00i_reducedFailurePenalty.out
#SBATCH --mail-type=ALL
#SBATCH --qos=long

# Copy this script as sweep.sh to modify it for your own use.


module purge

echo "Loading modules"
module load python/3.10.2
module load gcc

echo "Activating virtual environment"
source /projects/$USER/.venv/bin/activate
export TMPDIR=/scratch/alpine/$USER/temp_dir

echo "Running training script"
# Uses int(sys.argv[1]) in the script to get the array index
python3 /projects/$USER/bsk_rl/examples/train_Polaris2200.py

echo "== End of Job =="
