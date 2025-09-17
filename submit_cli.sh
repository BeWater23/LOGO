#!/bin/bash

# We assume running this from the script directory

job_directory=$(pwd)
script=${*}
name=${1}
output="${job_directory}/${name}.out"
error="${job_directory}/${name}.err"

echo "#!/bin/sh
#SBATCH --job-name=${name}
#SBATCH --mem=90gb
#SBATCH --cpus-per-task=32
#SBATCH -o ${output}
#SBATCH -e ${error}
#SBATCH --partition=notchpeak-shared
#SBATCH --account=sigman
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
hostname
PYTHONUNBUFFERED=1 ${script}
exit 0" > ${name}.job

sbatch ${name}.job


