#!/bin/bash

job_directory=$(pwd)
script=${*}
name=${1}

echo "#!/bin/sh
#SBATCH --job-name=${name}
#SBATCH --mem=90gb
#SBATCH --cpus-per-task=32
#SBATCH -o ${job_directory}/${name}_%j.out
#SBATCH -e ${job_directory}/${name}_%j.err
#SBATCH --partition=notchpeak-shared
#SBATCH --account=sigman
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
hostname
PYTHONUNBUFFERED=1 ${script}
exit 0" > ${name}.job

sbatch ${name}.job


