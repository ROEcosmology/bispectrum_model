#!/bin/bash
#SBATCH --job-name=test_Kernel
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=1G
#SBATCH --output=out/%x.%j.out
#SBATCH --error=err/%x.%j.err

cd /home/rneveux/bispectrum/theory/

module load anaconda

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/rneveux/cuba

for i in `seq $1 $(($1 + 9))`
do
for j in `seq 0 7`
do
python test_Kernel.py -cosmo $i -spectrum_part SN -kernel $j -ells 202
done
done
