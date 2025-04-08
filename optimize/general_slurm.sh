#!/bin/bash
#SBATCH -t 0-23:58
#SBATCH -A def-dkrass
#SBATCH --mem 55000
source /home/eliransc/projects/def-dkrass/eliransc/queues/bin/activate
python /home/eliransc/projects/def-dkrass/eliransc/one.deep.moment/optimize/multi_general.py