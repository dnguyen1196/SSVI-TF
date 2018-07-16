#!/bin/bash
module load python/3.5.0

opts="-p batch -c 1 --mem=4000 --time=10:00:00 --mail-type=ALL --mail-user=$USER"
outdir="/cluster/home/mnguye16/SSVI/SSVI-TF/learning_results/"

IFS=' ' read -r data model

outs="--output=$outdir$data-$model.out --error=$outdir$data-$model.err "
sbatch $opts $outs --wrap="python learning_curve.py -d $data -m $model"