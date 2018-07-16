#!/bin/bash

opts="-p batch -c 8 --mem=10000 --time=10:00:00 --mail-type=ALL --mail-user=$USER"

IFS=' ' read -r data model

outs="--output=$data-$model.out --error=$data-$model.err "
sbatch $opts $outs --wrap='python learning_curve.py -d $data -m $model'