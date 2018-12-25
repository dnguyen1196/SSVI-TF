#!/bin/bash

opts="-p batch -c 2 --mem=4196 --time=4-00:00:00 --mail-type=ALL --mail-user=$USER"

read outdir # output directory

echo "Output directory: $outdir"
script="./venv/bin/python -u learning_curve.py --rand -dim 50 50 50"

# NOTE: train size
# NOTE: --matrix
# NOTE: diag
# NOTE: dimension

while IFS=' ' read -r data model
do
    outs="--output=$outdir$data-$model.out --error=$outdir$data-$model.err "

    sbatch $opts $outs --wrap="$script -d $data -m $model"

    sleep 1
done
