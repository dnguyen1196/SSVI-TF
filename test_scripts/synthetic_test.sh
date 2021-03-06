#!/usr/bin/bash

opts="-p batch  -c 2 --mem=4196 --time=4-00:00:00 --mail-type=ALL --mail-user=$USER"

read outdir # output directory

echo "Output directory: $outdir"

# NOTE!!!!
# Depending on the type of experiments change this
#script="./venv/bin/python -u test.py -tr 0.05"
script="./venv/bin/python -u mini_test.py --rand -re 100 -meta 1 -ceta 1 -dim 50 50 50 -it 12000 -tr 0.05 -k1 32 --diag"
# NOTE: TRAIN SIZE
# NOTE; --matrix
# NOTE: --fixed cov
# NOTE: --diag
# NOTE: dimension

while IFS=' ' read -r data model noise
do
    outs="--output=$outdir$data-$model-$noise.out --error=$outdir$data-$model-$noise.err"

    sbatch $opts $outs --wrap="$script -d $data -m $model -n $noise"

    sleep 1
done
