#!/bin/bash
module load python/3.5.0

opts="-p batch -c 1 --mem=4000 --time=1-12:00:00 --mail-type=ALL --mail-user=$USER"

read outdir # output directory
read diag   # diagonal formulation

echo "Output directory: $outdir"
echo "Using diagonal covariance? $diag"

while IFS=' ' read -r data model
do
    outs="--output=$outdir$data-$model.out --error=$outdir$data-$model.err "
    if ["$diag"=="true"]
    then
        sbatch $opts $outs --wrap="python learning_curve.py -d $data -m $model --diag"
    else
        sbatch $opts $outs --wrap="python learning_curve.py -d $data -m $model"
    fi
    sleep 1
done