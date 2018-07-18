#!/bin/bash
module load python/3.5.0

opts="-p batch -c 2 --mem=4000 --time=4-00:00:00 --mail-type=ALL --mail-user=$USER"

read outdir # output directory
read diag   # diagonal formulation
read noise  # noise level

echo "Output directory: $outdir"
echo "Using diagonal covariance? $diag"
echo "Noise level: $noise"

while IFS=' ' read -r data model
do
    outs="--output=$outdir$data-$model.out --error=$outdir$data-$model.err "
    if [ "$diag" == "true" ]
    then
        sbatch $opts $outs --wrap="python learning_curve.py -d $data -m $model -n $noise --diag"
    else
        sbatch $opts $outs --wrap="python learning_curve.py -d $data -m $model -n $noise"
    fi
    sleep 1
done