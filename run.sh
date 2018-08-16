#!/usr/bin/bash
load python/3.5.0

opts="-p batch  -c 2 --mem=4196 --time=4-00:00:00 --mail-type=ALL --mail-user=$USER"

read outdir # output directory
read diag   # diagonal formulation
read noise  # noise level
read ratio  # using ratio

echo "Output directory: $outdir"
echo "Using diagonal covariance? $diag"
echo "Noise level: $noise"
echo "Using noise as ratio?: $ratio"

while IFS=' ' read -r data model
do
    outs="--output=$outdir$data-$model.out --error=$outdir$data-$model.err "
    choose_diag=""

    if [ "$diag" == "true" ]
    then choose_diag="--diag"
    fi

    choose_ratio=""
    if [ "$ratio" == "true" ]
    then choose_ratio="--ratio"
    fi

    #wrapper="python learning_curve.py -d $data -m $model -n $noise $choose_diag $choose_ratio"
    #echo "$wrapper"
    #sbatch $opts $outs --wrap="$wrapper"
    sbatch $opts $outs --wrap="python -u learning_curve.py -d $data -m $model -n $noise $choose_diag $choose_ratio -o $outdir"

    sleep 1
done
