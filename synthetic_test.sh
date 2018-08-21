#!/usr/bin/bash
module load python/3.5.0

opts="-p batch  -c 2 --mem=4196 --time=4-00:00:00 --mail-type=ALL --mail-user=$USER"

read outdir # output directory

echo "Output directory: $outdir"

script="./venv/bin/python -u test.py --diag --fixed_cov -tr 0.05"

while IFS=' ' read -r data model noise
do
    outs="--output=$outdir$data-$model-$noise.out --error=$outdir$data-$model-$noise.err"

    sbatch $opts $outs --wrap="$script -d $data -m $model -n $noise"

    sleep 1
done
