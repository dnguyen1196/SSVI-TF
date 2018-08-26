#!/usr/bin/bash
#module load python/3.5.0


opts="-p batch -c 2 --mem=16000 --time=4-00:00:00 --mail-type=ALL --mail-user=$USER"

read outdir # output directory
echo "Output directory: $outdir"

script="./venv/bin/python -u factorize.py --diag -f ../tensor-data/GDELT2011/GDELT.2011.REDUCED.cleaned.txt -d count"

while IFS=' ' read -r model
do
    outs="--output=$outdir$model.out --error=$outdir$model.err"

    sbatch $opts $outs --wrap="$script -m $model"

    sleep 1
done
