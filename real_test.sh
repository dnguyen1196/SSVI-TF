#!/usr/bin/bash
module load python/3.5.0

opts="-p batch -c 2 --mem=16000 --time=4-00:00:00 --mail-type=ALL --mail-user=$USER"

read outdir # output directory
read filename # Filename
read datatype # datatype
echo "Output directory: $outdir"
echo "Filename: $filename"

script="python -u real_test.py -it 75000 -re 5000 --rank 20 -f Baseline/Data/$filename -d $datatype"

while IFS=' ' read -r model
do
    outs="--output=$outdir$model.out --error=$outdir$model.err"
    sbatch $opts $outs --wrap="$script -m $model"
    sleep 1
done
