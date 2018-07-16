#!/bin/bash
#module load python/3.5.0

opts="-p batch -c 1 --mem=4000 --time=10:00:00 --mail-type=ALL --mail-user=$USER"
outdir="/cluster/home/mnguye16/SSVI/SSVI-TF/learning_results/"
#while read filenm; do
#      outs = "--output=$filenm.out --error=$filenm.err "
#      echo "sbatch $opts $outs --wrap='learning_curve.py -d $ -m $'"
#      sleep 1
#done

while IFS=' ' read -r data model
do
    outs="--output=$outdir$data-$model.out --error=$outdir$data-$model.err "
    sbatch $opts $outs --wrap='python3 learning_curve.py -d $data -m $model'
#    echo "sbatch $opts $outs --wrap='python learning_curve.py -d $data -m $model'"
    sleep 1
done