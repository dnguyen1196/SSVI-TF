#!/bin/bash
module load python/3.5.0

opts="-p batch -c 1 --mem=4000 --time=1-12:00:00 --mail-type=ALL --mail-user=$USER"
outdir="/cluster/home/mnguye16/SSVI/SSVI-TF/learning_results/diag_cov/"

#while read filenm; do
#      outs = "--output=$filenm.out --error=$filenm.err "
#      echo "sbatch $opts $outs --wrap='learning_curve.py -d $ -m $'"
#      sleep 1
#done

while IFS=' ' read -r data model
do
    outs="--output=$outdir$data-$model.out --error=$outdir$data-$model.err "
    sbatch $opts $outs --wrap="python learning_curve.py -d $data -m $model --diag"
    sleep 1
done