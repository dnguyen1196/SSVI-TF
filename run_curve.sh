#!/usr/bin/bash
#venv/bin/python learning_curve.py -d real -m deterministic -n 500 -o learning_results/jul_29_full_hardnoise/ &
#venv/bin/python learning_curve.py -d binary -m deterministic -n 0.5 -o learning_results/jul_29_full_hardnoise/ &
#venv/bin/python learning_curve.py -d count -m deterministic -n 1 -o learning_results/jul_29_full_hardnoise/ &
#venv/bin/python learning_curve.py -d real -m simple -n 500 -o learning_results/jul_29_full_hardnoise/ &
#venv/bin/python learning_curve.py -d binary -m simple -n 0.5 -o learning_results/jul_29_full_hardnoise/ &
#venv/bin/python learning_curve.py -d count -m simple -n 1 -o learning_results/jul_29_full_hardnoise/ &
venv/bin/python learning_curve.py -d real -m robust -n 500 -o learning_results/jul_29_full_hardnoise/ &
#venv/bin/python learning_curve.py -d binary -m robust -n 0.5 -o learning_results/jul_29_full_hardnoise/ &
#venv/bin/python learning_curve.py -d count -m robust -n 1 -o learning_results/jul_29_full_hardnoise/ &
