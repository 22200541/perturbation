## LSF job
    bsub -J "scgen_run" -Is -W 24:00 -q normal -gpu "num=1:mode=exclusive_process:mig=2" -n 2 -R "rusage[mem=10000]" bash

### Run
    conda activate scgen
    
    python systema/src/run_scgen.py

    conda activate scgpt

    python systema/src/run_scgpt.py

    conda activate gears

    python systema/src/run_gears.py
