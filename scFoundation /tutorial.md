## LSF job
    bsub -J "scfoundation" -W 24:00 -q jupyter_gpu -gpu "num=1:mode=exclusive_process:mig=2" -Is bash

## Training
    bash run_sh/run_singlecell_norman.sh
    bash run_sh/run_singlecell_maeautobin-0.1B-res0-norman.sh

