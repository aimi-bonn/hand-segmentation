#! /bin/bash

for filename in ACh  healthy_magd  HyCh  Noonan  PsHPT  SGA_Magd  shox_magdeburg  SRS_Magd  uts_leipzig  uts_magdeburg; do
    mkdir "/home/rassman/bone2gene/data/masks/unet/$filename"
    python predict.py --input /home/rassman/bone2gene/data/annotated/$filename --use_gpu --output_dir "/home/rassman/bone2gene/data/masks/unet/$filename"
done
