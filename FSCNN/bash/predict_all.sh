#! /bin/bash
target="/home/rassman/bone2gene/masking/output/masks"
for filename in ACh  healthy_magd  HyCh  Noonan  PsHPT  SGA_Magd  shox_magdeburg  SRS_Magd  uts_leipzig  uts_magdeburg; do
    mkdir "$target/$filename"
    python ../predict.py --input /home/rassman/bone2gene/data/annotated/$filename \
      --use_gpu --output_dir "$target/$filename" --checkpoint /home/rassman/bone2gene/masking/output/best_model.ckpt
done
