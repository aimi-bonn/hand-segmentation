# RSNA Bone Age - Artifact and Confounder removal by Hand Segmentation

This repository contains three Deep Learning approaches segment hands in the [RSNA Pediatric Bone Age Dataset](https://www.kaggle.com/datasets/kmader/rsna-bone-age). It is intended to remove potential artifacts from scanning or department-specific which could disturb or bias downstream analysis or learning. In all models, an array of data augmentations was employed to cope with different challenges such as white border from scanning, boxes, and gradients, as well as inverted intensities, etc. 

On a manually crafted test set within the RSNA training set, we achieve a DICE similarity score of $>0.99$.
The models were also qualitatively validated on the Los Angeles Digital Hand Atlas and private data sets: 

<img src="figs/examples.png" width="600" height="600" />

## FastSurferCNN

The main model is a semantic segmentation model based on [*FastSurferCNN*](https://github.com/Deep-MI/FastSurfer) [(Henschel et al., 2020)](https://www.sciencedirect.com/science/article/pii/S1053811920304985).

![FSCNN drawing](figs/fscnn_fig.png)

The model is rather lightweight and, therefore, can run without GPU acceleration in almost real-time. The model was trained based on the predictions of the other models. 


### Test models:

```bash
python FSCNN/predict.py \
    --checkpoint=/path/to/checkpoint.ckpt \
    --input=/path/to/input/ \
    --output=/path/to/target \
    --input_size=512 \
    --use_gpu
```

Hereby, the `input` can be either a whole directory containing the files or a single file. 


### Train / fine tune:

```bash
python FSCNN/train_model.py \
    --train_path=/path/to/train/dataset \
    --val_path=/path/to/val/dataset \
    --size=512
```

The model training can be configured using the YML files in `FSCNN/configs`. Note, that the model will generate pre-computed/cached files containing the loss weights. Input images are expected to be encoded as RGBA, whereby the Alpha channel is the target mask and color information is ignored. 

## Efficient-UNet

Here, another semantic segmentation [*Efficient-UNet*](https://github.com/pranshu97/effunet) model was used.

Usage:


```bash
python UNet/predict.py \
    --model=/path/to/checkpoint.ckpt \
    --input=/path/to/input/ \
    --output=/path/to/target 
    --input_size=512 \
    --use_gpu
```

## Tensormask

Here, an instance segmentation model implemented in [*Detectron2*](https://github.com/facebookresearch/detectron2/blob/main/projects/TensorMask/README.md) was used. Models were trained in Colab, so the requirements are specified there.


# Citation

tba
