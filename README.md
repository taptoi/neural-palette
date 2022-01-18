# neural-palette
Custom implementation of PaletteNet in PyTorch

*PaletteNet: Image Recolorization with Given Color Palette (Cho et. al 2017)
*PaletteNet takes two inputs: a source image to be recolored and a target palette. PaletteNet is then designed to change the color concept of a source image so that the palette of the output image is close to the target palette*

## Changes from original
- Instance normalization in decoder path seemed to cause artifacts and was removed
- Transpose-convolutions replaced with upsampling convolutions to reduce checkerboard artifacts as shown in https://distill.pub/2016/deconv-checkerboard/
- No adversarial training yet (todo)

train.py: Train the network
pre_process.py: Pre-process raw input data into original and augmented image-palette pairs and save in compressed npz format
evaluate.py: Perform the inference given a source image and a target color-palette image

## Example
**Input image:**

<img src="https://github.com/taptoi/neural-palette/blob/main/eval_in.png?raw=true" width="800">

**Target palette:**

<img src="https://github.com/taptoi/neural-palette/blob/main/eval_pal.png?raw=true">

**Recolored image:**

<img src="https://github.com/taptoi/neural-palette/blob/main/eval_out.png?raw=true" width="800">
