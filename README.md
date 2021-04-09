# Coordinate-based texture inpainting

This is repository with inference code for paper **"Coordinate-based texture inpainting for pose-guided human image generation"**.

## Data
To use this repository you first need to download following files

1. Download model checkpoints from **TODO**. It consists of two files `inpainter.pth` and `refiner.pth`. 
They need to be placed under `data/checkpoint/` directory.
2. Download `smpltexmap.npy` file from **TODO** and put under `data/` directory. 
It is required to convert uv renders produced by [densepose](http://densepose.org/) algorithm (`*_IUV.png` files) to SMPL format used by our model.

## Usage   
Two simple ways to run the code are:

- Use [demo.ipynb](demo.ipynb) notebook
- Run `convert_uv_render.py` and `infer_sample.py` scripts.

This repository contains some examples of input data (rgb images and UV renders) 
from [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) dataset. 
You can run the scripts above with these samples. 

The scripts are for demonstrational purposes and are not suited to process multiple files, so you'll probably need to write your own processing loop using functions from this repo.

#### Usage examples
`convert_uv_render.py` converts densepose IUV renders into smpl format. 
It takes output of densepose method (`*_IUV.png` file) from `data/samples/source_iuv` directory and
saves resulting uv render to `data/samples/source_uv` as a `.npy` file.


Usage example:
```
python convert_uv_render.py --sample_id=WOMEN#Blouses_Shirts#id_00000442#01_2_side
```  

`infer_sample.py` takes a source rgb image and a target uv render to produce an image of source person in target pose. 
Rusulting images are saved to `data/results`.

```
python infer_sample.py --source_sample=WOMEN#Blouses_Shirts#id_00000442#01_2_side --target_sample=WOMEN#Dresses#id_00000106#03_1_front
```

## Citation
This repository contains code corresponding to:

A. Grigorev, A. Sevastopolsky, A. Vakhitov, and V. Lempitsky.
**Coordinate-based texture inpainting for pose-guided human image generation**. In
*IEEE Conference on Computer Vision and Pattern Recognition*, 2019.

Please cite as:

```
@inproceedings{grigorev2019coordinate,
  title={Coordinate-based texture inpainting for pose-guided human image generation},
  author={Grigorev, Artur and Sevastopolsky, Artem and Vakhitov, Alexander and Lempitsky, Victor},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12135--12144},
  year={2019}
}
```