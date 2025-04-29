# FUSAR-Map
This repo holds code for [Object-level Semantic Segmentation on The High-Resolution Gaofen-3 FUSAR-Map Dataset](https:)

## Usage

### 1. Download pre-trained weights (2020 Gaofen Challenge & FUSAR-Map)
* [Get weights of 2020 Gaofen Challenge in this link](https://pan.baidu.com/s/1vO6Tr8eZTd9ICGKA9imUcQ)
extraction code：fsjy
* [Get weights of FUSAR-Map in this link](https://pan.baidu.com/s/1c9Pd_6UuwImG_s3jX04gEQ) 
extraction code：：elif 

### 2. Prepare data

Please go to ["FUSAR-Map"](https://shixianzheng.github.io/FUSAR-Map/) for details.

Google Drive download link of FUSAR-MAP:

["FUSAR-Map google drive"](https://drive.google.com/file/d/1dfr2YRFppjoPZi3KKlnd-m6ko-o20sNM/view?usp=sharing)

### 3. Environment

Please prepare an environment with python==3.6, tensorflow-gpu==1.15.

### 4. Train/Test

- Run the train script on your training dataset.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

- Run the test script (UNet, SegNet, DeepLabv3+ et. al)on your test dataset.

## Reference
* []()

## Citations

```bibtex
@ARTICLE{9369836,  author={X. {Shi} and S. {Fu} and J. {Chen} and F. {Wang} and F. {Xu}},  
    journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},   
    title={Object-level Semantic Segmentation on the High-Resolution Gaofen-3 FUSAR-Map Dataset},   
    year={2021},  
    volume={},  
    number={},  
    pages={1-1},  
    doi={10.1109/JSTARS.2021.3063797}}
```
