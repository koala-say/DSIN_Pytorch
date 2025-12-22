# DSIN: Hyperfunction-based Implicit Neural Compression For Any Medical Image

Welcome to **DSIN**!  
This repository provides the official implementation of our paper *"DSIN: Dynamic System-empowered Implicit Neural Representation for Compressing Any Medical Image"*. It is designed to help researchers and developers quickly understand, reproduce, and extend our latest work on neural medical image compression.  


## System requirements
- Graphics: Nvidia GPU (RTX 4090 recommended)
- Anaconda3
- PyTorch
  
## Installation guide

### 1. Download project
``` https://github.com/koala-say/DSIN_Pytorch.git ```

### 2. Prepare the Conda enviroments

``` 
cd DSIN_PyTorch
conda create -n dsin python=3.9
conda activate dsin
cd DSIN_Pytorch
pip install -r requirements.txt
```

## Usage Guide
### 1. INR Compression

``` python INR_inference.py ```
- The compressed results are saved in: output/unify/${dataset}/${params}_train/model_train_best.pth
- The decompressed files are saved in: output/unify/${dataset}/${params}_eval/visualize

### 2. HyperFunction Compression
``` python hyperF_inference.py -p model_train_best.pth ```
- This command executes the inference script hyperF_inference.py with the trained model checkpoint model_train_best.pth
- The compressed result will be located in : compressed_result/${paramters}/Compressed_Dir
- The decompressed data will be located in : compressed_result/${paramters}/Back_Params.pth



## Contact
If you need any help or are looking for cooperation feel free to contact us. 1284897384@qq.com


