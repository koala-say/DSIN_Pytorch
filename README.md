# DSIN: Hyperfunction-based Implicit Neural Compression For Any Medical Image

Welcome to **DSIN**!  
This repository provides the official implementation of our paper *"DSIN: Dynamic System-empowered Implicit Neural Representation for Compressing Any Medical Image"*. It is designed to help researchers and developers quickly understand, reproduce, and extend our latest work on neural medical image compression.  


## System requirements
- Linux: Ubuntu 16.04
- Graphics: Nvidia GPU (RTX 4090 recommended)
- Python: 3.10
- Anaconda3
  
## Installation guide

### 1. Download project
``` 
https://github.com/koala-say/DSIN_Pytorch.git
cd DSIN_Pytorch
```

### 2. Create conda environment
``` 
conda create -n dsin python=3.10 -y
conda activate dsin
```

### 3. Install PyTorch (CUDA 11.8 recommended)
```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install dependencies
```
pip install -r requirements.txt
```

## Usage Guide
### 1. INR Compression and DeCompression
 
``` python INR_inference.py -dataset file_path```
- The compressed results are saved in: output/unify/${dataset}/${params}_train/model_train_best.pth

``` python INR_inference.py -weigth model.pth```
- The decompressed files are saved in: output/unify/${dataset}/${params}_eval/visualize

### 2. HyperFunction Compression
``` python hyperF_inference.py -p model_train_best.pth ```
- This command executes the inference script hyperF_inference.py with the trained model checkpoint model_train_best.pth
- The compressed result will be located in : compressed_result/${paramters}/Compressed_Dir
- The decompressed data will be located in : compressed_result/${paramters}/Back_Params.pth





## Contact
If you need any help or are looking for cooperation feel free to contact us. 1284897384@qq.com


## License
This project is covered under the MIT License.
