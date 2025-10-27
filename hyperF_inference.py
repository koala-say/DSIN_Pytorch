# Use a pipeline as a high-level helper
#from transformers import pipeline
import numpy as np
import torch
import os
from lib.Compress_Params_Standard_uint_i_ClassCenterHyper_Multi_VersionC_matrix_CP_best import *
import contextlib
import shutil
import time
from optparse import OptionParser
import numpy as np


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=1, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=8,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    parser.add_option('-v', '--val_percent', dest='val_percent', type='float',
                      default=0.2, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':

    model_name = 'E-NERV'
    rect_l = 0.2
    num_inner_list = [22500]
    class_max = 3
    loss_max = 0.001
    loss_hope = 0.001
    num_cores = 10  # num_cpu
    mode = 'encode' # 'encode' or 'decode'
    model_path = "model_train_best.pth"
    ### NERV网络
    from utils.model import CustomDataSet, Generator
    model = Generator(embed_length=160, stem_dim_num='1024_1', fc_hw_dim='8_8_64', expansion=8, 
        num_blocks=1, norm='none', act='leaky01', bias = True, reduction=2, conv_type='conv',
        stride_list=[4, 4, 2, 2, 2],  sin_res=False,  lower_width=32, sigmoid=False)
    dict = torch.load(model_path)['state_dict']
    model.load_state_dict(dict)

    # encoding save path
    Save_Param_Path = f'./compressed_result/{model_name}_l_{str(rect_l)[0] + str(rect_l)[2:]}/'

    # decoding files path
    Decode_Param_Path = f'./compressed_result/{model_name}_l_{str(rect_l)[0] + str(rect_l)[2:]}'

    if mode == 'encode':

        # 若存在，则删除重建
        if os.path.exists(Save_Param_Path):
            shutil.rmtree(Save_Param_Path)
            os.makedirs(Save_Param_Path, exist_ok=True)
        else:
            os.makedirs(Save_Param_Path, exist_ok=True)


        # 原参数保存地址
        Save_OriParam_Path = Save_Param_Path + 'Origin_Params.pth'
        # SpeedUp + FineTuning 后的params压缩结果的文件夹root地址
        Save_CompressedResult_RootPath = Save_Param_Path + 'Compressed_Dir/'
        # 压缩后再还原的params保存地址
        Save_BackParam_Path = Save_Param_Path + 'Back_Params.pth'
        Save_BackParam_Path_bin = Save_Param_Path + 'pytorch_model_back.bin'


        ## 保存 model的原参数 ##
        torch.save(model.state_dict(), Save_OriParam_Path)

        t1_start = time.perf_counter()
        ### model ：参数复原后的model ###
        size_result, model = compress_params(model, Save_CompressedResult_RootPath, rect_l, num_inner_list, class_max,
                                             loss_max, loss_hope, num_cores)  # 返回 压缩后的文件大小
        t1_end = time.perf_counter()


        # 保存"Back_Params.pth"和"pytorch_model.bin"
        torch.save(model.state_dict(), Save_BackParam_Path)
        torch.save(model.state_dict(), Save_BackParam_Path_bin)


        print(f"原参数大小为 {os.path.getsize(Save_OriParam_Path)}字节")
        print(f"压缩结果的大小为 {size_result}字节")
        print(f"压缩倍数{os.path.getsize(Save_OriParam_Path) / size_result}倍")

        # 将模型的结构保存到 txt 文件
        file_path = Save_Param_Path + 'model_structure.txt'
        with open(file_path, 'w') as f:
            with contextlib.redirect_stdout(f):
                print(model)
        print("Encoding Finished!!!!!!!")
        print(f"Compression Time : {t1_end-t1_start}秒 = {(t1_end-t1_start)/60}分钟")


    elif mode == 'decode':
        t_start = time.perf_counter()
        num_inner_list = np.fromfile(Decode_Param_Path + '/Compressed_Dir/num_inner_list.bin', dtype=np.uint64)
        rect_l_str = Decode_Param_Path[find_nth_occurrence(Decode_Param_Path, "/", 2):].split('_')[2]
        rect_l = float(rect_l_str[:1] + '.' + rect_l_str[1:])
        decode_params_list, time_name, time_list, t0, t1 = decompress_params(Decode_Param_Path + '/Compressed_Dir/',
                                                                                     rect_l,
                                                                                     num_inner_list)
        t_end = time.perf_counter()
        print(f"{t_end-t_start}秒")





