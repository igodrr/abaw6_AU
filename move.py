import os
import pandas as pd
import numpy as np
import glob

# 定义源目录
src_dirs = [
    '/home/data/zhangzr22/abaw/ABAW5/code_AU_EXPR_VA/data/6th_ABAW_Annotations/AU/Train_Set',
    '/home/data/zhangzr22/abaw/ABAW5/code_AU_EXPR_VA/data/6th_ABAW_Annotations/AU/Validation_Set'
]

# 定义目标根目录
dst_root_dir = '/home/data/zhangzr22/abaw/ABAW5/code_AU_EXPR_VA/data/compacted_48'

for src_dir in src_dirs:
    # 遍历每个源目录下的所有.txt文件
    for txt_file in glob.glob(os.path.join(src_dir, '*.txt')):
        # 使用pandas读取.csv文件，跳过第一行
        data = np.genfromtxt(txt_file, delimiter=',', skip_header=1)
        
        # 获取文件名（不带扩展名），用作目标目录名
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        
        # 创建目标目录（如果不存在）
        dst_dir = os.path.join(dst_root_dir, base_name)
        os.makedirs(dst_dir, exist_ok=True)
        
        # 构建目标.npy文件路径
        npy_file_path = os.path.join(dst_dir, 'AU_continuous_label.npy')
        
        # 保存为.npy文件
        np.save(npy_file_path, data)
        print(f'Saved: {npy_file_path}')


# import os
# import numpy as np
# import glob

# # 定义源目录
# src_dirs = [
#     '/home/data/zhangzr22/abaw/ABAW5/code_AU_EXPR_VA/data/6th_ABAW_Annotations/AU/Train_Set',
#     '/home/data/zhangzr22/abaw/ABAW5/code_AU_EXPR_VA/data/6th_ABAW_Annotations/AU/Validation_Set'
# ]

# # 定义目标目录
# dst_dir = '/home/data/zhangzr22/abaw/ABAW5/code_AU_EXPR_VA/data/compacted_48'

# for src_dir in src_dirs:
#     # 遍历每个源目录
#     for txt_file in glob.glob(os.path.join(src_dir, '*.txt')):
#         # 读取.txt文件内容
#         data = np.genfromtxt(txt_file, delimiter=',', skip_header=1)
        
#         # 获取文件夹的名字作为文件名前缀
#         folder_name = os.path.basename(os.path.dirname(txt_file))
#         file_name_prefix = folder_name
        
#         # 构建目标.npy文件名
#         npy_file_name = file_name_prefix + '/AU_continuous_label.npy'
#         npy_file_path = os.path.join(dst_dir, npy_file_name)
        
#         # 保存为.npy文件
#         np.save(npy_file_path, data)
#         print(f'Saved: {npy_file_path}')