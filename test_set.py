import os
import shutil

# 定义路径
source_dir = '/home/data/czp/ABAW/data/zzr_video'
txt_file_path = '/home/data/zhangzr22/abaw/ABAW6/tmp.txt'
target_dir = '/home/data/lrd/zgp/abaw/AU_test/raw_video'

# 确保目标目录存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 读取txt文件，并处理前缀，去除_left或_right
prefixes = set()
with open(txt_file_path, 'r') as file:
    for line in file:
        prefix = line.strip()
        # 去除后缀_left和_right
        for suffix in ('_left', '_right'):
            if prefix.endswith(suffix):
                prefix = prefix[:-len(suffix)]
                break
        prefixes.add(prefix)

# 记录复制的文件数
i = 0
# 遍历去重和处理后的前缀集
for prefix in prefixes:
    matched = False
    # 遍历源目录中的所有文件
    for file_name in os.listdir(source_dir):
        # 使用os.path.splitext去除文件扩展名，仅保留文件名部分用于匹配
        file_name_without_ext, _ = os.path.splitext(file_name)
        
        # 如果去除扩展名后的文件名与前缀完全匹配
        if file_name_without_ext == prefix:
            source_file_path = os.path.join(source_dir, file_name)
            target_file_path = os.path.join(target_dir, file_name)
            
            # 检查是否为文件而非目录
            if os.path.isfile(source_file_path):
                # 复制文件
                shutil.copy(source_file_path, target_file_path)
                print(f"复制文件：{source_file_path} 到 {target_file_path}")
                i += 1
                matched = True
                break  # 找到匹配后跳出内层循环
    if not matched:
        print(f"未找到与前缀 '{prefix}' 匹配的文件")

print(f"总共复制了 {i} 个文件。")


# import os
# import shutil

# import os
# import shutil

# # 定义路径
# source_dir = '/home/data/czp/ABAW/data/zzr_video'
# txt_file_path = '/home/data/zhangzr22/abaw/ABAW6/tmp.txt'  # 已更正变量名
# target_dir = '/home/data/lrd/zgp/abaw/AU_test/raw_video'

# i = 0
# with open(txt_file_path, 'r') as file:
#     for line in file:
#         prefix = line.strip()  # 获取前缀
#         matched = False  # 标记是否找到匹配的文件

#         # 遍历源目录中的所有文件
#         for file_name in os.listdir(source_dir):
#             # 使用os.path.splitext去除文件扩展名，仅保留文件名部分用于匹配
#             file_name_without_ext, _ = os.path.splitext(file_name)
            
#             # 如果去除扩展名后的文件名与前缀完全匹配
#             if file_name_without_ext == prefix:
#                 source_file_path = os.path.join(source_dir, file_name)
#                 target_file_path = os.path.join(target_dir, file_name)
                
#                 # 检查是否为文件而非目录
#                 if os.path.isfile(source_file_path):
#                     # 复制文件
#                     shutil.copy(source_file_path, target_file_path)
#                     print(f"复制文件：{source_file_path} 到 {target_file_path}")
#                     i += 1
#                     matched = True
#                     break  # 找到匹配后跳出内层循环
#         if not matched:
#             print(f"未找到与前缀 '{prefix}' 匹配的文件")

# # 定义路径
# source_dir = '/home/data/czp/ABAW/data/zzr_video'
# txt_dir = '/home/data/zhangzr22/abaw/ABAW6/tmp.txt'  # 请替换为你的txt文件夹的真实路径
# target_dir = '/home/data/lrd/zgp/abaw/AU_test/raw_video'
# i = 0
# # 读取txt文件并遍历每一行
# with open(txt_dir, 'r') as file:
#     for line in file:
#         prefix = line.strip()  # 获取前缀
#         # 遍历源目录中的所有文件
#         for file_name in os.listdir(source_dir):
#             # 如果文件名以txt文件中指定的前缀开始
#             if file_name.startswith(prefix):
#                 source_file_path = os.path.join(source_dir, file_name)
#                 target_file_path = os.path.join(target_dir, file_name)
                
#                 # 检查是否为文件而非目录
#                 if os.path.isfile(source_file_path):
#                     # 复制文件
#                     i=i+1
#                     shutil.copy(source_file_path, target_file_path)
#                     print(f"复制文件：{source_file_path} 到 {target_file_path}")
#                     print(i)
#                 else:
#                     print(f"跳过目录：{source_file_path}")

# # 定义路径
# source_dir = '/home/data/czp/ABAW'
# txt_dir = '/home/data/zhangzr22/abaw/ABAW6/tmp.txt'  # 请替换为你的txt文件夹的真实路径
# target_dir = '/home/data/lrd/zgp/abaw/AU_test/cropped_aligned'

# # 读取txt文件并遍历每一行
# with open(txt_dir, 'r') as file:
#     for line in file:
#         # 移除行尾的换行符并获取文件夹名
#         folder_name = line.strip()
#         source_folder_path = os.path.join(source_dir, folder_name)

#         # 检查源目录下是否存在与txt文件中指定的文件夹
#         if os.path.exists(source_folder_path) and os.path.isdir(source_folder_path):
#             # 构造目标目录路径
#             target_folder_path = os.path.join(target_dir, folder_name)
            
#             # 复制整个文件夹
#             if not os.path.exists(target_folder_path):
#                 shutil.copytree(source_folder_path, target_folder_path)
#                 print(f"复制文件夹：{source_folder_path} 到 {target_folder_path}")
#             else:
#                 print(f"目标路径已存在文件夹：{target_folder_path}")
#         else:
#             print(f"未找到文件夹：{source_folder_path}")