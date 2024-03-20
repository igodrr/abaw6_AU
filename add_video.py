import os
import numpy as np
from PIL import Image

# 源目录和目标根目录
source_dir = '/home/data/lrd/zgp/abaw/AU_test/cropped_aligned'
target_root_dir = '/home/data/wyq/ABAW_VA/ABAW2-test/ABAW2-EXP/au_dataset_test/Affwild2_processed/compacted_48'

# 遍历源目录中的每个子目录
for subdir_name in os.listdir(source_dir):
    subdir_path = os.path.join(source_dir, subdir_name)
    if os.path.isdir(subdir_path):
        # 初始化列表以保存图片数组
        images_list = []
        # 按文件名排序以保持顺序
        file_names = sorted(os.listdir(subdir_path))
        for file_name in file_names:
            if file_name.endswith('.jpg'):  # 确保只处理图片文件
                file_path = os.path.join(subdir_path, file_name)
                # 读取图片并转换为NumPy数组
                try:
                    with Image.open(file_path) as img:
                        img_array = np.array(img)
                        images_list.append(img_array)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
        
        # 读取对应的 frame.npy 文件来确定期望的帧数
        frame_npy_path = os.path.join(target_root_dir, subdir_name, "frame.npy")
        if os.path.exists(frame_npy_path):
            frame_data = np.load(frame_npy_path)
            num_frames = frame_data.shape[0]  # 假设frame.npy第一个维度是帧数

            # 调整图片数量
            if len(images_list) > num_frames:
                images_list = images_list[:num_frames]  # 裁剪多余的图片
            elif len(images_list) < num_frames:
                # 复制最后一张图片直到达到 num_frames
                while len(images_list) < num_frames:
                    images_list.append(images_list[-1])
        
            if images_list:
                # 拼接图片数组
                images_array = np.stack(images_list)
                # 创建目标目录
                target_subdir = os.path.join(target_root_dir, subdir_name)
                if not os.path.exists(target_subdir):
                    os.makedirs(target_subdir)
                # 保存NumPy数组到文件
                np.save(os.path.join(target_subdir, "video.npy"), images_array)
                print(f"Saved {os.path.join(target_subdir, 'video.npy')} with shape {images_array.shape}")
            else:
                print(f"No images found in {subdir_path}")
        else:
            print(f"Frame data file not found: {frame_npy_path}")
