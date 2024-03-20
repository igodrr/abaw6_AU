import os

# 定义大目录的路径
root_dir = '/home/data/wyq/ABAW_VA/ABAW2-test/ABAW2-EXP/au_dataset_test/Affwild2_processed/compacted_48'

# 初始化没有 'video.npy' 文件的子目录列表
missing_video_npy = []

# 遍历 root_dir 下的每个子目录
for subdir_name in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir_name)
    if os.path.isdir(subdir_path):
        # 检查该子目录下是否存在 'video.npy' 文件
        video_npy_path = os.path.join(subdir_path, "video.npy")
        if not os.path.exists(video_npy_path):
            missing_video_npy.append(subdir_name)

# 打印结果
if missing_video_npy:
    print("以下子目录缺少 'video.npy' 文件：")
    for subdir_name in missing_video_npy:
        print(subdir_name)
else:
    print("所有子目录都包含 'video.npy' 文件。")