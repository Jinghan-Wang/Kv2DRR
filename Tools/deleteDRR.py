import os

# 1. 设置 dataset 文件夹的绝对路径 (请根据你的实际情况修改)
dataset_path = r"/data/dataSet/spine_reg_train"

# 遍历 dataset 下的所有样本文件夹 (例如 0-0098, 0-0316 等)
for sample_folder in os.listdir(dataset_path):
    sample_path = os.path.join(dataset_path, sample_folder)

    # 确保当前遍历到的是一个文件夹
    if not os.path.isdir(sample_path):
        continue

    kv_path = os.path.join(sample_path, "kv2_delete")
    #drr_all_path = os.path.join(sample_path, "drr_all")
    drr_spine2_path = os.path.join(sample_path, "drr_spine4_delete")

    # 如果该样本缺少 kv 文件夹，则跳过
    if not os.path.exists(kv_path):
        print(f"跳过: {sample_folder} (未找到 kv 文件夹)")
        continue

    # 2. 获取 kv 文件夹中保留的有效文件名称集合 (使用 set 提高查找效率)
    valid_filenames = set(os.listdir(kv_path))


    # 3. 定义一个内部函数来清理目标文件夹
    def clean_folder(target_path):
        if not os.path.exists(target_path):
            return

        for filename in os.listdir(target_path):
            # 如果 drr 文件夹中的文件不在 kv 的有效列表中，则执行删除
            if filename not in valid_filenames:
                file_to_delete = os.path.join(target_path, filename)
                try:
                    os.remove(file_to_delete)
                    print(f"已删除: {file_to_delete}")
                except Exception as e:
                    print(f"删除失败 {file_to_delete}: {e}")


    # 4. 对 drr_all 和 drr_spine2 执行清理
    #clean_folder(drr_all_path)
    clean_folder(drr_spine2_path)

print("------------------------")
print("所有多余文件清理完成！")




# import os
# import shutil
#
# root_dir = r"/data/dataSet/spine_reg_train"
#
# for name in os.listdir(root_dir):
#     case_dir = os.path.join(root_dir, name)
#     if not os.path.isdir(case_dir):
#         continue
#
#     src_dir = os.path.join(case_dir, "drr_spine4")
#     dst_dir = os.path.join(case_dir, "drr_spine4_delete")
#
#     if not os.path.isdir(src_dir):
#         print(f"[跳过] 没有 drr: {case_dir}")
#         continue
#
#     try:
#         if os.path.exists(dst_dir):
#             shutil.rmtree(dst_dir)
#             print(f"[删除旧目录] {dst_dir}")
#
#         shutil.copytree(src_dir, dst_dir)
#         print(f"[完成] {src_dir} -> {dst_dir}")
#
#     except Exception as e:
#         print(f"[失败] {src_dir} -> {dst_dir}, 原因: {e}")
#
# print("全部处理完成。")