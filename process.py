import os

# ========== 配置 ==========
# 数据集根目录
dataset_root = r"C:\pythoncode\yolocode\ultralytics-8.3.163\datasets\emotion"

# 原始类别名称列表（按顺序对应ID 0,1,2,...）
original_names = ["resistant", "irritated", "fear", "happy", "focused", "bored", "surprised"]

# 要移除的类别名称
remove_class_name = "fear"

# 需要处理的子集（根据你的实际文件夹名修改）
splits = ["train", "valid", "test"]  # 注意这里是 valid 不是 val
# ==========================

# 获取要移除的原始ID
remove_id = original_names.index(remove_class_name)
print(f"要移除的类别 '{remove_class_name}' 的原始ID是: {remove_id}")

# 构建新ID的映射表
# 对于原始ID小于 remove_id 的，新ID不变
# 对于原始ID大于 remove_id 的，新ID = 原ID - 1
# 原始ID等于 remove_id 的，直接删除
id_mapping = {}
new_names = []
for i, name in enumerate(original_names):
    if i == remove_id:
        continue
    if i < remove_id:
        id_mapping[i] = len(new_names)  # 新ID按保留顺序递增
    else:  # i > remove_id
        id_mapping[i] = len(new_names)  # 因为跳过了 remove_id，所以新ID就是当前保留的个数
    new_names.append(name)

print("新的类别名称及对应ID:")
for new_id, name in enumerate(new_names):
    print(f"  {new_id}: {name}")

# 遍历每个子集
for split in splits:
    labels_dir = os.path.join(dataset_root, split, "labels")
    if not os.path.exists(labels_dir):
        print(f"警告: 路径不存在，跳过 {labels_dir}")
        continue

    print(f"\n正在处理 {split} 中的标签文件...")
    txt_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    for filename in txt_files:
        file_path = os.path.join(labels_dir, filename)
        # 读取原文件
        with open(file_path) as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            class_id = int(parts[0])
            # 如果该行属于要删除的类别，跳过
            if class_id == remove_id:
                continue
            # 否则根据映射修改ID
            if class_id in id_mapping:
                parts[0] = str(id_mapping[class_id])
                new_lines.append(" ".join(parts) + "\n")
            else:
                # 理论上不会发生，因为所有ID都应该在映射中
                print(f"警告: 文件 {filename} 中发现未知ID {class_id}，已跳过此行")
                continue

        # 写回原文件（直接覆盖）
        with open(file_path, "w") as f:
            f.writelines(new_lines)

    print(f"{split} 处理完成，共处理 {len(txt_files)} 个文件")

print("\n所有处理完成！")
print("请记得更新你的 data.yaml 文件中的 names 列表为：")
print(new_names)
