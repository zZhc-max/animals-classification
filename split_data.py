import os
import shutil
import random

# 设置随机种子，保证结果可复现
random.seed(42)

# 源目录和目标目录
src_dir = r"D:\10animal\archive\raw-img"
dst_dir = r"D:\10animal\data"

train_dir = os.path.join(dst_dir, "train")
val_dir = os.path.join(dst_dir, "val")

# 获取所有类别文件夹
categories = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
print(f"发现 {len(categories)} 个类别: {categories}")

# 创建目标目录结构
for cat in categories:
    os.makedirs(os.path.join(train_dir, cat), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cat), exist_ok=True)

# 对每个类别进行 7:3 划分
total_train = 0
total_val = 0

for cat in categories:
    cat_path = os.path.join(src_dir, cat)
    images = [f for f in os.listdir(cat_path) if os.path.isfile(os.path.join(cat_path, f))]

    # 随机打乱
    random.shuffle(images)

    # 按 7:3 划分
    split_idx = int(len(images) * 0.7)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    # 复制文件到训练集
    for img in train_imgs:
        src = os.path.join(cat_path, img)
        dst = os.path.join(train_dir, cat, img)
        shutil.copy2(src, dst)

    # 复制文件到验证集
    for img in val_imgs:
        src = os.path.join(cat_path, img)
        dst = os.path.join(val_dir, cat, img)
        shutil.copy2(src, dst)

    print(f"{cat}: 总数={len(images)}, 训练集={len(train_imgs)}, 验证集={len(val_imgs)}")
    total_train += len(train_imgs)
    total_val += len(val_imgs)

print(f"\n总计: 训练集={total_train}, 验证集={total_val}, 比例={total_train/(total_train+total_val):.1%}:{total_val/(total_train+total_val):.1%}")
print(f"数据已保存到: {dst_dir}")
