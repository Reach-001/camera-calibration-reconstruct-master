import cv2
import glob
import os
import shutil
import sys

# 参数
image_dir = 'data/camera'
left_prefix = 'left_'
right_prefix = 'right_'
image_format = 'png'
width = 9
height = 6

# 创建备份目录
backup_dir = os.path.join(image_dir, 'backup')
if not os.path.exists(backup_dir):
    try:
        os.makedirs(backup_dir)
        print(f"创建备份目录: {backup_dir}")
    except Exception as e:
        print(f"创建备份目录失败: {e}")
        sys.exit(1)

pattern_size = (width, height)

# 获取所有图片
left_images = glob.glob(os.path.join(image_dir, f"{left_prefix}*.{image_format}"))
right_images = glob.glob(os.path.join(image_dir, f"{right_prefix}*.{image_format}"))
left_images.sort()
right_images.sort()

if not left_images or not right_images:
    print("错误：未找到图片文件！")
    print(f"左相机图片数量: {len(left_images)}")
    print(f"右相机图片数量: {len(right_images)}")
    sys.exit(1)

print(f"找到 {len(left_images)} 张左相机图片")
print(f"找到 {len(right_images)} 张右相机图片")

print("\n检查左相机图片...")
left_bad_images = []
for filename in left_images:
    try:
        img = cv2.imread(filename)
        if img is None:
            print(f"无法读取图片: {filename}")
            left_bad_images.append(filename)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, _ = cv2.findChessboardCorners(gray, pattern_size, None)
        if not ret:
            print(f"未检测到棋盘格: {filename}")
            left_bad_images.append(filename)
    except Exception as e:
        print(f"处理图片 {filename} 时出错: {e}")
        left_bad_images.append(filename)

print("\n检查右相机图片...")
right_bad_images = []
for filename in right_images:
    try:
        img = cv2.imread(filename)
        if img is None:
            print(f"无法读取图片: {filename}")
            right_bad_images.append(filename)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, _ = cv2.findChessboardCorners(gray, pattern_size, None)
        if not ret:
            print(f"未检测到棋盘格: {filename}")
            right_bad_images.append(filename)
    except Exception as e:
        print(f"处理图片 {filename} 时出错: {e}")
        right_bad_images.append(filename)

# 找出需要删除的图片对
left_bad_numbers = [os.path.basename(f).replace(left_prefix, '').replace(f'.{image_format}', '') for f in left_bad_images]
right_bad_numbers = [os.path.basename(f).replace(right_prefix, '').replace(f'.{image_format}', '') for f in right_bad_images]

if not left_bad_numbers and not right_bad_numbers:
    print("\n所有图片都能正确识别棋盘格！")
    sys.exit(0)

print("\n需要删除的图片对（左右相机）：")
for num in set(left_bad_numbers + right_bad_numbers):
    left_file = os.path.join(image_dir, f"{left_prefix}{num}.{image_format}")
    right_file = os.path.join(image_dir, f"{right_prefix}{num}.{image_format}")
    print(f"左: {left_file}")
    print(f"右: {right_file}")
    print("---")

# 询问是否删除
response = input("\n是否要删除这些无法识别的图片？(y/n): ")
if response.lower() == 'y':
    print("\n开始删除图片...")
    success_count = 0
    fail_count = 0
    
    for num in set(left_bad_numbers + right_bad_numbers):
        left_file = os.path.join(image_dir, f"{left_prefix}{num}.{image_format}")
        right_file = os.path.join(image_dir, f"{right_prefix}{num}.{image_format}")
        
        try:
            # 备份文件
            if os.path.exists(left_file):
                shutil.move(left_file, os.path.join(backup_dir, os.path.basename(left_file)))
                print(f"已备份并删除: {left_file}")
                success_count += 1
            if os.path.exists(right_file):
                shutil.move(right_file, os.path.join(backup_dir, os.path.basename(right_file)))
                print(f"已备份并删除: {right_file}")
                success_count += 1
        except Exception as e:
            print(f"处理文件时出错: {e}")
            fail_count += 1
    
    print(f"\n删除完成！")
    print(f"成功处理: {success_count} 个文件")
    print(f"处理失败: {fail_count} 个文件")
    print(f"已删除的图片已备份到: {backup_dir}")
else:
    print("\n已取消删除操作。")