# 双目相机深度估计系统

这是一个基于 OpenCV 和 Open3D 的双目相机深度估计系统，支持单 USB 双目相机和双 USB 双目相机。

## 功能特点

- 支持单 USB 双目相机和双 USB 双目相机
- 实时显示双目图像、视差图和深度图
- 支持图像保存功能
- 支持 Open3D 点云显示
- 支持视差图滤波

## 环境配置

### 1. 系统要求
- Windows 10/11 或 Linux
- Python 3.6+
- 双目相机（单 USB 或双 USB）

### 2. 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/Camera-Calibration-Reconstruct.git
cd Camera-Calibration-Reconstruct
```

2. 创建虚拟环境（推荐）：
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

### 3. 依赖包版本
```
opencv-python==4.8.0
numpy==1.24.3
open3d==0.17.0
pyyaml==6.0.1
```

## 使用步骤

### 1. 相机标定

#### 1.1 准备标定板
- 打印棋盘格标定板（9x6 内角点）
- 确保棋盘格平整，无褶皱
- 建议使用 A4 纸打印，实际尺寸约 25mm/格

#### 1.2 采集标定图像
1. 运行图像采集程序：
```bash
python get_stereo_images.py --left_video 0 --save_dir data/camera
```

2. 采集要求：
   - 采集 20-30 张图像
   - 不同角度（上下左右倾斜）
   - 不同距离（近、中、远）
   - 确保棋盘格完全在视野内
   - 按 's' 键保存图像
   - 按 'q' 键退出程序

#### 1.3 单目相机标定
1. 左相机标定：
```bash
python mono_camera_calibration.py --image_dir data/camera --prefix left_ --image_format png --square_size 25 --width 9 --height 6 --save_dir configs/lenacv-camera
```

2. 右相机标定：
```bash
python mono_camera_calibration.py --image_dir data/camera --prefix right_ --image_format png --square_size 25 --width 9 --height 6 --save_dir configs/lenacv-camera
```

3. 检查标定结果：
   - 查看 `configs/lenacv-camera/left__cam.yml` 和 `right__cam.yml`
   - 确保 RMS 误差小于 0.1
   - 如果误差过大，重新采集图像

#### 1.4 双目相机标定
```bash
python stereo_camera_calibration.py --left_dir data/camera --right_dir data/camera --left_prefix left_ --right_prefix right_ --image_format png --square_size 25 --width 9 --height 6 --save_dir configs/lenacv-camera
```

### 2. 深度估计

#### 2.1 单 USB 双目相机
1. 连接相机：
   - 将相机连接到电脑
   - 确保相机驱动正确安装
   - 在设备管理器中确认相机 ID

2. 运行程序：
```bash
python demo.py --left_video 0
```

3. 参数说明：
   - `--left_video`: 摄像头 ID（通常为 0）
   - `--stereo_file`: 标定文件路径
   - `--filter`: 是否使用视差图滤波

#### 2.2 双 USB 双目相机
1. 连接相机：
   - 将两个相机分别连接到电脑
   - 记录左右相机的 ID

2. 运行程序：
```bash
python demo.py --left_video 0 --right_video 1
```

### 3. 操作说明

#### 3.1 程序控制
- 按 'q' 键：退出程序
- 按 's' 键：保存当前帧的左右图像对
- 按 'f' 键：切换视差图滤波

#### 3.2 显示窗口
- Original：原始双目图像
- Disparity：视差图
- Depth：深度图
- Point Cloud：点云显示（如果启用）

## 常见问题解决

### 1. 无法打开摄像头
1. 检查摄像头 ID：
   - 在设备管理器中查看摄像头
   - 尝试不同的 ID（0, 1, 2...）
   - 使用 Windows 相机应用测试

2. 摄像头被占用：
   - 关闭其他使用摄像头的程序
   - 重启摄像头或电脑
   - 检查设备管理器中的摄像头状态

### 2. 标定问题
1. 标定误差过大：
   - 重新采集标定图像
   - 确保棋盘格完全在视野内
   - 增加标定图像数量
   - 检查棋盘格尺寸设置

2. 无法识别棋盘格：
   - 确保光照充足
   - 避免反光和阴影
   - 保持棋盘格平整

### 3. 深度估计问题
1. 深度图质量差：
   - 调整光照条件
   - 避免快速运动
   - 检查相机标定参数
   - 调整视差图滤波参数

2. 图像分割错误：
   - 检查相机分辨率设置
   - 确认左右图像正确分割
   - 调整图像处理参数

## 目录结构说明

```
.
├── configs/                # 配置文件目录
│   └── lenacv-camera/     # 相机标定结果
├── data/                  # 数据目录
│   ├── camera/           # 相机图像
│   └── temp/             # 临时文件
├── demo.py               # 主程序
├── get_stereo_images.py  # 图像采集程序
├── mono_camera_calibration.py    # 单目相机标定
├── stereo_camera_calibration.py  # 双目相机标定
└── requirements.txt      # 依赖包列表
```

## 注意事项

1. 相机标定
   - 确保棋盘格完全在视野内
   - 采集不同角度的图像（建议 20-30 张）
   - 标定误差（RMS）应小于 0.1
   - 保存标定结果前检查参数

2. 深度估计
   - 单 USB 双目相机分辨率设置为 1280x720
   - 确保光照充足
   - 避免快速运动
   - 定期检查标定参数

## 许可证

MIT License

## 联系方式

如有问题，请提交 Issue 或 Pull Request。 