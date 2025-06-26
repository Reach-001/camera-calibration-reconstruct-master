@echo off
echo 开始单目相机标定...

REM 设置 Python 环境
call .\venv\Scripts\activate.bat

REM 运行单目标定程序
python mono_camera_calibration.py --left_dir data/camera --right_dir data/camera --prefix left_ --image_format png --square_size 25 --width 9 --height 6 --save_dir configs/lenacv-camera

echo 单目相机标定完成！
pause 