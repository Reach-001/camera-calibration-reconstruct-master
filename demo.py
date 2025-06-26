# -*-coding: utf-8 -*-

import os
import cv2
import argparse
import numpy as np
from core.utils import image_utils, file_utils
from core import camera_params, stereo_matcher


class StereoDepth(object):
    """双目测距"""

    def __init__(self, stereo_file, width=640, height=480, filter=True, use_open3d=True, use_pcl=False):
        """
        :param stereo_file: 双目相机内外参数配置文件
        :param width: 相机分辨率width
        :param height:相机分辨率height
        :param filter: 是否使用WLS滤波器对视差图进行滤波
        :param use_open3d: 是否使用open3d显示点云
        :param use_pcl: 是否使用PCL显示点云
        """
        self.count = 0
        self.filter = filter
        self.camera_config = camera_params.get_stereo_coefficients(stereo_file)
        self.use_pcl = use_pcl
        self.use_open3d = use_open3d
        # 初始化3D点云
        if self.use_pcl:
            # 使用open3d显示点云
            from core.utils_pcl import pcl_tools
            self.pcl_viewer = pcl_tools.PCLCloudViewer()
        if self.use_open3d:
            # 使用PCL显示点云
            from core.utils_3d import open3d_visual
            self.open3d_viewer = open3d_visual.Open3DVisual(camera_intrinsic=self.camera_config["K1"],
                                                            depth_width=width,
                                                            depth_height=height)
            self.open3d_viewer.show_image_pcd(True)
            self.open3d_viewer.show_origin_pcd(True)
            self.open3d_viewer.show_image_pcd(True)
        assert (width, height) == self.camera_config["size"], Exception("Error:{}".format(self.camera_config["size"]))

    def test_pair_image_file(self, left_file, right_file):
        """
        测试一对左右图像
        :param left_file: 左路图像文件
        :param right_file: 右路图像文件
        :return:
        """
        frameR = cv2.imread(left_file)
        frameL = cv2.imread(right_file)
        self.task(frameR, frameL, waitKey=0)

    def capture1(self, video):
        """
        用于采集单USB连接线的双目摄像头(左右摄像头被拼接在同一个视频中显示)
        :param video:int or str,视频路径或者摄像头ID
        """
        print(f"正在打开摄像头 {video}...")
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"错误：无法打开摄像头 {video}")
            print("请检查：")
            print("1. 摄像头是否正确连接")
            print("2. 摄像头是否被其他程序占用")
            print("3. 摄像头驱动是否正确安装")
            return

        # 设置摄像头分辨率
        print("设置摄像头分辨率...")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640* 2)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)       
        
        # 验证设置是否成功
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"摄像头信息：")
        print(f"- 分辨率: {width}x{height}")
        print(f"- FPS: {fps}")
        
        if width != 640 or height != 720:
            print("警告：摄像头分辨率设置失败")
            print("尝试使用其他分辨率...")
            # 尝试其他常见分辨率
            resolutions = [(1280, 480), (640, 480), (1280, 720), (640, 240)]
            for w, h in resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if actual_w == w and actual_h == h:
                    print(f"成功设置分辨率: {w}x{h}")
                    width = w
                    height = h
                    break
        
        self.count = 0
        while True:
            success, frame = cap.read()
            if not success:
                print("无法读取摄像头画面")
                break
            
            if frame is None:
                print("摄像头返回空帧")
                break
            
            # 分割左右图像
            frameL = frame[:, :int(width/2), :]
            frameR = frame[:, int(width/2):, :]
            
            # 确保左右图像尺寸相同
            if frameL.shape != frameR.shape:
                print(f"警告：左右图像尺寸不匹配")
                print(f"左图像尺寸: {frameL.shape}")
                print(f"右图像尺寸: {frameR.shape}")
                # 调整右图像尺寸以匹配左图像
                frameR = cv2.resize(frameR, (frameL.shape[1], frameL.shape[0]))
            
            # 显示原始图像
            cv2.imshow('Original', frame)
 
            
            self.count += 1
            self.task(frameL, frameR, waitKey=5)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 按 q 退出
                break
            elif key == ord('s'):  # 按 s 保存图像
                save_dir = "./data/temp"
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, f"left_{self.count:04d}.png"), frameL)
                cv2.imwrite(os.path.join(save_dir, f"right_{self.count:04d}.png"), frameR)
                print(f"已保存图像对 {self.count}")
            
        cap.release()
        cv2.destroyAllWindows()

    def capture2(self, left_video, right_video):
        """
        用于采集双USB连接线的双目摄像头
        :param left_video:int or str,左路视频路径或者摄像头ID
        :param right_video:int or str,右视频路径或者摄像头ID
        :return:
        """
        capL = image_utils.get_video_capture(left_video)
        capR = image_utils.get_video_capture(right_video)
        width, height, numFrames, fps = image_utils.get_video_info(capL)
        width, height, numFrames, fps = image_utils.get_video_info(capR)
        self.count = 0
        while True:
            successL, frameL = capL.read()
            successR, frameR = capR.read()
            if not (successL and successR):
                print("No more frames")
                break
            self.count += 1
            self.task(frameL, frameR, waitKey=50)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Get key to stop stream. Press q for exit
                break
        capL.release()
        capR.release()
        cv2.destroyAllWindows()

    def get_3dpoints(self, disparity, Q, scale=1.0):
        """
        计算像素点的3D坐标（左相机坐标系下）
        reprojectImageTo3D(disparity, Q),输入的Q,单位必须是毫米(mm)
        :param disparity: 视差图
        :param Q: 重投影矩阵Q=[[1, 0, 0, -cx]
                           [0, 1, 0, -cy]
                           [0, 0, 0,  f]
                           [1, 0, -1/Tx, (cx-cx`)/Tx]]
            其中f为焦距，Tx相当于平移向量T的第一个参数
        :param scale: 单位变换尺度,默认scale=1.0,单位为毫米
        :return points_3d:ndarray(np.float32),返回三维坐标points_3d，三个通道分布表示(X,Y,Z)
                    其中Z是深度图depth, 即距离,单位是毫米(mm)
        """
        # 返回三维坐标points_3d，三个通道分布表示(X,Y,Z)
        # depth = stereo_matcher.get_depth(disparity, Q, scale=1.0)
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        # x, y, depth = cv2.split(points_3d)
        # baseline = abs(camera_config["T"][0])
        # baseline = 1 / Q[3, 2]  # 基线也可以由T[0]计算
        # fx = abs(Q[2, 3])
        # depth = (fx * baseline) / disparity
        points_3d = points_3d * scale
        points_3d = np.asarray(points_3d, dtype=np.float32)
        return points_3d

    def get_disparity(self, imgL, imgR, use_wls=True):
        """
        :param imgL: 畸变校正和立体校正后的左视图
        :param imgR：畸变校正和立体校正后的右视图
        :param use_wls：是否使用WLS滤波器对视差图进行滤波
        :return dispL:ndarray(np.float32),返回视差图
        """
        dispL = stereo_matcher.get_filter_disparity(imgL, imgR, use_wls=use_wls)
        # dispL = disparity.get_simple_disparity(imgL, imgR)
        return dispL

    def get_rectify_image(self, imgL, imgR):
        """
        畸变校正和立体校正
        根据更正map对图片进行重构
        获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
        :param imgL:
        :param imgR:
        :return:
        """
        # camera_params.get_rectify_transform(K1, D1, K2, D2, R, T, image_size)
        left_map_x, left_map_y = self.camera_config["left_map_x"], self.camera_config["left_map_y"]
        right_map_x, right_map_y = self.camera_config["right_map_x"], self.camera_config["right_map_y"]
        rectifiedL = cv2.remap(imgL, left_map_x, left_map_y, cv2.INTER_LINEAR, borderValue=cv2.BORDER_CONSTANT)
        rectifiedR = cv2.remap(imgR, right_map_x, right_map_y, cv2.INTER_LINEAR, borderValue=cv2.BORDER_CONSTANT)
        return rectifiedL, rectifiedR

    def task(self, frameL, frameR, waitKey=5):
        """
        :param frameL: 左路视频帧图像(BGR)
        :param frameR: 右路视频帧图像(BGR)
        """
        # 畸变校正和立体校正
        rectifiedL, rectifiedR = self.get_rectify_image(imgL=frameL, imgR=frameR)
        # 绘制等间距平行线，检查立体校正的效果
        # calibrate_tools.draw_line_rectify_image(rectifiedL, rectifiedR)
        # We need grayscale for disparity map.
        grayL = cv2.cvtColor(rectifiedL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectifiedR, cv2.COLOR_BGR2GRAY)
        # Get the disparity map
        dispL = self.get_disparity(grayL, grayR, self.filter)
        points_3d = self.get_3dpoints(disparity=dispL, Q=self.camera_config["Q"])
        self.show_3dcloud_for_open3d(frameL, frameR, points_3d)
        self.show_3dcloud_for_pcl(frameL, frameR, points_3d)
        self.show_2dimage(frameL, frameR, points_3d, dispL, waitKey=waitKey)

    def show_3dcloud_for_open3d(self, frameL, frameR, points_3d):
        """
        使用open3d显示点云
        :param frameL:
        :param frameR:
        :param points_3d:
        :return:
        """
        if self.use_open3d:
            x, y, depth = cv2.split(points_3d)  # depth = points_3d[:, :, 2]
            self.open3d_viewer.show(color_image=frameL, depth_image=depth)

    def show_3dcloud_for_pcl(self, frameL, frameR, points_3d):
        """
        使用PCL显示点云
        :param frameL:
        :param frameR:
        :param points_3d:
        :return:
        """
        if self.use_pcl:
            self.pcl_viewer.add_3dpoints(points_3d/1000, frameL)
            self.pcl_viewer.show()

    def show_2dimage(self, frameL, frameR, points_3d, dispL, waitKey=0):
        """
        :param frameL:
        :param frameR:
        :param dispL:
        :param points_3d:
        :return:
        """
        x, y, depth = cv2.split(points_3d)  # depth = points_3d[:, :, 2]
        depth_colormap = stereo_matcher.get_visual_depth(depth)
        dispL_colormap = stereo_matcher.get_visual_disparity(dispL)
        image_utils.addMouseCallback("left", depth, info="depth=%fmm")
        image_utils.addMouseCallback("right", depth, info="depth=%fmm")
        image_utils.addMouseCallback("disparity-color", depth, info="depth=%fmm")
        image_utils.addMouseCallback("depth-color", depth, info="depth=%fmm")
        result = {"frameL": frameL, "frameR": frameR, "disparity": dispL_colormap, "depth": depth_colormap}
        cv2.imshow('left', frameL)
        cv2.imshow('right', frameR)
        cv2.imshow('disparity-color', dispL_colormap)
        cv2.imshow('depth-color', depth_colormap)
        key = cv2.waitKey(waitKey)
        self.save_images(result, self.count, key)
        if self.count <= 2:
            cv2.moveWindow("left", 700, 0)
            cv2.moveWindow("right", 1400, 0)
            cv2.moveWindow("disparity-color", 700, 700)
            cv2.moveWindow("depth-color", 1400, 700)

    def save_images(self, result, count, key, save_dir="./data/temp"):
        """
        :param result:
        :param count:
        :param key:
        :param save_dir:
        :return:
        """
        if key == ord('q'):
            exit(0)
        elif key == ord('c') or key == ord('s'):
            file_utils.create_dir(save_dir)
            print("save image:{:0=4d}".format(count))
            cv2.imwrite(os.path.join(save_dir, "left_{:0=4d}.png".format(count)), result["frameL"])
            cv2.imwrite(os.path.join(save_dir, "right_{:0=4d}.png".format(count)), result["frameR"])
            cv2.imwrite(os.path.join(save_dir, "disparity_{:0=4d}.png".format(count)), result["disparity"])
            cv2.imwrite(os.path.join(save_dir, "depth_{:0=4d}.png".format(count)), result["depth"])


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', 'y', '1')


def get_parser():
    parser = argparse.ArgumentParser(description='双目相机深度估计')
    parser.add_argument('--stereo_file', type=str, default="configs/lenacv-camera/stereo_cam.yml", help='双目相机标定文件')
    parser.add_argument('--left_video', type=int, default=1, help='摄像头ID（单USB双目相机）')
    parser.add_argument('--right_video', type=int, default=-1, help='右摄像头ID（单USB双目相机设为-1）')
    parser.add_argument('--filter', type=str2bool, nargs='?', const=True, default=True, help='是否使用视差图滤波')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    stereo = StereoDepth(args.stereo_file, filter=args.filter)
    
    try:
        if args.right_video == -1:
            # 单USB双目相机模式
            print("使用单USB双目相机模式")
            stereo.capture1(video=args.left_video)
        else:
            # 双USB双目相机模式
            print(f"使用双USB双目相机模式: 左={args.left_video}, 右={args.right_video}")
            stereo.capture2(left_video=args.left_video, right_video=args.right_video)
    except Exception as e:
        print(f"运行出错: {str(e)}")
        print("\n可能的解决方案:")
        print("1. 确认摄像头ID是否正确")
        print("2. 检查摄像头是否被其他程序占用")
        print("3. 尝试不同的摄像头ID")
        print("4. 重启摄像头或电脑")
