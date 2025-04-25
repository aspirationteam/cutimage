import cv2
import numpy as np
from PIL import Image
import os
import sys
import argparse

# 默认配置
INPUT_DIR = "input"
OUTPUT_DIR = "output"
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.webp')

# 默认参数值
DEFAULT_COLOR_TOLERANCE = 10
DEFAULT_PADDING_PERCENT = 0.05

# Canny边缘检测参数
CANNY_THRESHOLD1 = 50
CANNY_THRESHOLD2 = 150

def extract_tshirt_design(input_path, output_path, color_tolerance=DEFAULT_COLOR_TOLERANCE, 
                         padding_percent=DEFAULT_PADDING_PERCENT, seed_point=None, debug=True,
                         use_canny=True):
    """
    从T恤图像中提取设计图案
    
    参数:
        input_path: 输入图像路径
        output_path: 输出图像路径
        color_tolerance: 颜色差异阈值
        padding_percent: 边框填充比例
        seed_point: 自定义种子点 (x, y)，如果为None则自动选择
        debug: 是否生成调试图像
        use_canny: 是否使用Canny边缘检测辅助识别
    
    返回:
        是否成功提取
    """
    print(f"处理图像: {os.path.basename(input_path)}")
    
    # 读取图像
    try:
        img = cv2.imread(input_path)
        if img is None:
            print(f"  无法读取图像: {input_path}")
            return False
    except Exception as e:
        print(f"  读取图像出错: {e}")
        return False
    
    # 获取图像尺寸
    height, width = img.shape[:2]
    
    # 创建调试目录
    if debug:
        debug_dir = os.path.join(OUTPUT_DIR, "debug")
        os.makedirs(debug_dir, exist_ok=True)
    
    # 尝试使用颜色分割算法
    print("  使用颜色分割算法...")
    
    # 转换为HSV颜色空间，更适合颜色分割
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 使用K-Means聚类进行颜色分割
    pixel_values = hsv.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 5  # 尝试将图像分为5个主要颜色
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # 将labels转换回原始形状
    segmented_image = labels.reshape(hsv.shape[0], hsv.shape[1])
    
    if debug:
        # 可视化分割结果
        visualized = np.zeros((height, width, 3), dtype=np.uint8)
        colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [0, 255, 255]]
        for i in range(k):
            visualized[segmented_image == i] = colors[i % len(colors)]
        
        vis_path = os.path.join(debug_dir, f"segments_{os.path.basename(input_path)}")
        cv2.imwrite(vis_path, visualized)
    
    # 分析每个分割区域的位置和大小
    # 优先查找中心区域的彩色部分
    
    # 确定中心区域
    center_x, center_y = width // 2, height // 2
    central_region_width = width // 3
    central_region_height = height // 3
    
    # 中心区域的边界
    center_left = center_x - central_region_width // 2
    center_right = center_x + central_region_width // 2
    center_top = center_y - central_region_height // 2
    center_bottom = center_y + central_region_height // 2
    
    # 针对每个分割类别创建掩码
    segment_sizes = []
    segment_masks = []
    for i in range(k):
        # 为当前分割创建掩码
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[segmented_image == i] = 255
        
        # 计算该分割在中心区域的像素数量
        central_mask = np.zeros((height, width), dtype=np.uint8)
        central_mask[center_top:center_bottom, center_left:center_right] = 1
        central_pixels = np.sum(mask[center_top:center_bottom, center_left:center_right] > 0)
        
        # 计算总像素数量
        total_pixels = np.sum(mask > 0)
        
        # 计算集中度（中心区域像素数与总像素数的比率）
        concentration = 0
        if total_pixels > 0:
            concentration = central_pixels / total_pixels
        
        # 计算与T恤灰色的色差
        color_diff = 0
        if total_pixels > 0:
            # 获取该分割的平均颜色（在BGR空间中）
            mean_color = cv2.mean(img, mask=mask)[:3]
            # T恤颜色通常是灰色，约(128, 128, 128)
            tshirt_color = np.array([128, 128, 128])
            color_diff = np.linalg.norm(mean_color - tshirt_color)
        
        # 保存掩码、大小、集中度和色差信息
        segment_masks.append(mask)
        segment_sizes.append((i, total_pixels, concentration, color_diff))
    
    # 按照以下规则排序：
    # 1. 优先选择与T恤色差大的区域（更可能是图案）
    # 2. 优先选择集中在中心区域的分割
    # 3. 避免太小或太大的区域
    
    # 过滤掉太小的区域（小于5%的图像面积）
    min_size = width * height * 0.01
    # 过滤掉太大的区域（大于80%的图像面积）
    max_size = width * height * 0.8
    
    valid_segments = [(i, pixels, conc, diff) for i, pixels, conc, diff in segment_sizes 
                      if min_size < pixels < max_size]
    
    # 如果没有有效分割，再次尝试，放宽条件
    if not valid_segments:
        valid_segments = [(i, pixels, conc, diff) for i, pixels, conc, diff in segment_sizes 
                          if pixels < max_size]
    
    # 如果仍然没有有效分割，使用Canny边缘检测或洪水填充
    if not valid_segments:
        print("  颜色分割未找到有效区域")
        
        # 如果请求了Canny边缘检测，尝试使用它
        if use_canny:
            print("  尝试使用Canny边缘检测...")
            return use_canny_detection(img, output_path, padding_percent, debug)
        else:
            print("  尝试使用洪水填充算法...")
            return use_flood_fill(img, output_path, color_tolerance, padding_percent, seed_point, debug)
    
    # 按色差和集中度的加权和排序
    weighted_segments = [(i, pixels, conc * 0.3 + (diff / 255) * 0.7) 
                         for i, pixels, conc, diff in valid_segments]
    sorted_segments = sorted(weighted_segments, key=lambda x: x[2], reverse=True)
    
    # 选择最佳分割区域
    best_segment_idx = sorted_segments[0][0]
    best_mask = segment_masks[best_segment_idx]
    
    if debug:
        # 保存最佳分割掩码
        mask_path = os.path.join(debug_dir, f"best_segment_{os.path.basename(input_path)}")
        cv2.imwrite(mask_path, best_mask)
    
    # 应用形态学操作清理掩码
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 查找掩码中的轮廓
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("  颜色分割未找到有效轮廓")
        
        # 如果请求了Canny边缘检测，尝试使用它
        if use_canny:
            print("  尝试使用Canny边缘检测...")
            return use_canny_detection(img, output_path, padding_percent, debug)
        else:
            print("  尝试使用洪水填充算法...")
            return use_flood_fill(img, output_path, color_tolerance, padding_percent, seed_point, debug)
    
    # 找到最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 计算边界框
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # 如果边界框太大（几乎覆盖整个图像），尝试其他方法
    if w > width * 0.9 and h > height * 0.9:
        print("  警告：检测到的区域几乎覆盖整个图像，可能是错误识别")
        
        # 如果请求了Canny边缘检测，尝试使用它
        if use_canny:
            print("  尝试使用Canny边缘检测...")
            return use_canny_detection(img, output_path, padding_percent, debug)
        else:
            print("  尝试使用洪水填充算法...")
            return use_flood_fill(img, output_path, color_tolerance, padding_percent, seed_point, debug)
    
    # 添加填充
    padding_x = int(w * padding_percent)
    padding_y = int(h * padding_percent)
    
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(width, x + w + padding_x)
    y2 = min(height, y + h + padding_y)
    
    # 保存带边界框的调试图像
    if debug:
        debug_img = img.copy()
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色矩形
        debug_path = os.path.join(debug_dir, f"debug_{os.path.basename(input_path)}")
        cv2.imwrite(debug_path, debug_img)
    
    # 裁剪图像
    cropped = img[y1:y2, x1:x2]
    
    # 保存结果
    cv2.imwrite(output_path, cropped)
    print(f"  成功提取设计并保存到: {output_path}")
    print(f"  裁剪尺寸: {x2-x1}x{y2-y1} 像素")
    
    return True

def use_canny_detection(img, output_path, padding_percent, debug=True):
    """使用Canny边缘检测算法"""
    height, width = img.shape[:2]
    
    # 创建调试目录
    if debug:
        debug_dir = os.path.join(OUTPUT_DIR, "debug")
        os.makedirs(debug_dir, exist_ok=True)
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 增强图像对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    if debug:
        # 保存增强后的图像
        enhanced_path = os.path.join(debug_dir, f"enhanced_{os.path.basename(output_path)}")
        cv2.imwrite(enhanced_path, enhanced)
    
    # 应用Canny边缘检测
    edges = cv2.Canny(enhanced, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
    
    # 使用膨胀操作连接边缘
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    
    if debug:
        # 保存边缘检测结果
        edges_path = os.path.join(debug_dir, f"edges_{os.path.basename(output_path)}")
        cv2.imwrite(edges_path, dilated_edges)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 分析所有轮廓的位置和大小
    if contours:
        # 将图像划分为9个网格，着重关注中央区域
        central_contours = []
        center_x, center_y = width // 2, height // 2
        grid_size_w, grid_size_h = width // 3, height // 3
        
        # 中心区域的边界
        center_left = center_x - grid_size_w
        center_right = center_x + grid_size_w
        center_top = center_y - grid_size_h
        center_bottom = center_y + grid_size_h
        
        # 创建一个用于显示网格的调试图像
        if debug:
            grid_img = img.copy()
            # 绘制网格线
            cv2.line(grid_img, (center_left, 0), (center_left, height), (0, 255, 255), 1)
            cv2.line(grid_img, (center_right, 0), (center_right, height), (0, 255, 255), 1)
            cv2.line(grid_img, (0, center_top), (width, center_top), (0, 255, 255), 1)
            cv2.line(grid_img, (0, center_bottom), (width, center_bottom), (0, 255, 255), 1)
            grid_path = os.path.join(debug_dir, f"grid_{os.path.basename(output_path)}")
            cv2.imwrite(grid_path, grid_img)
        
        # 过滤小轮廓并优先选择中心区域的轮廓
        min_contour_area = width * height * 0.005  # 最小轮廓面积为图像的0.5%
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_contour_area:
                # 获取轮廓的中心
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 检查轮廓是否在中心区域
                    in_center_region = (center_left <= cx <= center_right and 
                                       center_top <= cy <= center_bottom)
                    
                    # 使用更高的权重添加中心区域的轮廓
                    if in_center_region:
                        central_contours.append((cnt, area * 1.5))  # 中心区域轮廓权重提高50%
                    else:
                        central_contours.append((cnt, area))
        
        # 按面积降序排序轮廓
        sorted_contours = [c[0] for c in sorted(central_contours, key=lambda x: x[1], reverse=True)]
        
        # 绘制所有符合条件的轮廓用于调试
        if debug and sorted_contours:
            contour_img = img.copy()
            cv2.drawContours(contour_img, sorted_contours, -1, (0, 255, 0), 2)
            contour_path = os.path.join(debug_dir, f"contours_{os.path.basename(output_path)}")
            cv2.imwrite(contour_path, contour_img)
        
        # 如果找到轮廓
        if sorted_contours:
            # 尝试合并靠近的轮廓
            # 创建一个新的掩码，绘制所有轮廓
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, sorted_contours, -1, 255, -1)
            
            # 应用形态学操作连接相近的轮廓
            kernel = np.ones((20, 20), np.uint8)  # 使用较大的核以连接相近轮廓
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            if debug:
                # 保存掩码
                mask_path = os.path.join(debug_dir, f"mask_{os.path.basename(output_path)}")
                cv2.imwrite(mask_path, mask)
            
            # 查找掩码的外轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 找到面积最大的轮廓
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 计算边界框
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # 如果边界框太大（几乎覆盖整个图像），则可能是错误识别
                if w > width * 0.9 and h > height * 0.9:
                    print("  警告：检测到的区域几乎覆盖整个图像，可能是错误识别")
                    return False
                else:
                    # 添加填充
                    padding_x = int(w * padding_percent)
                    padding_y = int(h * padding_percent)
                    
                    x1 = max(0, x - padding_x)
                    y1 = max(0, y - padding_y)
                    x2 = min(width, x + w + padding_x)
                    y2 = min(height, y + h + padding_y)
                    
                    # 保存带边界框的调试图像
                    if debug:
                        debug_img = img.copy()
                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色矩形
                        debug_path = os.path.join(debug_dir, f"debug_{os.path.basename(output_path)}")
                        cv2.imwrite(debug_path, debug_img)
                    
                    # 裁剪图像
                    cropped = img[y1:y2, x1:x2]
                    
                    # 保存结果
                    cv2.imwrite(output_path, cropped)
                    print(f"  成功提取设计并保存到: {output_path}")
                    print(f"  裁剪尺寸: {x2-x1}x{y2-y1} 像素")
                    
                    return True
    
    return False

def use_flood_fill(img, output_path, color_tolerance, padding_percent, seed_point, debug=True):
    """使用洪水填充算法"""
    height, width = img.shape[:2]
    
    # 创建调试目录
    if debug:
        debug_dir = os.path.join(OUTPUT_DIR, "debug")
        os.makedirs(debug_dir, exist_ok=True)
    
    # 定义种子点
    if seed_point is None:
        # 自动选择种子点
        center_x = width // 2
        
        # 根据图像类型自动调整种子点位置
        if height > width * 1.5:  # 如果是长方形图像，可能是T恤照片
            center_y = height // 3  # 上移到图像1/3处，通常更接近胸前图案位置
        else:
            center_y = height // 2  # 对于接近正方形的图像，使用中心点
    else:
        # 使用自定义种子点
        center_x, center_y = seed_point
    
    print(f"  使用种子点: ({center_x}, {center_y})")
    
    # 创建掩码
    mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    
    # 应用高斯模糊减少噪声和颜色变化
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 执行洪水填充算法
    lo_diff = (color_tolerance, color_tolerance, color_tolerance)  # 下限颜色差异
    up_diff = (color_tolerance, color_tolerance, color_tolerance)  # 上限颜色差异
    
    # 创建用于填充的图像副本
    img_copy = blurred.copy()
    
    # 执行洪水填充
    cv2.floodFill(
        img_copy,  # 图像
        mask,      # 掩码
        (center_x, center_y),  # 种子点
        (255, 255, 255),       # 新颜色值
        lo_diff,               # 下限差异
        up_diff,               # 上限差异
        cv2.FLOODFILL_MASK_ONLY  # 只修改掩码
    )
    
    # 提取掩码的有效部分 (去掉边框)
    fill_mask = mask[1:-1, 1:-1].copy()
    
    # 应用形态学操作清理掩码
    kernel = np.ones((5, 5), np.uint8)
    fill_mask = cv2.morphologyEx(fill_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 调试图像
    if debug:
        # 保存填充掩码
        mask_path = os.path.join(debug_dir, f"mask_{os.path.basename(output_path)}")
        cv2.imwrite(mask_path, fill_mask)
    
    # 查找填充区域的轮廓
    contours, _ = cv2.findContours(fill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 如果没有找到轮廓，尝试增加容差
    if not contours:
        print("  未找到轮廓，尝试增加颜色容差...")
        higher_tolerance = color_tolerance * 2
        lo_diff = (higher_tolerance, higher_tolerance, higher_tolerance)
        up_diff = (higher_tolerance, higher_tolerance, higher_tolerance)
        
        # 重置掩码
        mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        img_copy = blurred.copy()
        
        # 再次尝试填充
        cv2.floodFill(
            img_copy, mask, (center_x, center_y), (255, 255, 255),
            lo_diff, up_diff, cv2.FLOODFILL_MASK_ONLY
        )
        
        fill_mask = mask[1:-1, 1:-1].copy()
        fill_mask = cv2.morphologyEx(fill_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 更新轮廓
        contours, _ = cv2.findContours(fill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 如果仍然没有找到轮廓，失败
    if not contours:
        print("  未能找到有效轮廓，处理失败")
        return False
    
    # 找到面积最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 获取边界框
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # 添加填充
    padding_x = int(w * padding_percent)
    padding_y = int(h * padding_percent)
    
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(width, x + w + padding_x)
    y2 = min(height, y + h + padding_y)
    
    # 保存带边界框的调试图像
    if debug:
        debug_img = img.copy()
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色矩形
        cv2.circle(debug_img, (center_x, center_y), 5, (0, 0, 255), -1)  # 红色中心点
        
        debug_path = os.path.join(debug_dir, f"debug_{os.path.basename(output_path)}")
        cv2.imwrite(debug_path, debug_img)
    
    # 裁剪图像
    cropped = img[y1:y2, x1:x2]
    
    # 保存结果
    cv2.imwrite(output_path, cropped)
    print(f"  成功提取设计并保存到: {output_path}")
    print(f"  裁剪尺寸: {x2-x1}x{y2-y1} 像素")
    
    return True

def parse_arguments():
    parser = argparse.ArgumentParser(description='从T恤或其他图像中提取设计图案')
    
    parser.add_argument('-i', '--input', default=INPUT_DIR,
                        help=f'输入目录，默认为 "{INPUT_DIR}"')
    parser.add_argument('-o', '--output', default=OUTPUT_DIR,
                        help=f'输出目录，默认为 "{OUTPUT_DIR}"')
    parser.add_argument('-t', '--tolerance', type=int, default=DEFAULT_COLOR_TOLERANCE,
                        help=f'颜色容差，默认为 {DEFAULT_COLOR_TOLERANCE}')
    parser.add_argument('-p', '--padding', type=float, default=DEFAULT_PADDING_PERCENT,
                        help=f'裁剪边框填充比例，默认为 {DEFAULT_PADDING_PERCENT}')
    parser.add_argument('-s', '--seed', type=str, 
                        help='自定义种子点坐标，格式为 "x,y"，如 "100,200"')
    parser.add_argument('-f', '--file', 
                        help='处理单个文件而不是整个目录')
    parser.add_argument('--no-debug', action='store_true',
                        help='不生成调试图像')
    parser.add_argument('--no-canny', action='store_true',
                        help='不使用Canny边缘检测')
    parser.add_argument('--canny-t1', type=int, default=CANNY_THRESHOLD1,
                        help=f'Canny边缘检测第一阈值，默认为 {CANNY_THRESHOLD1}')
    parser.add_argument('--canny-t2', type=int, default=CANNY_THRESHOLD2,
                        help=f'Canny边缘检测第二阈值，默认为 {CANNY_THRESHOLD2}')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置目录
    input_dir = args.input
    output_dir = args.output
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 解析种子点
    seed_point = None
    if args.seed:
        try:
            x, y = map(int, args.seed.split(','))
            seed_point = (x, y)
        except ValueError:
            print(f"警告: 无法解析种子点坐标 '{args.seed}'，将使用自动选择的种子点")
    
    # 更新Canny阈值
    global CANNY_THRESHOLD1, CANNY_THRESHOLD2
    CANNY_THRESHOLD1 = args.canny_t1
    CANNY_THRESHOLD2 = args.canny_t2
    
    # 处理单个文件或目录
    processed = 0
    failed = 0
    
    if args.file:
        # 处理单个文件
        if not os.path.isfile(args.file):
            print(f"错误: 未找到文件 '{args.file}'")
            sys.exit(1)
        
        # 生成输出路径
        output_filename = f"design_{os.path.basename(args.file)}"
        output_path = os.path.join(output_dir, output_filename)
        
        # 提取设计
        if extract_tshirt_design(args.file, output_path, 
                               args.tolerance, args.padding, 
                               seed_point, not args.no_debug,
                               not args.no_canny):
            processed += 1
        else:
            failed += 1
    else:
        # 处理整个目录
        if not os.path.isdir(input_dir):
            print(f"错误: 未找到输入目录 '{input_dir}'")
            print("请创建该目录并放入图像文件。")
            sys.exit(1)
        
        print(f"处理目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        print(f"颜色容差: {args.tolerance}")
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(SUPPORTED_FORMATS):
                input_path = os.path.join(input_dir, filename)
                output_filename = f"design_{filename}"
                output_path = os.path.join(output_dir, output_filename)
                
                # 提取设计
                if extract_tshirt_design(input_path, output_path, 
                                       args.tolerance, args.padding, 
                                       seed_point, not args.no_debug,
                                       not args.no_canny):
                    processed += 1
                else:
                    failed += 1
    
    print("\n处理完成:")
    print(f"  成功: {processed} 图像")
    print(f"  失败: {failed} 图像")

if __name__ == "__main__":
    main()