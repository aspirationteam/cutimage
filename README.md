# T恤图案裁剪工具 (CutImage)

这是一个用于自动识别和裁剪T恤上图案的工具，也可用于从各种图像中提取主要设计元素。

## 功能特点

- 自动识别T恤上的设计图案
- 使用颜色分割算法精确提取图案区域
- 使用Canny边缘检测算法提高识别准确度
- 支持不同类型的图像和设计
- 可调整的颜色容差和边框填充
- 自定义种子点位置
- 生成调试图像以供参考

## 安装依赖

确保已安装Python 3.6+，然后安装所需依赖：

```bash
pip install opencv-python numpy pillow
```

## 使用方法

### 基本用法

1. 在`input`目录中放置T恤图像
2. 运行程序：

```bash
python cutimage.py
```

3. 输出的裁剪图像将保存在`output`目录中

### 命令行参数

```
用法: cutimage.py [-h] [-i INPUT] [-o OUTPUT] [-t TOLERANCE] [-p PADDING] [-s SEED] [-f FILE] 
                  [--no-debug] [--no-canny] [--canny-t1 CANNY_T1] [--canny-t2 CANNY_T2]

从T恤或其他图像中提取设计图案

可选参数:
  -h, --help            显示帮助信息并退出
  -i INPUT, --input INPUT
                        输入目录，默认为 "input"
  -o OUTPUT, --output OUTPUT
                        输出目录，默认为 "output"
  -t TOLERANCE, --tolerance TOLERANCE
                        颜色容差，默认为 10
  -p PADDING, --padding PADDING
                        裁剪边框填充比例，默认为 0.05
  -s SEED, --seed SEED  自定义种子点坐标，格式为 "x,y"，如 "100,200"
  -f FILE, --file FILE  处理单个文件而不是整个目录
  --no-debug            不生成调试图像
  --no-canny            不使用Canny边缘检测
  --canny-t1 CANNY_T1   Canny边缘检测第一阈值，默认为 50
  --canny-t2 CANNY_T2   Canny边缘检测第二阈值，默认为 150
```

### 示例

处理单个文件：
```bash
python cutimage.py -f path/to/image.jpg
```

使用自定义种子点：
```bash
python cutimage.py -s 400,500
```

增加颜色容差以处理复杂图像：
```bash
python cutimage.py -t 40
```

减小边框填充：
```bash
python cutimage.py -p 0.02
```

调整Canny边缘检测参数：
```bash
python cutimage.py --canny-t1 50 --canny-t2 150
```

禁用Canny边缘检测：
```bash
python cutimage.py --no-canny
```

## 工作原理

该程序使用三种算法来识别图像中的图案，按优先顺序尝试：

### 1. 颜色分割算法（主要方法）
1. 将图像转换为HSV颜色空间
2. 使用K-Means聚类将图像分割为5个主要颜色区域
3. 分析每个区域的位置、大小和颜色特征
4. 优先选择位于中心区域且颜色与T恤背景差异大的区域
5. 确定边界框并裁剪图像

### 2. Canny边缘检测算法（备用方法）
1. 将图像转换为灰度图并增强对比度
2. 应用Canny边缘检测算法检测边缘
3. 查找图像中的轮廓并过滤小的轮廓
4. 合并轮廓生成掩码
5. 确定边界框并裁剪图像

### 3. 洪水填充算法（最后尝试）
1. 从图像中心或指定点开始填充，直到达到指定的颜色差异阈值
2. 应用形态学操作清理掩码
3. 查找最大轮廓并确定边界框
4. 添加适当的填充并裁剪图像

## 调试信息

程序会在`output/debug`目录中生成调试图像，包括：

- `debug_*.jpg`：显示绿色矩形边框和红色种子点
- `mask_*.jpg`：显示生成的掩码
- `edges_*.jpg`：Canny边缘检测结果
- `contours_*.jpg`：轮廓检测结果
- `segments_*.jpg`：颜色分割结果可视化
- `best_segment_*.jpg`：选中的最佳颜色分割区域

## 故障排除

- 如果颜色分割算法效果不佳，尝试调整边框填充比例（-p参数）
- 如果未能识别图案，尝试调整Canny边缘检测的阈值（--canny-t1和--canny-t2参数）
- 如果洪水填充算法效果不佳，尝试增加颜色容差（-t参数）
- 如果种子点不在图案上，手动指定种子点（-s参数） 