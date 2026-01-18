# 3D 日珥速度分析 (3D Velocity Analysis for Prominences)

## 项目概述

本项目基于 **CHASE (Chinese H-alpha Solar Explorer)** 卫星的 RSM (Ramsey Spectral Module) 数据，对太阳日珥的三维速度进行系统分析。

## 主要功能模块

### 1. **数据加载与基本可视化** 📊
- 加载 FITS 格式的 CHASE/RSM 光谱数据
- 显示 Ha 核心 (Ha Core) 和 Ha 翼 (Ha Wing) 的空间分布
- 构建 SunPy Map 对象用于进一步分析

### 2. **图像对齐** 🎯
基于 FITS 头部的 CRPIX 信息补偿太阳中心位移：
- `align_images_by_crpix()` - 全图对齐
- `align_submaps_by_crpix()` - 感兴趣区域对齐
- 可选的 FFT 相关性精细对齐（±5像素范围）
- 输出位移量时间序列

### 3. **视频生成** 🎬
#### 对齐视频
- **全过程视频** - Ha Core 和 Ha Wing 并排显示
- **子图视频** - 主图+细节图+连接线的复合视频
- **对比视频** - 对齐前后的并排对比

关键函数：
- `create_aligned_video()` - 生成全过程视频
- `create_aligned_subplot_video()` - 生成子图视频
- `create_comparison_video()` - 生成对比视频

### 4. **LOS (Line-of-Sight) 速度计算** 📈
#### 点分类
识别三类点：
- **On Plate** (吸收线) - 类型 0
- **On Limb** (日珥发射) - 类型 1
- **In Space** (弱信号) - 类型 2

关键函数：
- `wave_pattern()` - 谱线模式分类
- `classify_region()` - 区域分类
- `majority_filter()` - 多数投票滤波
- `clean_prominence_mask()` - 形态学清理

#### 速度计算（Moment 方法）
基于光谱谱线的矩方法计算 LOS 速度：
- `moment_velocity_emission()` - 日珥发射谱线速度
- `velocity_map_from_mask_on_limb()` - 生成速度图

### 5. **POS (Plane-of-Sky) 速度计算** 🌪️
#### 时间序列追踪
基于连通域匹配的速度计算：
- `pos_velocity_from_masks()` - 最近邻匹配
- `extract_objects()` - 连通分量提取

#### 光流方法
- **Farneback** - 密集光流计算
- **FLCT** (Fourier Local Correlation Tracking) - 相关性追踪

### 6. **谱线分析** 📊
- **高斯拟合** - 单/双高斯分量拟合
- **等强度线** - 多层次等强度线提取
- **相关性分析** - Pearson 相关系数计算
- **中心重心法** - 谱线中心确定

## 数据流程

```
FITS 数据加载
    ↓
图像对齐 (CRPIX)
    ↓
区域分类 (On Plate/Limb/Space)
    ↓
├─ LOS 速度 (Moment 方法)
├─ POS 速度 (Farneback / FLCT)
└─ 谱线分析 (高斯拟合)
    ↓
视频/报告输出
```

## 使用指南

### 快速开始

```python
# 1. 加载数据
from astropy.io import fits
rsms = []
for file in files:
    rsm = fits.open(file)
    rsms.append(rsm)

# 2. 对齐图像
aligned_data, shifts = align_images_by_crpix(rsms, reference_idx=0)

# 3. 生成视频
create_aligned_video(aligned_data, rsms, fps=5)
create_aligned_subplot_video(aligned_data, rsms, 800, 1100, -100, 200, fps=5)

# 4. 计算速度
type_mask = classify_region(rsm, left, right, bottom, top)
vel_limb = velocity_map_from_mask_on_limb(rsm, left, right, bottom, top, type_mask)
```

### 参数配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `left, right, bottom, top` | 感兴趣区域范围 (arcsec) | 800, 1100, -100, 200 |
| `ang_res` | 角分辨率 | 0.5218 × 2 arcsec/pixel |
| `fps` | 视频帧率 | 5 fps |
| `snr_th` | 信噪比阈值 | 5.0 |
| `core_half_A` | 谱线核心半宽 (Å) | 0.6 |

## 输出文件

```
frames/
├── aligned_video/
│   ├── aligned_full_video.mp4          # 全过程视频
│   └── frames_tmp/                      # 临时帧文件
├── aligned_subplot/
│   ├── aligned_subplot_video.mp4       # 子图视频
│   └── frames_tmp/
└── comparison/
    ├── comparison_video.mp4             # 对比视频
    └── frames_tmp/
```

## 核心特性

✨ **多层次分析**
- 全图到子图的分层分析
- 时间序列连贯性保证

🔬 **先进的速度测量**
- 多种谱线分析方法
- LOS 和 POS 速度的联合计算

📹 **高质量可视化**
- 高分辨率视频输出
- 实时位移量可视化
- 对齐效果对比

## 技术栈

- **数据处理**: NumPy, SciPy, Astropy
- **太阳物理**: SunPy, Helioprojective 坐标系
- **可视化**: Matplotlib, GridSpec
- **光流计算**: OpenCV (Farneback), pyflct (FLCT)
- **FITS I/O**: astropy.io.fits

## 参考信息

- **卫星**: CHASE (Chinese H-alpha Solar Explorer)
- **仪器**: RSM (Ramsey Spectral Module)
- **光谱线**: Hα (6562.8 Å)
- **空间分辨率**: ~1.04 arcsec/pixel

## 关键函数速查表

| 功能 | 函数 |
|------|------|
| 对齐 | `align_images_by_crpix()`, `align_submaps_by_crpix()` |
| 分类 | `classify_region()`, `wave_pattern()` |
| LOS速度 | `moment_velocity_emission()`, `velocity_map_from_mask_on_limb()` |
| POS速度 | `pos_velocity_from_masks()`, `pos_velocity_from_masks_dense()` |
| 谱线 | `gaussfit()`, `bi_sectrix()`, `pearson()` |
| 视频 | `create_aligned_video()`, `create_aligned_subplot_video()`, `create_comparison_video()` |

## 笔记

- 数据时间戳使用 ISO 8601 格式
- 所有坐标基于 Helioprojective 系统
- 对齐参考帧默认为第一帧
- 视频生成过程会产生临时文件，完成后自动清理

---

**最后更新**: 2025年12月17日
