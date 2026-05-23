# VGGT-Omega 代码仓库分析

> 基于 Transformer 架构的前向相机和深度重建模型

---

## 项目概述

VGGT-Omega 是由**牛津大学 Visual Geometry Group (VGG)** 和 **Meta AI** 联合开发的相机和深度重建模型。该项目基于 Transformer 架构，能够从单目视频或多张图像中预测相机位姿（camera poses）和深度信息。

| 项目信息 | 说明 |
|---------|------|
| 项目地址 | https://github.com/facebookresearch/vggt-omega |
| 论文 | https://arxiv.org/abs/2605.15195 |
| 许可证 | Meta License |
| Python版本 | >=3.10 |

---

## 目录结构

```
vggt-omega/
├── vggt_omega/                    # 主包目录
│   ├── __init__.py               # 包入口，导出 VGGTOmega 类
│   │
│   ├── models/                   # 核心模型模块
│   │   ├── __init__.py
│   │   ├── vggt_omega.py         # 主模型类 VGGTOmega
│   │   ├── aggregator.py          # 聚合器（交替注意力编码器）
│   │   │
│   │   ├── heads/                # 输出头模块
│   │   │   ├── camera_head.py     # 相机位姿预测头
│   │   │   ├── dense_head.py      # 深度预测头
│   │   │   ├── text_alignment_head.py  # 文本对齐头
│   │   │   └── utils.py
│   │   │
│   │   └── layers/               # 基础层模块
│   │       ├── attention.py       # 注意力机制
│   │       ├── block.py           # Transformer 块
│   │       ├── ffn_layers.py      # FFN 前馈网络
│   │       ├── layer_scale.py     # 层缩放
│   │       ├── patch_embed.py     # 图像块嵌入
│   │       ├── rms_norm.py        # RMS 归一化
│   │       ├── rope_position_encoding.py  # RoPE 位置编码
│   │       ├── vision_transformer.py   # DINOv3 Vision Transformer
│   │       └── utils.py
│   │
│   └── utils/                    # 工具函数模块
│       ├── load_fn.py            # 图像加载和预处理
│       ├── pose_enc.py           # 相机位姿编码/解码
│       ├── rotation.py           # 旋转矩阵/四元数转换
│       └── geometry.py           # 几何工具函数
│
├── examples/                     # 示例视频
│   ├── desert_road.mp4
│   ├── forest_road.mp4
│   ├── lake_speedboat.mp4
│   └── snow_lift.mp4
│
├── demo_gradio.py                # Gradio 交互式演示
├── visual_util.py               # 3D 可视化工具
├── pyproject.toml               # Python 项目配置
├── requirements.txt             # 基础依赖
├── requirements_demo.txt        # Demo 依赖
└── README.md                    # 项目文档
```

---

## 核心模块详解

### 1. 主模型 (`vggt_omega.py`)

**核心类**: `VGGTOmega`

```python
class VGGTOmega(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_alignment: bool = False,
    ) -> None:
```

**主要功能**:
- 整合 Aggregator、CameraHead、DenseHead、TextAlignmentHead
- 使用混合精度推理（AMP, bfloat16/float16）
- 返回字典包含: `pose_enc`、`depth`、`depth_conf`、`camera_and_register_tokens`

---

### 2. 聚合器 (`aggregator.py`)

**核心类**: `Aggregator`

**架构特点**:

| 组件 | 说明 |
|------|------|
| 交替注意力编码器 | 24层交替的自注意力和帧间注意力块 |
| DINOv3 Vision Transformer | 作为图像块嵌入（patch embedding） |
| RoPE 位置编码 | 使用 "max" 归一化模式 |
| Register Tokens | 16个可学习的寄存器令牌 |
| Camera Token | 相机相关特殊令牌 |

```python
# 24层交替块
self.frame_blocks = nn.ModuleList([SelfAttentionBlock(...) for _ in range(24)])
self.inter_frame_blocks = nn.ModuleList([SelfAttentionBlock(...) for _ in range(24)])

# 缓存层索引 - 用于多尺度特征提取
self.cached_layer_indices = (4, 11, 17, 23)
```

---

### 3. 输出头模块 (`heads/`)

#### 3.1 相机头 (`camera_head.py`)

**输出**: 9维位姿编码

```
[translation(3) + quaternion(4) + fov(2)]
```

**网络结构**:
```
TokenNorm -> 4个SelfAttentionBlock -> TokenNorm -> CameraBranch(Linear->GELU->Linear)
```

#### 3.2 密集预测头 (`dense_head.py`)

**输出**: 深度图 `depth` 和深度置信度 `depth_conf`

**网络结构**:
- 多尺度特征融合（来自层4, 11, 17, 23）
- 4个 `FeatureFusionBlock` 级联
- 最终 `pixel_shuffle` 上采样到原始分辨率

**激活函数**:
- 深度: `depth = exp(logits)`
- 置信度: `depth_conf = 1 + exp(logits)` (初始值约1.05)

#### 3.3 文本对齐头 (`text_alignment_head.py`)

**输出**: `text_alignment_embedding`、`text_alignment_token`

> 仅在 `enable_alignment=True` 时启用

---

### 4. 基础层模块 (`layers/`)

| 文件 | 功能 |
|------|------|
| `attention.py` | 自注意力机制，支持 Q/K 归一化、RoPE 位置编码 |
| `block.py` | Transformer 块：`norm1 -> attn -> ls1 -> norm2 -> mlp -> ls2` |
| `vision_transformer.py` | 基于 DINOv3 的 Vision Transformer |
| `rope_position_encoding.py` | 旋转位置编码，支持 `min/max/separate` 归一化模式 |

---

### 5. 工具模块 (`utils/`)

#### 5.1 图像加载 (`load_fn.py`)

```python
def load_and_preprocess_images(
    image_path_list,
    mode="balanced",       # 或 "max_size"
    image_resolution=512,
    patch_size=16
) -> torch.Tensor
```

**预处理模式**:

| 模式 | 说明 |
|------|------|
| `balanced` | 保持总token数接近 `image_resolution^2` |
| `max_size` | 将最长边缩放到 `image_resolution` |

**处理流程**:
1. 加载 RGB 图像
2. 裁剪极端长宽比到 [0.5, 2.0] 范围
3. 根据模式计算目标尺寸
4. 调整大小并转换为张量

#### 5.2 位姿编码 (`pose_enc.py`)

```python
# 编码：外参+内参 -> 9维编码
extri_intri_to_pose_encoding(extrinsics, intrinsics, image_size_hw)

# 解码：9维编码 -> 外参+内参
encoding_to_camera(pose_encoding, image_size_hw, build_intrinsics=True)
```

**9维编码格式**: `[tx, ty, tz, qx, qy, qz, qw, fov_h, fov_w]`

#### 5.3 旋转转换 (`rotation.py`)

四元数格式: `XYZW` (scalar-last)

---

## 数据处理流程

```
输入图像/视频
     │
     ▼
load_and_preprocess_images()      [utils/load_fn.py]
     │
     ▼
VGGTOmega.forward()               [models/vggt_omega.py]
     │
     ├─► Aggregator              [models/aggregator.py]
     │      │
     │      ├─► PatchEmbed (DINOv3 ViT)
     │      ├─► 24层交替自注意力块
     │      └─► 输出多尺度特征 [层4, 11, 17, 23]
     │
     ├─► CameraHead              [models/heads/camera_head.py]
     │      └─► 输出 pose_enc (9D)
     │
     └─► DenseHead                [models/heads/dense_head.py]
             └─► 输出 depth, depth_conf
     │
     ▼
encoding_to_camera()             [utils/pose_enc.py]
     │
     ▼
unproject_depth_map_to_point_map() [demo_gradio.py]
     │
     ▼
predictions_to_glb()             [visual_util.py]
     │
     ▼
GLB 场景输出
```

---

## 依赖关系

### 核心依赖 (`requirements.txt`)

```
torch>=2.3
torchvision>=0.18
numpy<2
Pillow
einops
safetensors
opencv-python
```

### Demo 依赖 (`requirements_demo.txt`)

```
gradio>=5.17.1
viser>=0.2.23
tqdm
scipy
trimesh
matplotlib
requests
onnxruntime
```

---

## 关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `patch_size` | 16 | 图像块大小 |
| `embed_dim` | 1024 | 嵌入维度 |
| `image_resolution` | 512 | 输入图像分辨率 |
| `depth` | 24 | Transformer层数 |
| `num_heads` | 16 | 注意力头数 |
| `num_register_tokens` | 16 | 寄存器令牌数 |
| `cached_layer_indices` | (4, 11, 17, 23) | 缓存层索引 |

---

## 使用示例

```python
import torch
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

# 加载模型
model = VGGTOmega().to("cuda").eval()
model.load_state_dict(torch.load("checkpoint.pt", map_location="cpu"))

# 预处理图像
images = load_and_preprocess_images(["img1.png", "img2.png"], image_resolution=512).to("cuda")

# 推理
with torch.inference_mode():
    predictions = model(images)

# 获取相机参数
extrinsics, intrinsics = encoding_to_camera(
    predictions["pose_enc"],
    predictions["images"].shape[-2:],
)

# 获取深度
depth = predictions["depth"]
depth_conf = predictions["depth_conf"]
```

---

## 可视化工具

### `visual_util.py`

**核心函数**: `predictions_to_glb()`

将 VGGT-Omega 预测结果转换为 GLB 3D 场景：

1. **深度反投影**: 将深度图反投影到3D点云
2. **置信度过滤**: 基于深度不连续性过滤边缘点
3. **天空遮罩**: 可选的天空分割遮罩（使用 skyseg.onnx）
4. **相机可视化**: 在场景中绘制相机位姿
5. **颜色映射**: 从输入图像采样颜色到点云

### `demo_gradio.py`

基于 Gradio 的交互式演示界面：

- 视频/图像上传
- 视频帧采样率控制
- 置信度阈值滑块
- 点数限制滑块
- 相机可视化开关
- 天空/背景过滤开关

---

## 预训练模型

| 模型 | 分辨率 | 文本对齐 | 下载链接 |
|------|--------|----------|----------|
| VGGT-Omega-1B-512 | 512 | 否 | Hugging Face |
| VGGT-Omega-1B-256-Text-Alignment | 256 | 是 | Hugging Face |