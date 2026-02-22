# MuJoCo Viewer 控制说明

## 键盘快捷键

### 环境控制
- `Enter` - 重置环境
- `Space` - 暂停/继续
- `-` - 减慢速度
- `=` - 加快速度
- `,` - 切换到上一个环境
- `.` - 切换到下一个环境

### 可视化控制
- `P` - 切换奖励图表显示
- `R` - 切换 debug 可视化（目标位置、速度箭头等）
- `A` - 切换显示所有环境（默认只显示当前环境）

## 关闭碰撞体显示

MuJoCo 原生 viewer 使用鼠标右键菜单控制可视化选项。

### 方法 1：使用 MuJoCo Viewer UI（推荐）

1. 运行环境后，在 viewer 窗口中**右键点击**
2. 在弹出菜单中找到 **"Rendering"** 或 **"Visualization"** 选项
3. 取消勾选以下选项：
   - `Collision` - 关闭碰撞几何体
   - 或者调整 `Geom Group` 设置

### 方法 2：通过代码修改（需要修改 mjlab）

如果需要默认关闭碰撞体显示，可以修改 `mjlab/src/mjlab/viewer/native/viewer.py`：

```python
def setup(self) -> None:
    """Setup MuJoCo viewer resources."""
    # ... 现有代码 ...
    
    self.vopt = mujoco.MjvOption()
    
    # 关闭碰撞几何体显示
    self.vopt.flags[mujoco.mjtVisFlag.mjVIS_COLLISION] = 0
    
    # 或者只显示视觉几何体（不显示碰撞体）
    # self.catmask = mujoco.mjtCatBit.mjCAT_STATIC.value
```

### 方法 3：通过环境变量（如果 mjlab 支持）

某些 MuJoCo 配置可以通过环境变量控制：

```bash
export MUJOCO_GL=egl  # 使用 EGL 渲染（无头模式）
```

## MuJoCo 可视化标志说明

常用的 `mjtVisFlag` 选项：

- `mjVIS_COLLISION` - 碰撞几何体
- `mjVIS_JOINT` - 关节
- `mjVIS_ACTUATOR` - 执行器
- `mjVIS_CAMERA` - 相机
- `mjVIS_LIGHT` - 光源
- `mjVIS_TENDON` - 肌腱
- `mjVIS_RANGEFINDER` - 测距仪
- `mjVIS_CONSTRAINT` - 约束
- `mjVIS_INERTIA` - 惯性
- `mjVIS_SCLINERTIA` - 缩放惯性
- `mjVIS_PERTFORCE` - 扰动力
- `mjVIS_PERTOBJ` - 扰动对象
- `mjVIS_CONTACTPOINT` - 接触点
- `mjVIS_CONTACTFORCE` - 接触力
- `mjVIS_CONTACTSPLIT` - 接触分离
- `mjVIS_TRANSPARENT` - 透明
- `mjVIS_AUTOCONNECT` - 自动连接
- `mjVIS_COM` - 质心
- `mjVIS_SELECT` - 选择
- `mjVIS_STATIC` - 静态几何体
- `mjVIS_SKIN` - 皮肤

## 几何体类别（Category）说明

MuJoCo 使用 `mjtCatBit` 来分类几何体：

- `mjCAT_STATIC` (1) - 静态几何体（地形、墙壁等）
- `mjCAT_DYNAMIC` (2) - 动态几何体（机器人、物体等）
- `mjCAT_DECOR` (4) - 装饰几何体（可视化标记、箭头等）
- `mjCAT_ALL` (7) - 所有几何体

当前 mjlab viewer 默认使用 `mjCAT_DYNAMIC`，这会显示机器人的碰撞体。

## 临时解决方案

如果你想快速测试不显示碰撞体，可以：

1. 在 viewer 中右键点击
2. 找到 "Geom" 或 "Collision" 相关选项
3. 取消勾选

或者在代码中临时修改（不推荐，仅用于测试）：

```python
# 在 play.py 或训练脚本中
if hasattr(env.unwrapped, 'viewer') and env.unwrapped.viewer:
    viewer = env.unwrapped.viewer
    if hasattr(viewer, 'vopt'):
        viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_COLLISION] = 0
```

## 推荐设置

对于 parkour 任务，推荐的可视化设置：

- ✅ 保留视觉几何体（机器人外观）
- ✅ 显示 debug 可视化（目标位置、速度箭头）
- ❌ 关闭碰撞几何体（避免视觉混乱）
- ❌ 关闭关节、执行器等辅助显示

这样可以获得最清晰的可视化效果。
