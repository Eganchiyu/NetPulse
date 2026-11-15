# NetPulse

**NetPulse** 是一个面向深度学习架构实验、数学运算组合与可精确计算正则化目标的**节点流计算图沙盒**（ComfyUI 的拓扑工作流 + EDA/嘉立创式的元件封装与电气规则检查）。

将组件库中的模型模块、原子神经网络层、数学算子与损失/正则化目标以 DAG 连接为 `GraphIR`（JSON/YAML），经静态检查（ERC）与 `meta` 设备维度推导后，编译为原生 `torch.nn.Module` 运行训练与推理。

## 核心特性

- **原生 PyTorch 编译**：`GraphCompiler.compile()` 直接生成标准 `CompiledGraph(nn.Module)`，无缝复用 PyTorch `autograd`、优化器与 `state_dict` 导出。
- **EDA 式电气规则检查（ERC）与 AutoBridge**：
  - 自动执行环检测、端口契约校验与 `torch.device("meta")` 零显存形状推导。
  - 维度不匹配时报错提示插入转换元件（如 `Flatten`），或在 `bridge="auto"` 时自动插入可学习的 `MatrixBridge`（`Linear` / `1x1 Conv2d`）。
- **隐层抽头（Tap）与多目标正则化**：
  - 通过约定返回（`forward_with_taps`）或 PyTorch `register_forward_hook` 无侵入抽取任意隐藏层输出至 `tap_bus`。
  - Loss 与正则项节点（如 `cross_entropy_loss`、`activation_reg_loss` 的 `l1`/`l2`/`orthogonal` 模式）与网络层平级连线，统一加权反传。
- **梯度控制（边级 `stop_grad` / `watch_grad`）**：
  - `EdgeIR.stop_grad`：切断该边的反向传播（信号 `detach`），把「特征干预 / 无侵入表征评估」与「训练干扰」解耦——例如挂在隐层上的线性探针或诊断性正则不再污染主干梯度。WebUI 连线以虚线 ⊘ 标出，检查器一键开关。
  - `EdgeIR.watch_grad`：编译期在该边注册 `Tensor.register_hook`，`backward` 时把梯度张量写入 `CompiledGraph.grad_bus`（`GraphOutput.grad_bus`），连线上实时显示 ∇ L2 范数，用于观测梯度消失/爆炸与多 Loss 分支的梯度冲突。
  - 探针元件（`monitor` / `scalar_watch` / `heatmap_preview` / `tensor_peek`）支持 `watch_grad=true` 参数：除前向激活外，还给出该端口的反向梯度范数曲线与梯度分布热力图。
  - ERC 会提示 `STOP_GRAD_ON_LOSS_EDGE`（Loss 分支失去梯度）与 `LOSS_GRAD_BLOCKED`（全图无任何可反传的 Loss 输入，训练必报错）。
- **双粒度元件库（30 个内置元件）与统一初始化**：
  - **数据源 (`source`，4)**：`spiral_data`、`mnist_data`（纯 PyTorch 加载与缓存，无需 `torchvision`）、`seq_wave_data`（波形/记忆线索序列，变长 `mask`）、`seq_forecast_data`（滑窗时序回归，`sine_mix` / `damped_osc` / `ar_trend`）。
  - **主干与原子层 (`backbone`，5 / `op`，10)**：`conv2d`、`pool2d`、`linear`、`activation`、`batch_norm2d`、`layer_norm`、`dropout`（支持 `kaiming` / `xavier` / `orthogonal` / `normal` / `constant` 等权重与偏置初始化）、`cnn_block`、`mlp_block`、`lstm`（步进 `LSTMCell`，Mask 感知）。
  - **算子与桥接 (`op` / `bridge`)**：`flatten`、`global_avg_pool`、`image_to_seq`、`seq_pool`、`tensor_add`、`matrix_bridge`。
  - **目标 (`loss`，3)**：`cross_entropy_loss`、`activation_reg_loss`（`l1` / `l2` / `orthogonal`）、`mse_loss`。
  - **可视化探针 (`output`，7)**：`monitor`、`scalar_watch`、`heatmap_preview`、`tensor_peek`、`gate_dynamics`、`hidden_trajectory`、`step_energy`（零梯度旁路，训练中实时渲染）。
  - 支持在 `plugins/*.py` 或 `plugins/*/__init__.py` 中通过 `@register_component` 热加载自定义元件（支持单插件错误隔离）。
- **元件自带中文元数据（Schema 内声明，UI 零映射表）**：
  - `BaseComponent.title` 声明中文译名、`description` 声明中文简介；`PortSpec.title` / `PortSpec.description` 声明每个输入/输出/Tap 端口的**中文含义**与**简单介绍**。
  - `registry.describe_component()` 将上述元数据随 `/api/components` 一起下发（插件元件同样适用），WebUI 直接消费，无需前端硬编码名称表。
  - 节点可通过 `NodeIR.label` 自定义显示名，随 `GraphIR` 保存与往返。
- **外挂交互式 WebUI**：基于 `aiohttp` + WebSocket 的零构建单页节点编辑 IDE，支持元件库拖放/双击画布快速建点、端口拖拽连线与实时 ERC/AutoBridge 校验、动态参数表单编辑、撤销重做、保存/另存为、工程文件导入导出、模板库、Minimap 小地图、动态调节步进延迟观察张量流动脉冲与实时分项 Loss 曲线。
- **中文优先的元件说明与可视排版**：
  - 元件库按「中文名 + 英文名 + 两句简介」呈现，支持中文关键词搜索。
  - 画布节点卡显示中文标题（或自定义标签）、元件简介与**中文端口标签**，悬停提示英文端口名、布局与端口说明。
  - 属性面板含元件信息卡（中文名/英文名/类别/简介）与端口说明表（中文含义 + 英文端口名 + 布局 + 实时 `shape`）。

## 目录结构

```text
netpulse/
├── core/           # Signal / PortSpec(title, description) / ParamSpec / GraphIR / Registry / ERC / GraphCompiler / Trainer
├── components/     # 内置 30 个元件（含参数 Schema、端口中文元数据）与统一初始化工具
├── webui/          # 外挂可视化与交互编辑服务端 (aiohttp + WS) 与单文件前端 index.html
└── cli.py          # 命令行入口 (list / check / train / serve)
templates/          # 内置经典工作流模板 (01_spiral_mlp … 05_lstm_time_series_forecast)
examples/           # 示例计算图 (spiral_quickstart / mnist_atomic / mnist_tap_reg / lstm_forecast_classic …)
tests/              # 单元测试与端到端训练 / WebUI 测试
docs/               # 架构设计文档 (design.md) 与系统交接文档 (ARCHITECTURE.md)
```

## 部署与运行

### 环境要求

- `Python >= 3.10`
- 核心依赖：`torch`、`pydantic`、`pyyaml`、`aiohttp`
- 测试依赖（可选）：`pytest`

### 1. 安装依赖

建议在独立虚拟环境中安装依赖（请根据本机硬件选择对应的 PyTorch CPU 或 CUDA 版本）：

```bash
git clone https://github.com/Eganchiyu/NetPulse
cd netpulse

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install torch pydantic pyyaml aiohttp
# 如需运行单元测试，可额外安装 pytest：
pip install pytest
```

### 2. 启动交互式 WebUI（推荐）

在项目根目录运行：

```bash
python -m netpulse.cli serve --host 127.0.0.1 --port 8765
```

启动后在浏览器打开：**`http://127.0.0.1:8765`**

常用参数：
- `--host`：监听地址（默认 `127.0.0.1`）
- `--port`：监听端口（默认 `8765`）
- `--graphs-dir`：计算图目录，保存与加载均限定在该目录下（默认 `examples`）
- `--templates-dir`：工作流模板目录（默认 `templates`）
- `--plugins`：自定义插件目录（默认 `plugins`）

WebUI 常用操作：
- **添加节点**：从左侧元件库点击/拖入画布，或**双击画布空白处**打开快速搜索框回车创建。
- **拖拽连线**：从右侧 `output` 或金色 `tap` 端口圆点按住拖至目标节点左侧 `input` 端口释放，连线时自动触发 ERC 静态校验与 AutoBridge 推导。
- **画布内直接编辑元件**：节点标题右侧 **✎** 可重命名显示标签（不影响节点 id 与连线）；**⚙** 展开节点内联「快速参数编辑」面板，改值即触发 ERC 重算，无需切换到属性面板。
- **编辑参数与连接**：点击节点/连线在右侧 **属性编辑** 面板修改参数、权重或 `AutoBridge` 策略；每个参数项可 **↺ 恢复默认值**，数值越界或 JSON 格式错误会就地标红提示；同时可查看该元件的**中文简介**与**每个端口的中文含义与说明**（含布局与实时 `shape`）。支持 `Delete/Backspace` 删除选中项、`Ctrl+Z`/`Ctrl+Y` 撤销重做、`Ctrl+S` 保存。
- **模板与文件**：顶部 **新建 / 模板 / 导入 / 导出 / 另存为 / 删除** 管理工程文件；模板库内置 `01_spiral_mlp` 至 `05_lstm_time_series_forecast` 五个经典工作流。
- **直接训练未保存图**：编辑后无需先落盘，点击 **开始训练** 即可直接提交当前画布图运行，并在 **训练遥测 & ERC** 面板实时调速观察 Loss 曲线与张量脉冲。

### 3. CLI 检查与训练命令

```bash
# 列出已注册元件
python -m netpulse.cli list

# 执行 ERC 静态检查与 Meta 形状推导
python -m netpulse.cli check examples/mnist_atomic.json

# 编译并运行训练闭环（支持导出 .pt 权重）
python -m netpulse.cli train examples/mnist_atomic.json --save .cache/mnist_atomic.pt
```

### 4. 运行测试

```bash
python -m pytest tests -v
```

## 开源协议

本项目采用 [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE) 协议开源。
