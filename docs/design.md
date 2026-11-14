# NetPulse 设计文档（v0.2）

## 1. 项目定位

**NetPulse** 是一个面向深度学习架构实验、数学组合与可精确计算正则化目标的**节点流计算图沙盒**（ComfyUI 的拓扑工作流 + EDA/嘉立创式的元件封装与电气规则检查）。

### 核心目标
1. **双粒度元件化组合**：同时提供原子级神经网络层（`conv2d`、`pool2d`、`linear`、`activation`、`batch_norm2d`、`layer_norm`、`dropout`）与宏模块（`cnn_block`、`mlp_block`），并支持精细定制核尺寸、池化策略与权重初始化方式，按 DAG 连接成训练与推理流。
2. **隐层抽头（Tap）与矩阵连线**：支持提取任意模块的指定隐藏层输出，经矩阵投影/张量运算重新接入下游或正则化分支。
3. **可精确计算的正则化目标**：将主任务 Loss 与特征/激活/结构正则项抽象为平级目标节点，支持多目标加权与精确约束实验。
4. **可编程自定义元件**：通过简洁的 Python 基类与端口契约，将手写 PyTorch 核心代码快速注册为标准元件。

### 核心设计原则
- **内核优先，画布外挂**：先夯实元件协议、静态检查与 PyTorch 图编译器，跑通 JSON/YAML 配置驱动的训练闭环，再通过外挂 WebUI 观察拓扑、张量流动与实时 Loss。
- **EDA 式电气规则检查（ERC）**：连线即校验维度、Rank 与 Mask 兼容性；维度冲突时给出明确诊断或按需插入可学习投影桥（AutoBridge）。
- **原生 PyTorch 产物**：不重造算子与求导引擎，图编译的最终产物就是一个标准的 `torch.nn.Module`。

---

## 2. 总体架构

```text
[ CLI / JSON / YAML / 外挂 WebUI ]  ──读写──▶  Graph IR (JSON/YAML 唯一事实源)
                                                    │
           ┌────────────────────────────────────────┴────────────────────────────────────────┐
           ▼                                                                                 ▼
[1. 元件注册表 Registry]                                                          [2. 静态检查与桥接 ERC]
  - 端口契约 PortSpec / TapSpec                                                     - 拓扑排序与环检测
  - 17 个双粒度内置元件 + plugins/ 扫描                                             - Meta Device 形状推导
                                                                                    - AutoBridge 维度适配
           │                                                                                 │
           └────────────────────────────────────────┬────────────────────────────────────────┘
                                                    ▼
                                  [3. 图编译器 GraphCompiler]
                                    - 生成 CompiledGraph(nn.Module)
                                    - Signal 数据流 + Tap 总线 + Loss 汇集
                                                    │
                                                    ▼
                                  [4. 训练与遥测引擎 Trainer / LiveTrainRunner]
                                    - PyTorch 优化循环 / 梯度裁剪 / .pt 导出
                                    - 调速步进、实时分项 Loss 与节点激活采集
```

---

## 3. 核心协议与数据结构（`netpulse/core/`）

### 3.1 统一边载荷：`Signal`（`types.py`）
节点之间传递的不是裸 `Tensor`，而是携带上下文语义的信号包，以解决序列模型 Mask 与跨模态对齐问题：

```python
@dataclass
class Signal:
    data: torch.Tensor
    mask: Optional[torch.Tensor] = None
    meta: dict[str, Any] = field(default_factory=dict)  # 如 layout="BCHW"/"BLD"/"BD"
```

### 3.2 端口契约：`PortSpec` 与 `TapSpec`（`types.py`）
- **PortSpec**：`name`、`kind`（`"tensor" | "loss" | "data_batch"`）、`rank`、`layout`（`"BD" | "BLD" | "BCHW" | "B" | "SCALAR" | "ANY"`）、`require_mask`、`optional`、`dtype`。
- **TapSpec**（隐层抽头）：`name`（如 `"layer_1"`）、`source`（子模块路径如 `"layers.1"`）、`spec`（对应 `PortSpec`）。

### 3.3 元件基类：`BaseComponent`（`component.py`）
- `declare_ports(params) -> ComponentSignature`：仅依赖静态参数声明输入/输出/Tap 端口。
- `expected_input_spec(params, port_name, actual, resolved_in_specs) -> Optional[ResolvedSpec]`（可选钩子）：当元件显式配置了输入尺寸（如 `conv2d` 的 `in_channels`、`linear` 的 `in_features`）或要求双输入对齐（如 `tensor_add`）时，返回期望规格以触发 ERC 维度检查与 `AutoBridge`。
- `build(params, in_specs) -> nn.Module`：根据上游推导出的实际维度构造标准 PyTorch 模块。

### 3.4 图描述：`GraphIR`（`ir.py`）
基于 Pydantic v2 解析同一份 JSON/YAML：
- `nodes`: `[{ id, type, params, position? }]`
- `edges`: `[{ id, src_node, src_port, dst_node, dst_port, edge_kind: "signal"|"tap"|"loss", bridge?: "auto"|"none" }]`
- `losses`: `[{ node_id, weight, enabled }]`
- `trainer`: `optimizer`、`lr`、`weight_decay`、`batch_size`、`epochs`、`max_steps`、`grad_clip`、`device`、`seed` 等。

---

## 4. ERC 静态检查与自动投影桥（`netpulse/core/erc.py`）

在实例化重型权重之前，通过以下流程完成图校验：
1. **拓扑与端口检查**：检测重复节点 ID、未知元件、悬空边、未知端口/抽头、必填端口未连线、单输入口多重连线、孤立节点警告、有向环（Kahn 拓扑排序）。
2. **Meta 形状推导**：在 `with torch.device("meta"):` 下实例化节点模块并执行空前向传播（含 Hook 与 `forward_with_taps` 抽头捕获），零显存开销推导出每条边与每个 Tap 的精确 `shape`、`dtype`、`layout`。
3. **兼容判定与 AutoBridge**：
   - 形状与布局匹配 → 通过。
   - 若目标节点通过 `expected_input_spec()` 声明了固定特征维，且仅特征维不等（如 `(B, 16) -> (B, 8)` 或 `(B, C1, H, W) -> (B, C2, H, W)`）：
     - `bridge="auto"` → 自动生成 `AutoBridgePlan`，由编译器插入可学习的 `_AutoBridgeModule`（`nn.Linear` 或 `1x1 nn.Conv2d`）。
     - `bridge="none"` → ERC 报错 `DIM_MISMATCH`。
   - Rank 或空间维度冲突（如 `BCHW -> BD`）→ 报错 `RANK_MISMATCH` 并提示插入 `Flatten` / `GlobalAvgPool`。

---

## 5. 图编译器与 PyTorch 运行时（`netpulse/core/compiler.py`）

### 5.1 编译产物：`CompiledGraph(nn.Module)`
- 内部以 `nn.ModuleDict` 持有所有节点模块 `nodes_mod` 与自动桥接模块 `bridges_mod`。
- `forward(batch, mode="train", compute_losses=None) -> GraphOutput`：
  - 按拓扑序执行每个节点，将连线引用的 `Tap` 写入 `tap_bus`。
  - `mode == "train"` 时执行所有 `LossNode` 并按权重汇总 `total_loss` 与 `loss_dict`。
  - `mode in ("eval", "infer")` 且未显式开启 `compute_losses` 时跳过 `LossNode`。

### 5.2 隐层抽取（Tap）双模式
1. **约定返回**：如 `MLPBlock` 实现 `forward_with_taps()` 直接返回输出信号与各层 Tap 字典。
2. **PyTorch Forward Hook**：如 `CNNBlock` 无需重写前向代码，编译器按 `TapSpec.source` 自动注册 `register_forward_hook` 捕获中间激活。

---

## 6. 当前内置元件库（17 个，双粒度并存）

| 类别 | `type_id` | 说明 |
| :--- | :--- | :--- |
| **数据源 (`source`)** | `spiral_data` | 秒开 2D 螺旋多分类合成数据 |
| | `mnist_data` | 纯 PyTorch 轻量 MNIST 下载/解析与缓存（`.cache/mnist`，零 `torchvision` 依赖） |
| **原子层 (`backbone`/`op`)** | `conv2d` | 可配 `out_channels`、`kernel_size`、`stride`、`padding`、`dilation`、`groups`、`init`/`bias_init`、可选 `in_channels` |
| | `pool2d` | `max` / `avg` 池化，可配 `kernel_size`、`stride`、`padding` |
| | `linear` | 全连接层（可作特征层或 $k$ 维分类头），可配 `out_features`、`init`/`bias_init`、可选 `in_features` |
| | `activation` | `relu`、`gelu`、`silu`、`tanh`、`sigmoid`、`leaky_relu`、`softmax`、`identity` |
| | `batch_norm2d` | 2D 批归一化（`BCHW`） |
| | `layer_norm` | 层归一化（`BD` / `BLD`） |
| | `dropout` | 随机失活层 |
| **宏模块 (`backbone`)** | `cnn_block` | 多层 2D CNN，支持逐层 `layers` 参数数组或标量参数、`init`、每层 Hook Tap |
| | `mlp_block` | 多层 MLP（`BD` / `BLD`），支持 `init`、每层 `forward_with_taps` Tap |
| **算子/桥接 (`op`/`bridge`)** | `flatten` | 高维张量展平（`BCHW -> BD`） |
| | `global_avg_pool` | 空间全局平均池化（`BCHW -> BD`） |
| | `matrix_bridge` | 显式矩阵投影桥（`Linear` / `1x1 Conv2d`） |
| | `tensor_add` | 双路加权融合（`alpha * a + beta * b`），支持 `b` 自动桥接对齐 `a` |
| **损失与正则 (`loss`)** | `cross_entropy_loss` | 多分类交叉熵监督目标 |
| | `activation_reg_loss` | 隐层激活精确正则化（`l1` / `l2` / `orthogonal` 去相关） |

- **统一初始化器（`netpulse/components/init_utils.py`）**：支持 `kaiming_normal` / `kaiming_uniform`、`xavier_normal` / `xavier_uniform`、`orthogonal`、`normal`、`uniform`、`zeros`、`ones`、`constant`。
- **自定义插件热加载**：在 `plugins/*.py` 中继承 `BaseComponent` 并使用 `@register_component("my_id")` 即可自动加载。

---

## 7. 外挂 WebUI 与 CLI（`netpulse/webui/`、`netpulse/cli.py`）

- **CLI**：
  - `python -m netpulse.cli list`：列出已注册元件。
  - `python -m netpulse.cli check <graph.json>`：执行 ERC 静态检查与 Meta 形状推导。
  - `python -m netpulse.cli train <graph.json> [--save out.pt]`：编译、训练并导出权重快照。
  - `python -m netpulse.cli serve [--host 127.0.0.1] [--port 8765] [--graphs-dir examples]`：启动外挂交互式 WebUI。
- **外挂交互式 WebUI（`aiohttp` + 原生 WebSocket + 单文件 HTML/SVG/JS）**：
  - **组件元数据协议（`/api/components`）**：基于 `BaseComponent.param_schema()` 与 `ComponentRegistry.describe_all()` 向前端提供全量元件参数 Schema、端口定义与分类。
  - **交互式画布编排**：左侧可折叠元件库 + 双击画布快速搜索建点、端口拖拽连线（含 Tap 虚线与 `AutoBridge` 策略切换）、属性检查器（Inspector）动态参数表单、撤销/重做栈、Minimap 小地图与安全写盘（`/api/graph/save`）。
  - **实时 ERC 预检与训练遥测**：每次连线或改参触发 `POST /api/erc/check` 即时高亮报错节点与推导维度；支持直接传入未保存的内存 `GraphIR` 启动 `LiveTrainRunner` 训练，动态调节步进延迟（0~800ms+）、观察连线脉冲动画、节点激活 `norm/mean/std` 与分项 Loss 曲线。

---

## 8. 演进路线图与当前状态

### Phase 0：内核、双粒度元件与闭环验证（✅ 已完成）
- 核心协议、ERC（Meta 推导 + AutoBridge）、`GraphCompiler`、`Trainer`、CLI、`plugins/` 热加载已落地。
- 17 个双粒度内置元件与统一初始化器已实现。
- 三个端到端示例已跑通：`spiral_quickstart.json`（val_acc=1.0000）、`mnist_tap_reg.json`（val_acc=0.9160）、`mnist_atomic.json`（val_acc=0.9629）。

### Phase 1：交互式画布编辑与元件 Schema（✅ 已完成）
- 完成 `ParamSpec` 参数契约、`ComponentRegistry.describe_all()` 元数据导出及单插件错误隔离加载。
- 完成交互式 WebUI 编辑器：元件库面板/快速搜索建点、端口拖拽连线、实时 `/api/erc/check` 预检、属性编辑表单、撤销/重做、Minimap、`/api/graph/save` 落盘与直接训练内存计算图。

### Phase 2：序列/生成扩展与在线流式数据源
- 完善 `Signal.mask` 透传机制，新增 `TransformerBlock`（Causal/Padding Mask 变体）、`LSTMBlock`、基础 `DiffusionSchedule` 与噪声预测分支。
- 完善训练期 Tensor 统计探针（激活分布、梯度范数导出）与本地 CSV/Text/ImageFolder、HF `datasets`（`streaming=True`）/ModelScope 流式数据源适配器。
