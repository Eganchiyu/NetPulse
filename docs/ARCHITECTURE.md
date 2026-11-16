# NetPulse 系统架构与开发交接文档

本文档描述 **NetPulse** 当前已实现的完整代码架构、运行时数据流、核心协议与模块接口，用于代码审阅及跨会话（Session）交接。

---

## 1. 分层架构总览

```text
+-----------------------------------------------------------------------------+
| 入口层: CLI (netpulse/cli.py)  |  外挂 WebUI (netpulse/webui/)              |
|  - list / check / train / serve |  - aiohttp REST + WS + index.html         |
+---------------------------------+-------------------------------------------+
                                  | 读取 GraphIR (JSON/YAML)
                                  v
+-----------------------------------------------------------------------------+
| 内核编译与校验层 (netpulse/core/)                                            |
|  1. ir.py       : GraphIR / NodeIR / EdgeIR / LossConfig / TrainerConfig    |
|  2. registry.py : ComponentRegistry / @register_component / plugins 扫描    |
|  3. erc.py      : 拓扑排序 + 端口契约校验 + Meta Device 推导 + AutoBridge    |
|  4. compiler.py : GraphCompiler -> CompiledGraph(nn.Module)                 |
|  5. trainer.py  : Trainer 训练/验证循环 + 权重导出 (.pt)                     |
+---------------------------------+-------------------------------------------+
                                  | 实例化 & 组装
                                  v
+-----------------------------------------------------------------------------+
| 元件库层 (netpulse/components/)                                              |
|  - sources.py    : SpiralData, MNISTData (纯 PyTorch 无 torchvision 依赖)    |
|  - layers.py     : Conv2d, Pool2d, Linear, Activation, BatchNorm2d, LN, Drop|
|  - backbones.py  : CNNBlock (Hook Tap), MLPBlock (forward_with_taps)        |
|  - ops.py        : Flatten, GlobalAvgPool, MatrixBridge, TensorAdd          |
|  - losses.py     : CrossEntropyLoss, ActivationRegLoss (l1/l2/orthogonal)   |
|  - init_utils.py : 统一权重/偏置初始化器 (自动跳过 meta device)               |
+-----------------------------------------------------------------------------+
```

---

## 2. 仓库目录与文件职责

```text
netpulse/
├── netpulse/
│   ├── __init__.py                  # 公共 API 导出
│   ├── cli.py                       # CLI 主入口: list, check, train, serve
│   ├── core/
│   │   ├── types.py                 # Signal, PortSpec, TapSpec, ParamSpec, ComponentSignature, ResolvedSpec, Diagnostic, GraphOutput
│   │   ├── ir.py                    # Pydantic v2 图结构模型与 JSON/YAML 加载/保存 (to_file)
│   │   ├── component.py             # BaseComponent 抽象基类（含 param_schema / display_name / description）
│   │   ├── registry.py              # 注册表、describe_all() 元数据导出与 plugins 错误隔离加载
│   │   ├── subgraph.py              # 子图递归展开引擎（add_graph_prefix 风格节点/边/端口重映射）
│   │   ├── erc.py                   # ERC 静态检查、Meta 形状推导（联动 batch_size）与 AutoBridgePlan
│   │   ├── compiler.py              # CompiledGraph(nn.Module)、_AutoBridgeModule 与 GraphCompiler
│   │   └── trainer.py               # Trainer、多 Source 协同训练、可扩展 metrics 与 checkpoint 保存
│   ├── components/
│   │   ├── __init__.py              # 导入全部内置元件完成默认注册
│   │   ├── init_utils.py            # apply_init() 统一初始化工具
│   │   ├── sources.py               # spiral_data, mnist_data, seq_wave_data, seq_forecast_data
│   │   ├── layers.py                # 原子层: conv2d, pool2d, linear, activation, batch_norm2d, layer_norm, dropout
│   │   ├── backbones.py             # 宏模块: cnn_block, mlp_block, lstm
│   │   ├── ops.py                   # flatten, global_avg_pool, image_to_seq, seq_pool, matrix_bridge, tensor_add
│   │   ├── losses.py                # cross_entropy_loss, activation_reg_loss, mse_loss
│   │   └── outputs.py               # monitor, scalar_watch, heatmap_preview, tensor_peek, gate_dynamics, hidden_trajectory, step_energy
│   └── webui/
│       ├── __init__.py
│       ├── runner.py                # LiveTrainRunner: 后台调速训练、多 Source 遥测、bridge_stats 与子图回映
│       ├── server.py                # aiohttp HTTP + WebSocket 服务端（/static 静态路由、/api/erc/check 子图下钻）
│       ├── static/probe_canvas.js   # 独立拆出的 Canvas 探针绘图模块
│       └── index.html               # 零构建交互式节点编辑 IDE、子图框选下钻、Minimap 与 SVG Loss 曲线
├── examples/
│   ├── spiral_quickstart.json       # 4 节点极速合成螺旋集闭环
│   ├── mnist_tap_reg.json           # CNNBlock + Hook Tap + MatrixBridge + 正交正则闭环
│   └── mnist_atomic.json            # 15 节点全原子层拼装 MNIST 闭环
├── tests/
│   ├── test_erc_and_compiler.py     # 环检测、Rank 报错、AutoBridge、TensorAdd、插件热加载
│   ├── test_atomic_layers.py        # 原子层初始化、维度校验、增强 Block 与 mnist_atomic
│   ├── test_e2e_training.py         # spiral / mnist_tap_reg 端到端训练与 .pt 重载推理
│   └── test_webui.py                # WebUI REST API 与 WebSocket 实时数据流集成测试
├── docs/
│   ├── design.md                    # 产品与演进设计文档 (v0.2)
│   └── ARCHITECTURE.md              # 本文档
├── memo/
│   └── Log.md                       # 阶段开发日志
├── README.md
├── LICENSE                          # GNU AGPL-3.0
└── .gitignore
```

---

## 3. 核心生命周期与数据流

### 3.1 编译期（`run_erc` -> `GraphCompiler.compile`）

1. **图展开与校验（`netpulse/core/subgraph.py`、`erc.py`）**：
   - 若图包含 `subgraphs` 或 `type == "subgraph"` 节点，先经 `expand_graph()` 递归展平为标准 `GraphIR`（检测 `SUBGRAPH_CYCLE`、`UNKNOWN_SUBGRAPH`、`INVALID_SUBGRAPH_MAPPING`），并保留 `flat_to_outer` / `outer_port_targets` 回映表。
   - 验证节点 ID 唯一性、元件已在 `ComponentRegistry` 注册。
   - 调用 `cls.declare_ports(node.params)` 获取静态签名 `ComponentSignature`。
   - 检查所有边引用的节点、输出端口 / Tap 名称、目标输入端口、必填端口连通性及单输入唯一性。
   - Kahn 算法执行拓扑排序并检测有向环（`CYCLE_DETECTED`）。
   - 联动 `graph.trainer.batch_size` 注入 `source` 节点 Meta 推导，并在 `batch_size < 2` 时拦截 `BatchNorm2d`（`BATCH_SIZE_TOO_SMALL_FOR_BN`）或警告 `orthogonal` 正则退化（`DEGENERATE_ORTHOGONAL_REG_BS1`）。
2. **Meta Device 形状推导与 AutoBridge**：
   - 按 `topo_order` 遍历每个节点，对每个输入端口检查 `Rank`、`Layout`、`require_mask`。
   - 若组件实现了 `expected_input_spec(params, port_name, actual, resolved_in_specs)` 并返回固定期望尺寸，且与上游特征维不一致：
     - 边满足 `_can_autobridge`（同 Rank、同 Layout、仅特征维不同）且 `edge.bridge == "auto"` → 生成 `AutoBridgePlan`，更新下游感知的 `in_specs`；
     - 若 `edge.bridge == "none"` → 报告 `DIM_MISMATCH` 错误。
   - 在 `with torch.device("meta"):` 下调用 `comp.build(params, in_specs)` 构造无显存占用的模块，对 `meta` 张量执行一次前向传播（含 `forward_with_taps` 或临时 `register_forward_hook`），精确记录所有输出及被连线 Tap 的 `ResolvedSpec(shape, dtype, layout, has_mask)`。
3. **图编译（`netpulse/core/compiler.py`）**：
   - 在真实设备上实例化所有节点模块放入 `CompiledGraph.nodes_mod`（`nn.ModuleDict`），并为 `AutoBridgePlan` 实例化 `_AutoBridgeModule`（`Linear` 或 `1x1 Conv2d`）放入 `CompiledGraph.bridges_mod`。
   - 对非 `forward_with_taps` 的被抽头节点，在编译期一次性注册 `register_forward_hook` 将中间激活缓存至 `_hook_buffer`。

### 3.2 运行期（`CompiledGraph.forward`）

1. 按 `topo_order` 顺序执行：
   - `category == "source"`：从 `batch[nid]` 读取 `{x: Signal, y: Signal}` 输出（Meta 推导期或空输入时生成虚拟符号张量）。
   - 其他节点：从上游 `outputs` 或 `tap_bus` 取出对应 `Signal`，若该边存在 `AutoBridge` 则先经过 `bridges_mod[edge.id]` 投影，组装 `node_inputs: dict[str, Signal]` 传入节点执行。
   - 梯度控制（见 §3.2.1）：边带 `watch_grad` 时对过边张量注册反向 Hook 写入 `grad_bus`；边带 `stop_grad` 时把 `Signal.detached()` 交给下游。
   - 节点若有被引用 Tap，写入 `tap_bus[f"{nid}.{tap_name}"]`。
   - `category == "loss"`：在 `compute_losses=True`（默认 `mode == "train"`）时执行并写入 `loss_dict[nid]`；`mode in ("eval", "infer")` 默认跳过。
2. `total_loss` 根据 `GraphIR.losses` 中的 `weight` 与 `enabled` 汇总，供原生 `total_loss.backward()` 反传。

#### 3.2.1 梯度控制与梯度总线

- **`EdgeIR.stop_grad`**：边组装完成后执行 `sig.detached()`，下游分支（诊断探针、表征评估 Loss、特征干预）的计算图在此截断，梯度不再回传上游主干；前向数值不受影响。ERC 对「阻断梯度却接进 Loss 节点」给出 `STOP_GRAD_ON_LOSS_EDGE` 警告，对「全图所有可反传 Loss 输入均被阻断」给出 `LOSS_GRAD_BLOCKED` 错误（否则 `backward()` 必然报错）。
- **`EdgeIR.watch_grad`**：前向时对过边张量 `Tensor.register_hook(...)`，`backward` 期间把 `grad.detach()` 写入 `CompiledGraph.grad_bus[edge_id]`（每次 `forward` 开头清空，`GraphOutput.grad_bus` 持有同一字典引用，故 `backward()` 后即可读取）。`LiveTrainRunner` 将每条观测边的梯度 `mean/std/norm` 作为 `grad_stats` 随 `step` 消息推送，前端在连线上显示 ∇ L2 范数。
- **探针侧梯度观测**：`_OutputProbeModule(watch_grad=True)`（`monitor` / `scalar_watch` / `heatmap_preview` / `tensor_peek` 的 `watch_grad` 参数）在自身输入张量上注册 Hook，快照中附 `grad` 段（梯度统计、梯度分布热力图、梯度范数历史曲线），与 `tap_bus` 的可视化链路完全复用。注意 Hook 只在张量仍属反传路径时触发：若该端口的输出不参与任何 Loss，梯度为空属正常。

### 3.3 外挂交互式 WebUI 数据流（`netpulse/webui/`）

- **前端初始化与元件协议**：请求 `/api/components`（导出全量元件 `ParamSpec`、端口定义与分类）、`/api/graphs` 和 `/api/graph?name=...`，服务端运行 `run_erc` 并通过 `_compute_auto_layout` 计算分层坐标，前端渲染节点卡片、端口引脚、贝塞尔连线与静态形状标签。
- **交互式编辑与实时 ERC 预检**：
  - 前端在内存中维护 `draftGraph` 与撤销/重做栈；每次增删节点、拖拽端口连线、修改参数或切换 `AutoBridge` 时防抖调用 `POST /api/erc/check`（`serialize_graph_ir`），实时高亮 ERC 错误节点/边并刷新形状标注。
  - `POST /api/graph/save` 经文件名安全校验后将 `GraphIR` 写回 `graphs_dir`；所有本地环回请求均受 `_create_origin_only_middleware()` 跨站校验保护。
- **实时训练与调速**：
  - `POST /api/train/start` 支持读取磁盘文件或直接接收画布当前的内存 `graph` JSON 启动 `LiveTrainRunner` 后台线程。
  - 每步前向/反传后提取各节点输出、Tap 与梯度观测边的 `shape`、`mean`、`std`、`norm`（`node_stats` / `tap_stats` / `grad_stats`），经线程安全 `asyncio.Queue` 投递至 `_broadcaster` 任务推送到 `/ws`。
  - `POST /api/train/speed` 在训练途中实时修改 `runner.step_delay_ms`（0~3000ms）与 `paused` 状态，实现“全速训练”与“慢放观察张量流动”自由切换。

---

## 4. 关键工程约定（交接必读）

1. **Python 运行环境**：当前开发主机使用 `~/miniconda3/envs/yuki/bin/python`（含 `torch 2.13.0+cpu`、`pydantic`、`pyyaml`、`aiohttp`、`pytest`）。
2. **Meta 初始化安全**：在 `init_utils.apply_init()` 中必须保留 `device.type != "meta"` 判断；新增带权重初始化的元件时应复用 `apply_init()`，避免在 Meta 推导期触发非法 tensor 写入。
3. **无 torchvision 依赖原则**：`MNISTData` 通过标准库 `urllib` + `gzip` + `struct` 直接解析 idx-ubyte 缓存至 `.cache/mnist`，保证核心运行时极简。
1. **自定义元件规范**：在 `plugins/*.py` 或 `plugins/*/__init__.py` 中继承 `BaseComponent`，使用 `@register_component("<type_id>")` 装饰；`declare_ports(params)` 只能依赖 `params`，不得依赖输入张量维度；可选覆盖 `param_schema()`、`display_name`、`description` 供 WebUI 表单渲染，`build(params, in_specs)` 根据 `in_specs` 构建 `nn.Module`；Loss 类插件可覆盖 `compute_metrics(params, inputs)` 接入 `Trainer` / `WebUI` 准确率等指标聚合。
5. **验收命令**：
   - 全量测试：`~/miniconda3/envs/yuki/bin/python -m pytest tests -v`（当前 13 项全通过）。
   - 启动 WebUI：`~/miniconda3/envs/yuki/bin/python -m netpulse.cli serve --host 127.0.0.1 --port 8765`。
