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
- **双粒度元件库（17 个内置元件）与统一初始化**：
  - **数据源 (`source`)**：`spiral_data`、`mnist_data`（纯 PyTorch 加载与缓存，无需 `torchvision`）。
  - **原子层 (`backbone`/`op`)**：`conv2d`、`pool2d`、`linear`、`activation`、`batch_norm2d`、`layer_norm`、`dropout`（支持 `kaiming` / `xavier` / `orthogonal` / `normal` / `constant` 等权重与偏置初始化）。
  - **宏模块 (`backbone`)**：`cnn_block`、`mlp_block`。
  - **算子与桥接 (`op`/`bridge`)**：`flatten`、`global_avg_pool`、`matrix_bridge`、`tensor_add`。
  - **目标 (`loss`)**：`cross_entropy_loss`、`activation_reg_loss`。
  - 支持在 `plugins/*.py` 中通过 `@register_component` 热加载自定义元件。
- **外挂可视化 WebUI**：基于 `aiohttp` + WebSocket 的轻量画布，支持查看拓扑连线与 ERC 推导维度、启停训练、动态调节步进延迟（慢放观察张量流动脉冲）与实时分项 Loss 曲线。

## 目录结构

```text
netpulse/
├── core/           # Signal / PortSpec / GraphIR / Registry / ERC / GraphCompiler / Trainer
├── components/     # 内置双粒度元件库与统一初始化工具
├── webui/          # 外挂可视化服务端 (aiohttp + WS) 与单文件前端
└── cli.py          # 命令行入口 (list / check / train / serve)
examples/           # 示例计算图 (spiral_quickstart / mnist_tap_reg / mnist_atomic)
tests/              # 单元测试与端到端训练 / WebUI 测试
docs/design.md      # 架构设计文档
```

## 快速开始

依赖：`Python >= 3.10`、`torch`、`pydantic`、`pyyaml`、`aiohttp`、`pytest`（测试）。

```bash
# 1. 列出已注册元件
python -m netpulse.cli list

# 2. 执行 ERC 静态检查与 Meta 形状推导
python -m netpulse.cli check examples/mnist_atomic.json

# 3. 编译并运行训练闭环（支持导出 .pt 权重）
python -m netpulse.cli train examples/mnist_atomic.json --save .cache/mnist_atomic.pt

# 4. 启动外挂 WebUI 可视化界面
python -m netpulse.cli serve --host 127.0.0.1 --port 8765
```

运行全部测试：

```bash
python -m pytest -v
```

## 开源协议

本项目采用 [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE) 协议开源。
