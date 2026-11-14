# 开发日志

## 2026-11-14
- 确定 NetPulse 项目初步定位：ComfyUI 式节点流工作流 + EDA（嘉立创）式元件封装与电气规则检查（ERC），底层编译为原生 PyTorch `nn.Module`。
- 新建初版架构设计文档 `docs/design.md`，涵盖 5 层架构、`Signal`/`PortSpec`/`TapSpec`/`GraphIR` 协议、Meta Device 静态形状推导、Hook 隐层抽头（Tap）、多目标正则化汇集、数据源策略及 Phase 0~2 路线图。
- 当前状态：处于设计定稿与开坑准备阶段，尚未初始化 Python 代码与环境。

## 2026-11-14（Phase 0 最小闭环实现与 MNIST 可行性验证）
- 完成 `netpulse` 内核：`Signal` / `PortSpec` / `TapSpec` / `GraphIR` / `BaseComponent` / `ComponentRegistry`（含 `plugins/` 扫描）。
- 完成 ERC 静态检查（环检测、端口校验、`meta` device 零显存形状推导、`AutoBridge` 自动矩阵桥接）、`GraphCompiler`（输出 `CompiledGraph(nn.Module)`，支持 `forward_with_taps` 与 `register_forward_hook` 双模式抽头及多 Loss 汇总）与 `Trainer`。
- 完成最小内置元件库：`SpiralData`、`MNISTData`（纯 PyTorch 下载与缓存，无 torchvision 依赖）、`CNNBlock`、`MLPBlock`、`Flatten`、`GlobalAvgPool`、`MatrixBridge`、`TensorAdd`、`CrossEntropyLoss`、`ActivationRegLoss`（`l1`/`l2`/`orthogonal`）。
- 验证通过：`examples/spiral_quickstart.json`（val_acc=1.0000）、`examples/mnist_tap_reg.json`（CNN + 第2层 Tap + GlobalAvgPool + MatrixBridge + 正交正则与交叉熵联合训练，val_acc=0.9160 并导出 `.pt`），`pytest -v` 7/7 全部通过。

## 2026-11-14（双粒度元件库扩展：原子层 + 统一初始化）
- 新增 `netpulse/components/init_utils.py`：统一权重/偏置初始化（`kaiming_normal`/`uniform`、`xavier_normal`/`uniform`、`orthogonal`、`normal`、`uniform`、`zeros`/`ones`/`constant`），自动跳过 `meta` device。
- 新增 `netpulse/components/layers.py`（7 个原子层元件）：`conv2d`、`pool2d`、`linear`、`activation`、`batch_norm2d`、`layer_norm`、`dropout`，支持显式输入尺寸校验（可触发 ERC / AutoBridge）或自适应推导。
- 增强 `netpulse/components/backbones.py`：`cnn_block` 支持逐层 `layers` 配置或逐层参数数组与 `init`/`bias_init`；`mlp_block` 支持 `init`/`bias_init`。
- 新增全原子层拼装示例 `examples/mnist_atomic.json`（15 节点，`conv2d -> activation -> pool2d -> conv2d -> batch_norm2d -> activation -> pool2d -> flatten -> linear -> activation -> dropout -> linear(10)` + 正交正则），验证跑通 `val_acc=0.9629`，`pytest -v` 10/10 全部通过。

## 2026-11-14（外挂 WebUI 可视化界面）
- 新增 `netpulse/webui/`（基于 `aiohttp` + 原生 WebSocket + 单文件原生 HTML/SVG/JS 前端 `index.html`，零前端构建依赖）：
  - `server.py`：提供 `/api/graphs`、`/api/graph`（自动拓扑分层排布 + ERC 状态与每条边推导维度）、`/api/train/start`、`/api/train/speed`、`/api/train/stop` 与 `/ws` 实时遥测广播。
  - `runner.py`：`LiveTrainRunner` 支持后台训练、动态调节步进延迟（0~800ms+）、暂停/继续/停止、分项 Loss 与每节点激活 `norm/mean/std` 采集。
  - `index.html`：支持画布拖拽/缩放/平移、元件分类色条、实线/Tap 虚线/AutoBridge 连线、边粒子数据流动画与实时 Loss 曲线图。
- 在 `netpulse/cli.py` 新增 `serve` 子命令；`pytest -v`（含 `tests/test_webui.py`）11/11 全部通过。

## 2026-11-14（文档同步与仓库初始化）
- 更新 `docs/design.md` 至 v0.2，同步双粒度元件库、Meta/AutoBridge 协议与外挂 WebUI 现状。
- 新增 `docs/ARCHITECTURE.md`（完整分层架构、目录职责、编译/运行期数据流与交接约定）。
- 初始化 Git 仓库，新增 `README.md`、`LICENSE`（GNU AGPL-3.0）与 `.gitignore`，完成初始版本提交。
