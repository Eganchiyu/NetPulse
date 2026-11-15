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

## 2026-11-14（ComfyUI 代码复用评估）
- 浅克隆 `Comfy-Org/ComfyUI` 至工作区 `reference_comfyui/`（未纳入 git 追踪）。
- 完成逐模块对比评估：核心执行引擎（解释型推理+缓存）与 NetPulse（`GraphCompiler` 编译为 `nn.Module` + `meta` ERC + `autograd` 训练）不兼容，不可照抄；外围基础设施（`server.py` 安全中间件与 `/object_info` 协议、`PromptQueue` 队列管理、`load_custom_node` 插件隔离加载、`graph_utils.add_graph_prefix` 子图展开、`progress.py` 进度回调抽象）及 GPL-3.0 许可证均可借鉴/改写复用。

## 2026-11-14（借鉴 ComfyUI 升级 WebUI 为可编辑 IDE 及底层协议适配）
- 核心协议扩展：
  - `netpulse/core/types.py`、`component.py`、`ir.py` 新增 `ParamSpec` 参数契约、`BaseComponent.param_schema()`/`display_name`/`description`、`GraphIR.to_file()`；为全部 17 个内置元件补齐参数 Schema。
  - `netpulse/core/registry.py` 新增 `describe_component()` / `describe_all()` 元数据导出，并增强 `load_plugins()` 支持包目录与单插件异常隔离。
- WebUI 后端适配（`netpulse/webui/server.py`）：新增 `/api/components`、`/api/erc/check`（内存图实时 ERC 校验与自动排布）、`/api/graph/save`（含路径穿越防护）、支持直接传入内存 `GraphIR` 启动训练，并加入借鉴自 ComfyUI 的本地环回防跨站中间件与 `web.AppKey` 规范。
- WebUI 前端升级（`netpulse/webui/index.html`）：新增左侧可折叠元件库面板、双击画布快速搜索建点、端口拖拽实时连线（含 Tap 虚线）、动态参数表单检查器（Inspector）、节点/连线删除与 `AutoBridge` 切换、撤销/重做（`Ctrl+Z/Y`）、自动排版、保存/另存为与右下角 Minimap。
- 验证通过：`pytest tests -v` 13/13 全部通过。

## 2026-11-14（文档与 README 启动说明同步）
- 同步更新 `README.md`、`docs/design.md` 与 `docs/ARCHITECTURE.md`，补充交互式 WebUI、元件 Schema 协议与完整 CLI/WebUI 启动命令。

## 2026-11-14（输出监视工具节点与内嵌可视化）
- 协议与编译器适配：`netpulse/core/types.py`、`compiler.py` 新增 `"output"` 类别与 `"any"` 端口类型；`CompiledGraph` 对 `output` 节点以 `torch.no_grad()` 零梯度旁路执行（上游 `loss` 跳过时安全跳过）。
- 新增输出工具元件库（`netpulse/components/outputs.py`，并在 `__init__.py` 注册）：`monitor`（通用自适应监视器）、`scalar_watch`（标量数值与 EMA 曲线）、`heatmap_preview`（2D/特征图多色标热力图）、`tensor_peek`（原始浮点切片表格、16-bin 直方图与 4D 图像网格预览）。
- WebUI 遥测与前端渲染：`runner.py` 节流推送 `probes` 快照，`server.py` 补充 `in_specs`；`index.html` 支持在画布节点卡片内直接绘制 Canvas 曲线/热力图/图像与原始数据表，支持悬停读数与点击 `⤢` 放大弹窗切换视图。
- 验证通过：`pytest tests -v` 16/16 全部通过，`node --check` 前端脚本语法无误。

## 2026-11-14（README 部署运行说明去本机化）
- 更新 `README.md`“部署与运行”部分：移除本机 Conda 解释器路径 `~/miniconda3/envs/yuki/bin/python`，改为通用的虚拟环境创建、依赖安装与 WebUI/CLI/测试运行说明。

## 2026-11-14（LSTM 序列套件与专属可视化分析探针）
- 底层 `Signal` 序列适配：`netpulse/core/trainer.py` 支持 `BLD` 布局与变长 `mask` 透传；`sources.py` 新增 `_SourceModule.sample_has_mask` 支持 `meta` 推导。
- 新增序列与循环元件：
  - `seq_wave_data`（`sources.py`）：合成序列分类数据源（`memory_cue` / `freq_pattern`），支持变长 `mask`。
  - `image_to_seq` 与 `seq_pool`（`ops.py`）：`BCHW -> BLD` 图像序列切片、`BLD -> BD` 感知 Mask 池化（`last/mean/max`）。
  - `lstm`（`backbones.py`）：基于步进 `LSTMCell` 的可观测门控 LSTM，支持变长 Mask 状态冻结、`last_h/last_c` 输出及 `gate_i/gate_f/gate_o/cell_state/hidden_states` 抽头。
- 新增 3 个专属可视化分析探针（`outputs.py` + `webui/index.html` Canvas 渲染）：`gate_dynamics`（多门控时序曲线 + 神经元条带热图 + 饱和率）、`hidden_trajectory`（PCA/first2 2D 相空间演化轨迹）、`step_energy`（时间步响应 L2 能量与步间跃迁）。
- 新增示例工作流 `examples/lstm_seq_observatory.json` 与测试 `tests/test_lstm_and_probes.py`，`pytest tests -v` 20/20 全部通过。

## 2026-11-15（工作流文件导入导出、模板库与 LSTM 经典时序预测案例）
- 新增元件适配回归预测：`sources.py` 扩展 `_SourceModule`（支持 `y_dtype`/`y_layout`）并新增 `seq_forecast_data`（`sine_mix` / `damped_osc` / `ar_trend` 滑窗时间序列回归数据源）；`losses.py` 新增 `mse_loss`（支持 ERC `expected_input_spec` 形状对齐检查）；`trainer.py` 适配多维 `y` 的 `BD` 布局推导。
- 新增 `templates/` 模板目录（5 个经典模板：`01_spiral_mlp.json`、`02_mnist_cnn_atomic.json`、`03_mnist_resnet_bridge.json`、`04_seq_mnist_lstm.json`、`05_lstm_time_series_forecast.json`）与经典案例 `examples/lstm_forecast_classic.json`（`seq_forecast_data -> lstm -> layer_norm -> mlp_block -> mse_loss` + 门控/相空间/步能量/标量探针）。
- WebUI 文件与模板管理：`server.py` 与 `cli.py` 新增 `--templates-dir`、`/api/templates`、`/api/template`、`/api/graph/delete`、`/api/graph/import`、`/api/graph/export`；`index.html` 新增“新建”、“模板”弹窗、“导入”（含拖放 `.json`/`.yaml` 文件）、“导出”（JSON/YAML 下载）与“删除”功能。
- 验证通过：`pytest tests -v` 22/22 全部通过。

## 2026-11-16（元件中文元数据、可编辑性与文字排版）
- 元件元数据中文化：`types.py` 的 `PortSpec` 新增 `title`（中文含义）/`description`（端口简介）；`component.py` 的 `BaseComponent` 新增 `title`（中文译名）。30 个内置元件的 `declare_ports` 全部补齐端口中文名与说明，`registry.describe_component` 导出 `title` 及各端口 `title/description`（插件新元件同样生效）。
- 节点可编辑性：`NodeIR` 新增 `label`（自定义显示名），`serialize_graph_ir` 透传 `title/label`；画布节点卡片新增 ✎ 重命名与 ⚙ 展开内联「快速参数编辑」面板（保持展开状态、改动即触发 ERC）。
- 参数表单增强：抽出共用 `paramFieldHtml / bindParamForm`，支持逐项「恢复默认值」、数值范围与 JSON 语法校验提示、JSON 改多行文本域。
- 检查器改版：新增元件信息卡（中文名/英文名/类别/简介）、端口说明表（中文含义+英文端口名+布局+实时 shape）、参数分卡排版；连线检查器显示源/宿端口中文名。
- 文字视觉：统一中文标签（`.zh-name/.en-name/.hint-text/.pf-*`），元件库/节点卡/表格字号与行高重排，节点卡简介两行截断。
- 验证：`pytest tests -q` 22/22 通过；jsdom 冒烟测试（元件库、节点渲染、LSTM 抽头、改名、快速参数、校验与重置、连线检查器）无 JS 报错。
- 文档同步：`README.md` 更新元件库为 30 个（按 source/backbone/op/bridge/loss/output 分类）、补充序列数据源与可视化探针、新增「元件自带中文元数据」与「中文优先可视排版」章节、WebUI 常用操作加入 ✎ 改名 / ⚙ 快速参数 / 参数重置与校验 / 模板库、目录结构补 `templates/`、serve 参数补 `--templates-dir`。
