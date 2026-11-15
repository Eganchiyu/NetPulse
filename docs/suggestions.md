### 一、 批判性审视：当前架构的 5 个盲区与隐患

1. **缺少 `stop_gradient`（梯度阻断），导致“探针观测”与“训练干扰”耦合**
* **问题定位**：在 `netpulse/core/compiler.py` 中，`tap_bus` 传出的 `Signal` 完整保留了计算图（`requires_grad=True`）。当你把某个隐藏层连到 `activation_reg_loss` 时，正则项的梯度会反传回主干，这是预期行为；但如果你想在训练过程中挂一个在线线性探针（Linear Probe）来评估中间层的表征线性可分性，探针的交叉熵梯度会直接污染主干权重。


* **改进建议**：在 `EdgeIR`（`netpulse/core/ir.py`）增加 `stop_grad: bool = False`（或在 `ops.py` 增加一个 `stop_grad` 算子）。在 `CompiledGraph.forward` 取边信号时判断 `if edge.stop_grad: sig = sig.clone_with(data=sig.data.detach())`，画布连线上用截断符号标出。改动不到 10 行，却能把“特征干预”和“无侵入表征评估”彻底解耦。




2. **只有前向激活探针，完全缺失“反向梯度流”观测**
* **问题定位**：`netpulse/components/outputs.py` 里的 7 个探针全都在 `forward` 阶段通过 `detach()` 抓取前向激活。但 NetPulse 的核心定位之一是**架构实验与多目标正则化**——在多目标训练（比如主任务 Loss + 激活正交正则）中，真正决定训练成败的往往不是前向激活值，而是**梯度消失/爆炸**以及**不同 Loss 分支之间的梯度冲突（Gradient Conflict）**。


* **改进建议**：在 `CompiledGraph` 中仿照 `tap_bus` 引入 `grad_bus`，利用 `tensor.register_hook` 或模块的 `register_full_backward_hook` 捕获指定层的反向梯度张量。这样无需新增复杂逻辑，现有的 `scalar_watch` 和 `heatmap_preview` 就能直接可视化各层的梯度范数和梯度分布。




3. **画布拓扑缺乏“子图封装（Subgraph / Group Node）”，复杂度一高就会崩**
* **问题定位**：看 `examples/mnist_atomic.json`，仅仅一个 2 层卷积的原子化网络加上探针，就已经有 25 个节点，画布 X 坐标拉到了 `4225px`。一旦尝试搭 ResNet 残差块或者多层 Transformer，画布会立刻变成难以阅读的长蛇阵。


* **改进建议**：既然你之前调研过 ComfyUI 的 `add_graph_prefix` 机制，下一步在交互层最该补的是**子图折叠**：允许框选一组节点封装为一个复合节点（内部保存为嵌套 `GraphIR`，对外暴露指定的输入/输出/Tap 端口），编译期自动展开进主拓扑。




4. **ERC 编译期与 Trainer 运行期存在几处契约裂缝**
* **Batch 维度脱节**：`sources.py` 在 Meta 推导期默认使用 `meta_batch_size`（32 或 64），没有读取 `GraphIR.trainer.batch_size`。如果用户在属性面板里把 `batch_size` 改成 `1`，`BatchNorm2d` 在训练模式会直接报错，`activation_reg_loss(mode="orthogonal")` 里的批次内中心化（`centered = feat - feat.mean(dim=0)`）也会退化为零张量，但 ERC 预检阶段会显示 `PASS`。


* **单 Source 硬限制**：`netpulse/core/trainer.py` 里的 `_find_source_id()` 写死了 `len(sources) == 1`，这堵死了双流对比学习（如 SimCLR 两路增强视图）或教师-学生蒸馏实验的可能。


* **准确率统计硬编码**：`trainer.py` 的 `_extract_acc()` 硬编码寻找 `cross_entropy_loss` 节点来算分类准确率，如果用户通过 `plugins/` 写了一个自定义的 FocalLoss 或 LabelSmoothingLoss，验证集准确率指标就会直接消失。




5. **几处顺手可修的工程细节**
* **`AutoBridge` 过于“隐身”**：`compiler.py` 自动插入的 `_AutoBridgeModule` 没有调用 `init_utils.apply_init()`，且在画布上只是一个连线标签，用户无法查看这个自动投影层的参数量、权重范数或单独给它加正则化。


* **前端尺寸常量不同步**：`netpulse/webui/index.html` 顶部的 `const NODE_W = 228;` 还是旧值，但下方 CSS 已经把 `.node-card` 改成了 `236px`、`output` 探针改成了 `276px`，导致 `fitView()` 和右下角 `renderMinimap()` 的包围盒计算略微偏窄。另外 `index.html` 已经膨胀到上千行，后续再加新可视化视图建议把 Canvas 渲染器拆成独立 JS 模块。