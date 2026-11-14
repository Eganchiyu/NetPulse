"""NetPulse 命令行入口：ERC 检查与图编译训练。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from netpulse.core.compiler import GraphCompiler
from netpulse.core.erc import run_erc
from netpulse.core.ir import GraphIR
from netpulse.core.registry import get_default_registry
from netpulse.core.trainer import Trainer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="netpulse", description="NetPulse 计算图检查与训练 CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="列出所有已注册元件")
    p_list.add_argument("--plugins", default="plugins", help="自定义插件目录")

    p_check = sub.add_parser("check", help="执行 ERC 静态检查与 Meta 形状推导")
    p_check.add_argument("graph", help="GraphIR JSON/YAML 文件路径")
    p_check.add_argument("--plugins", default="plugins", help="自定义插件目录")

    p_train = sub.add_parser("train", help="编译计算图并运行训练闭环")
    p_train.add_argument("graph", help="GraphIR JSON/YAML 文件路径")
    p_train.add_argument("--plugins", default="plugins", help="自定义插件目录")
    p_train.add_argument("--save", default=None, help="导出 .pt 权重路径")

    p_serve = sub.add_parser("serve", help="启动外挂 WebUI 可视化界面")
    p_serve.add_argument("--host", default="127.0.0.1", help="监听地址")
    p_serve.add_argument("--port", type=int, default=8765, help="监听端口")
    p_serve.add_argument("--graphs-dir", default="examples", help="计算图目录")
    p_serve.add_argument("--templates-dir", default="templates", help="工作流模板目录")
    p_serve.add_argument("--plugins", default="plugins", help="自定义插件目录")

    args = parser.parse_args(argv)
    if args.cmd == "serve":
        from netpulse.webui.server import run_server

        run_server(
            host=args.host,
            port=args.port,
            graphs_dir=args.graphs_dir,
            plugins_dir=args.plugins,
            templates_dir=args.templates_dir,
        )
        return 0

    reg = get_default_registry(args.plugins)

    if args.cmd == "list":
        for t in reg.list_types():
            print(t)
        return 0

    graph = GraphIR.from_file(args.graph)
    if args.cmd == "check":
        res = run_erc(graph, reg)
        print(res.format_report())
        if res.passed:
            print("Resolved Output Shapes:")
            for nid in res.topo_order:
                outs = {k: tuple(v.shape) for k, v in res.node_out_specs.get(nid, {}).items()}
                taps = {k: tuple(v.shape) for k, v in res.node_tap_specs.get(nid, {}).items()}
                print(f"  {nid}: out={outs}" + (f" taps={taps}" if taps else ""))
        return 0 if res.passed else 1

    if args.cmd == "train":
        compiled = GraphCompiler.compile(graph, reg)
        print(compiled.erc_result.format_report())
        trainer = Trainer(compiled)
        result = trainer.fit(verbose=True)
        print(f"Done: steps={result.steps}, final_train_loss={result.final_train_loss:.4f}")
        if args.save:
            out_path = trainer.save_checkpoint(args.save)
            print(f"Saved checkpoint: {out_path}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
