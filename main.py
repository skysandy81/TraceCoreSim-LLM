"""
项目入口，两种模式：训练仿真、计算过程演示
entry point：training simulation and trace demo.
"""
import argparse
from coresim_llm.trace_demo import format_trace_text, write_trace_html
from coresim_llm.train import run_training

def main():
    """
    默认运行完整训练流程
    `--trace` 输出 d_model=8 的逐步数值; `--trace-html` 生成可视化网页。
    By default this runs the full staged training flow.
    `--trace` prints a d_model=8 numeric trace
    `--trace-html` writes an interactive HTML view of the same trace.
    """
    parser = argparse.ArgumentParser(description="CoreSimLLM training simulation and trace demo.")
    parser.add_argument("--trace", action="store_true", help="Print a numeric d_model=8 forward-pass trace.")
    parser.add_argument("--trace-html", action="store_true", help="Write an interactive d_model=8 trace page.")
    parser.add_argument("--seed", type=int, default=None, help="Override the training random seed.")
    args = parser.parse_args()

    if args.trace:
        print(format_trace_text())

    if args.trace_html:
        output_path = write_trace_html()

    if not args.trace and not args.trace_html:
        # 无trace参数时执行 训练仿真 主流程
        # Without trace flags, run “training simulation”.
        run_training(show_plots=not args.no_plots, show_comparison=not args.no_comparison, run_seed=args.seed)


if __name__ == "__main__":
    main()
