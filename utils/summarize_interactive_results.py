#!/usr/bin/env python3
"""
Process raw benchmark result JSONs from the interactive script and summarize into a markdown table.

Follows the same processing flow as benchmark-tmpl.yml and e2e-tests.yml:
1. Process each result_*.json (raw benchmark_serving output) into agg format
2. Collect all combinations
3. Output markdown table via tabulate

Usage:
    python utils/summarize_interactive_results.py <result_dir> [options]

Examples:
    python utils/summarize_interactive_results.py /workspace/
    python utils/summarize_interactive_results.py ./results -o summary.md
    python utils/summarize_interactive_results.py ./results --hw mi355x --model-prefix qwen3.5
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


# Filename pattern: result_TP{TP}_CONC{CONC}_ISL{ISL}_OSL{OSL}.json
RESULT_PATTERN = re.compile(
    r"result_TP(\d+)_CONC(\d+)_ISL(\d+)_OSL(\d+)\.json", re.IGNORECASE
)


def parse_result_filename(path: Path) -> Optional[tuple[int, int, int, int]]:
    """Extract TP, CONC, ISL, OSL from filename. Returns (tp, conc, isl, osl) or None."""
    m = RESULT_PATTERN.match(path.name)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return None


def process_raw_result(
    raw: dict[str, Any],
    tp: int,
    conc: int,
    isl: int,
    osl: int,
    *,
    hw: str = "mi355x",
    model_prefix: str = "qwen3.5",
    framework: str = "sglang",
    precision: str = "bf16",
    image: str = "",
) -> dict[str, Any]:
    """
    Convert raw benchmark_serving JSON to agg format (same schema as process_result.py output).
    """
    model_id = raw.get("model_id", "unknown")
    total_tput = float(raw.get("total_token_throughput", 0))
    output_tput = float(raw.get("output_throughput", 0))
    max_conc = int(raw.get("max_concurrency", conc))

    data: dict[str, Any] = {
        "hw": hw,
        "conc": max_conc,
        "image": image,
        "model": model_id,
        "infmax_model_prefix": model_prefix,
        "framework": framework,
        "precision": precision,
        "spec_decoding": "none",
        "disagg": False,
        "isl": isl,
        "osl": osl,
        "is_multinode": False,
        "tp": tp,
        "ep": 1,
        "dp_attention": False,
        "tput_per_gpu": total_tput / tp,
        "output_tput_per_gpu": output_tput / tp,
        "input_tput_per_gpu": (total_tput - output_tput) / tp,
    }

    # Convert *_ms to seconds (process_result logic)
    for key, value in raw.items():
        if key.endswith("_ms") and isinstance(value, (int, float)):
            data[key.replace("_ms", "")] = float(value) / 1000.0
        if "tpot" in key and isinstance(value, (int, float)) and value > 0:
            data[key.replace("_ms", "").replace("tpot", "intvty")] = 1000.0 / float(value)

    # Ensure required fields for summarize (fallbacks if raw uses different keys)
    if "median_ttft" not in data and "median_ttft_ms" in raw:
        data["median_ttft"] = float(raw["median_ttft_ms"]) / 1000.0
    if "median_tpot" not in data and "median_tpot_ms" in raw:
        data["median_tpot"] = float(raw["median_tpot_ms"]) / 1000.0
    if "median_e2el" not in data and "median_e2el_ms" in raw:
        data["median_e2el"] = float(raw["median_e2el_ms"]) / 1000.0
    if "median_intvty" not in data and "median_tpot_ms" in raw:
        data["median_intvty"] = 1000.0 / float(raw["median_tpot_ms"])
    data.setdefault("median_ttft", 0.0)
    data.setdefault("median_tpot", 0.0)
    data.setdefault("median_e2el", 0.0)
    data.setdefault("median_intvty", 0.0)

    return data


def load_and_process(
    result_dir: Path,
    *,
    hw: str = "mi355x",
    model_prefix: str = "qwen3.5",
    framework: str = "sglang",
    precision: str = "bf16",
    image: str = "",
) -> list[dict[str, Any]]:
    """Load all result_*.json files, process, and return agg records."""
    agg_results: list[dict[str, Any]] = []

    for path in sorted(result_dir.rglob("*.json")):
        parsed = parse_result_filename(path)
        if not parsed:
            continue

        tp, conc, isl, osl = parsed
        try:
            with open(path) as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Skipping {path}: {e}", file=sys.stderr)
            continue

        if "total_token_throughput" not in raw or "model_id" not in raw:
            print(f"Warning: Skipping {path}: missing required fields", file=sys.stderr)
            continue

        agg = process_raw_result(
            raw,
            tp,
            conc,
            isl,
            osl,
            hw=hw,
            model_prefix=model_prefix,
            framework=framework,
            precision=precision,
            image=image,
        )
        agg_results.append(agg)

    return agg_results


def to_markdown_table(results: list[dict[str, Any]]) -> str:
    """Format results as markdown table (same columns as summarize.py single-node)."""
    if not results:
        return "No results to display.\n"

    results.sort(
        key=lambda r: (
            r["infmax_model_prefix"],
            r["hw"],
            r["framework"],
            r["precision"],
            r["isl"],
            r["osl"],
            r["tp"],
            r["ep"],
            r["conc"],
        )
    )

    headers = [
        "Model",
        "Served Model",
        "Hardware",
        "Framework",
        "Precision",
        "ISL",
        "OSL",
        "TP",
        "EP",
        "DP Attention",
        "Conc",
        "TTFT (ms)",
        "TPOT (ms)",
        "Interactivity (tok/s/user)",
        "E2EL (s)",
        "TPUT per GPU",
        "Output TPUT per GPU",
        "Input TPUT per GPU",
    ]

    rows = [
        [
            r["infmax_model_prefix"],
            r["model"],
            r["hw"].upper(),
            r["framework"].upper(),
            r["precision"].upper(),
            r["isl"],
            r["osl"],
            r["tp"],
            r["ep"],
            r["dp_attention"],
            r["conc"],
            f"{r['median_ttft'] * 1000:.4f}",
            f"{r['median_tpot'] * 1000:.4f}",
            f"{r['median_intvty']:.4f}",
            f"{r['median_e2el']:.4f}",
            f"{r['tput_per_gpu']:.4f}",
            f"{r['output_tput_per_gpu']:.4f}",
            f"{r['input_tput_per_gpu']:.4f}",
        ]
        for r in results
    ]

    if tabulate:
        return tabulate(rows, headers=headers, tablefmt="github")
    # Fallback: simple markdown table
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    header_row = "|" + "|".join(headers) + "|"
    data_rows = "\n".join("|" + "|".join(str(c) for c in row) + "|" for row in rows)
    return f"{header_row}\n{sep}\n{data_rows}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process interactive benchmark results and output markdown table"
    )
    parser.add_argument(
        "result_dir",
        type=Path,
        help="Directory containing result_TP*_CONC*_ISL*_OSL*.json files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write markdown to file (default: stdout)",
    )
    parser.add_argument(
        "--hw",
        type=str,
        default="mi355x",
        help="Hardware label (default: mi355x)",
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="qwen3.5",
        help="Model prefix (default: qwen3.5)",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="sglang",
        help="Framework (default: sglang)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        help="Precision (default: bf16)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Docker image (optional)",
    )
    parser.add_argument(
        "--agg-json",
        type=Path,
        default=None,
        help="Also write aggregated JSON to file (for collect_results compatibility)",
    )

    args = parser.parse_args()

    if not args.result_dir.is_dir():
        print(f"Error: {args.result_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    results = load_and_process(
        args.result_dir,
        hw=args.hw,
        model_prefix=args.model_prefix,
        framework=args.framework,
        precision=args.precision,
        image=args.image,
    )

    if not results:
        print("No result files found matching result_TP*_CONC*_ISL*_OSL*.json", file=sys.stderr)
        sys.exit(1)

    md = to_markdown_table(results)
    full_output = f"## Interactive Benchmark Results\n\n{md}\n"

    if args.output:
        args.output.write_text(full_output)
        print(f"Wrote summary to {args.output}", file=sys.stderr)
    else:
        print(full_output)

    if args.agg_json:
        with open(args.agg_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote aggregated JSON to {args.agg_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
