"""Search as Code (SaC) — agentic search orchestration.

Uses OpenAI-compatible API (opencode Zen by default).

Usage:
    sac "Your research task here"
    sac -v "Verbose research"
    sac --endpoint https://opencode.ai/zen/v1 --model big-pickle "task"
    sac --final-report ./report.md "task"
    sac                         # interactive mode

Requires:
    pip install sac
    export OPENAI_API_BASE="https://opencode.ai/zen/v1"   # default
    export OPENAI_API_KEY="public"                         # default
    export SAC_MODEL="big-pickle"                          # default

Reference:
    Perplexity AI, "Rethinking Search as Code Generation" (2026)
    https://research.perplexity.ai/articles/rethinking-search-as-code-generation
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from sac import core
from sac.agent import SaCAgent
from sac.sdk import AgenticSearchSDK

DEFAULT_BASE_URL = "https://opencode.ai/zen/v1"
DEFAULT_API_KEY = "public"
DEFAULT_MODEL = "big-pickle"

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search as Code — agentic research orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Reference: Perplexity AI, 'Rethinking Search as Code Generation' (2026)\n"
            "https://research.perplexity.ai/articles/rethinking-search-as-code-generation"
        ),
    )
    parser.add_argument("task", nargs="*", help="Research task description")
    parser.add_argument(
        "--endpoint",
        default=None,
        help=f"OpenAI-compatible API endpoint (default: $OPENAI_API_BASE or {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model name (default: $SAC_MODEL or {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=f"API key (default: $OPENAI_API_KEY or {DEFAULT_API_KEY})",
    )
    parser.add_argument(
        "--final-report",
        default=None,
        help="Path or directory for final report (default: pwd/<task>_synthesis.md)",
    )
    parser.add_argument(
        "--final-report-format",
        default="md",
        choices=["md", "txt", "json"],
        help="Report format (default: md)",
    )
    parser.add_argument(
        "--http-proxy",
        default=None,
        help="HTTP proxy URL (default: $HTTP_PROXY or http://127.0.0.1:8118)",
    )
    parser.add_argument(
        "--https-proxy",
        default=None,
        help="HTTPS proxy URL (default: $HTTPS_PROXY or --http-proxy value)",
    )
    parser.add_argument(
        "--with-code-library",
        action="store_true",
        help="Save generated code snippets to ~/.cache/sac-agent/ as reusable functions",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=6,
        help="Maximum research iterations before synthesis (default: 6)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def _resolve_report_path(report_arg: str, task: str, fmt: str) -> Path:
    p = Path(report_arg)
    ext = f".{fmt}"
    if p.is_dir() or (not p.suffix and not p.exists()):
        safe_name = _safe_filename(task)[:64]
        return p / f"{safe_name}_synthesis{ext}"
    return p.with_suffix(ext)


def _safe_filename(text: str) -> str:
    return re.sub(r"[^\w.-]", "_", text.strip().replace(" ", "_"))


def _write_report(path: Path, content: str, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        data = {"answer": content}
        path.write_text(json.dumps(data, indent=2))
    else:
        path.write_text(content)
    console.print(f"\n[green]Report saved to:[/] {path}")


def _execute(task: str, args: argparse.Namespace) -> str:
    base_url = args.endpoint or os.environ.get("OPENAI_API_BASE") or DEFAULT_BASE_URL
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or DEFAULT_API_KEY
    model = args.model or os.environ.get("SAC_MODEL") or DEFAULT_MODEL
    brave_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    http_proxy = args.http_proxy
    https_proxy = args.https_proxy

    sdk = AgenticSearchSDK(
        llm_base_url=base_url,
        llm_api_key=api_key,
        llm_model=model,
        brave_key=brave_key,
        http_proxy=http_proxy,
        https_proxy=https_proxy,
    )
    agent = SaCAgent(
        task=task,
        sdk=sdk,
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_turns=args.max_turns,
        http_proxy=http_proxy,
        https_proxy=https_proxy,
        with_code_library=args.with_code_library,
    )
    return agent.run()


def main() -> None:
    parser = build_parser()
    parsed = parser.parse_args()

    if parsed.verbose:
        core.VERBOSE = True

    if not parsed.task:
        interactive()
        return

    task = " ".join(parsed.task)

    if parsed.verbose:
        console.print(
            f"[dim]Endpoint: {parsed.endpoint or os.environ.get('OPENAI_API_BASE') or DEFAULT_BASE_URL}[/]"
        )
        console.print(
            f"[dim]Model: {parsed.model or os.environ.get('SAC_MODEL') or DEFAULT_MODEL}[/]"
        )

    answer = _execute(task, parsed)

    if parsed.final_report is not None:
        path = _resolve_report_path(
            parsed.final_report, task, parsed.final_report_format
        )
        _write_report(path, answer, parsed.final_report_format)


def interactive() -> None:
    parser = build_parser()
    parsed = parser.parse_args([])

    console.print("[bold green]Search as Code (SaC)[/] — interactive mode")
    console.print(
        "[dim]Set BRAVE_SEARCH_API_KEY for Brave search, otherwise Exa MCP (free) or simulation.[/]\n"
    )

    while True:
        task = Prompt.ask("\n[bold]Research task[/]")
        if task.lower() in ("quit", "exit", "q"):
            break
        if not task.strip():
            continue

        answer = _execute(task, parsed)

        if parsed.final_report is not None:
            path = _resolve_report_path(
                parsed.final_report, task, parsed.final_report_format
            )
            _write_report(path, answer, parsed.final_report_format)
