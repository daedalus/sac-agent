"""Search as Code (SaC) — agentic search orchestration.

Uses OpenAI-compatible API (opencode Zen by default).

Usage:
    sac "Your research task here"
    sac                         # interactive mode

Requires:
    pip install sac
    export OPENAI_API_BASE="https://opencode.ai/zen/v1"   # default
    export OPENAI_API_KEY="public"                         # default
    export SAC_MODEL="big-pickle"                          # default
    # Exa MCP is free and requires no API key
"""

from __future__ import annotations

import os
import sys

from rich.console import Console
from rich.prompt import Prompt

from sac import core
from sac.agent import SaCAgent
from sac.sdk import AgenticSearchSDK

DEFAULT_BASE_URL = "https://opencode.ai/zen/v1"
DEFAULT_API_KEY = "public"
DEFAULT_MODEL = "big-pickle"

console = Console()


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return

    if "-v" in args or "--verbose" in args:
        core.VERBOSE = True
        args = [a for a in args if a not in ("-v", "--verbose")]

    if not args:
        print("Usage: sac [-v] <research task>")
        return

    task = " ".join(args)

    base_url = os.environ.get("OPENAI_API_BASE") or DEFAULT_BASE_URL
    api_key = os.environ.get("OPENAI_API_KEY") or DEFAULT_API_KEY
    model = os.environ.get("SAC_MODEL") or DEFAULT_MODEL
    brave_key = os.environ.get("BRAVE_SEARCH_API_KEY")

    sdk = AgenticSearchSDK(
        llm_base_url=base_url,
        llm_api_key=api_key,
        llm_model=model,
        brave_key=brave_key,
    )
    agent = SaCAgent(
        task=task,
        sdk=sdk,
        base_url=base_url,
        api_key=api_key,
        model=model,
    )
    agent.run()


def interactive() -> None:
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

        base_url = os.environ.get("OPENAI_API_BASE") or DEFAULT_BASE_URL
        api_key = os.environ.get("OPENAI_API_KEY") or DEFAULT_API_KEY
        model = os.environ.get("SAC_MODEL") or DEFAULT_MODEL
        brave_key = os.environ.get("BRAVE_SEARCH_API_KEY")

        sdk = AgenticSearchSDK(
            llm_base_url=base_url,
            llm_api_key=api_key,
            llm_model=model,
            brave_key=brave_key,
        )
        agent = SaCAgent(
            task=task,
            sdk=sdk,
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
        agent.run()
