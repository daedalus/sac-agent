from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, cast

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from sac.core import _httpx_client, image_to_data_uri
from sac.library import CodeLibrary
from sac.models import ModelLimits
from sac.sandbox import Sandbox
from sac.sdk import AgenticSearchSDK

DEFAULT_CONTEXT_LIMIT = 128_000


@dataclass
class UsageTracker:
    prompt: int = 0
    completion: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion


SYSTEM_PROMPT_TEMPLATE = """You are an expert research agent using the Search as Code SDK.

You have access to an `sdk` object with these methods:
- sdk.search.web(query, limit=8) -> list[SearchResult]
- sdk.search.web_many(queries, limit_per_query=8, concurrency=6) -> list[list[SearchResult]]
- sdk.llm.synthesize(items, instruction) -> str
- sdk.llm.plan(context, goal) -> str
- sdk.llm.extract_many(items, instruction, schema) -> list[dict]
- sdk.vision.analyze(image, prompt) -> str
- sdk.vision.analyze_url(url, prompt) -> str
- sdk.fs.write(key, data), .read(key), .list(), .exists(key)
- sdk.utils.dedupe_by(items, key), .filter_by(items, field, value)
- sdk.utils.summarize_coverage(items, by_fields)
- sdk.utils.flatten(list_of_lists), .join_result_fields(result)

Context budget: {context_limit:,} tokens · used {pct_used:.0f}% · {remaining:,} remaining.
The feedback message after each turn will update this usage.

Protocol: respond with ONE JSON object (no markdown fences).
- Code turn: {{"turn_type": "code", "reasoning": "...", "code": "python code using sdk"}}
- Synthesis turn: {{"turn_type": "synthesis", "reasoning": "...", "answer": "final answer with ## References section"}}

Your synthesis answer MUST end with a "## References" section listing every URL
you used as evidence throughout the research. Use the actual URLs from search
results you stored in sdk.fs. Format as a markdown bullet list.

You have a limited turn budget (max_turns). The context will tell you
how many turns remain. Plan accordingly.

Strategy:
1. Fan out many parallel queries first (web_many for top speed)
2. Read sdk.fs.list() on later turns to see persisted state
3. Do gap analysis, backfill, then structured extraction
4. Synthesize well before turns run out. Do NOT repeat yourself across turns.
5. If this is your last or second-to-last turn, you MUST synthesize now."""


class SaCAgent:
    def __init__(
        self,
        task: str,
        sdk: AgenticSearchSDK | None = None,
        max_turns: int = 15,
        max_fixes_per_turn: int = 3,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "big-pickle",
        http_proxy: str | None = None,
        https_proxy: str | None = None,
        with_code_library: bool = False,
        sandbox_backend: str = "exec",
        context_limit: int | None = None,
        max_tokens: int = 8192,
        truncation: int = 10000,
        context_force_threshold: float = 0.80,
        images: list[str] | None = None,
    ) -> None:
        self.task = task
        self.max_turns = max_turns
        self.max_fixes_per_turn = max_fixes_per_turn
        self.model = model
        self._images: list[str] = []
        if images:
            for img_path in images:
                self._images.append(image_to_data_uri(img_path))
        self._with_code_library = with_code_library
        self._sandbox_backend = sandbox_backend
        self.context_limit = ModelLimits.get_context_limit(
            model, override=context_limit
        )
        self._context_limit = self.context_limit
        self._max_tokens = max_tokens
        self._truncation = truncation
        self._context_force_threshold = context_force_threshold
        self._usage = UsageTracker()
        self._fix_usage = UsageTracker()
        self._synthesis_usage = UsageTracker()
        self._base_url = (
            base_url
            or os.environ.get("OPENAI_API_BASE")
            or "https://opencode.ai/zen/v1"
        )
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or "public"
        self.sdk = sdk or AgenticSearchSDK(
            llm_base_url=self._base_url,
            llm_api_key=self._api_key,
            llm_model=model,
            http_proxy=http_proxy,
            https_proxy=https_proxy,
            llm_max_tokens=max_tokens,
            max_chars=truncation,
        )
        self.sandbox = Sandbox(self.sdk, backend=self._sandbox_backend)
        self.library = (
            CodeLibrary(
                base_url=self._base_url,
                api_key=self._api_key,
                model=model,
            )
            if with_code_library
            else None
        )
        self._turn = 0
        self._history: list[dict[str, Any]] = []
        self._start_time = 0.0

        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            http_client=_httpx_client(http_proxy, https_proxy),
        )

    def _record_usage(self, resp: Any, tracker: UsageTracker | None = None) -> None:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        t = tracker or self._usage
        pt = getattr(usage, "prompt_tokens", None)
        ct = getattr(usage, "completion_tokens", None)
        if isinstance(pt, int):
            t.prompt += pt
        if isinstance(ct, int):
            t.completion += ct

    @property
    def _system_prompt(self) -> str:
        pct = (
            self._usage.prompt / self._context_limit if self._context_limit > 0 else 0.0
        )
        remaining = max(0, self._context_limit - self._usage.prompt)
        return SYSTEM_PROMPT_TEMPLATE.format(
            context_limit=self._context_limit,
            pct_used=pct * 100,
            remaining=remaining,
        )

    @property
    def _context_used_pct(self) -> float:
        return (
            self._usage.prompt / self._context_limit if self._context_limit > 0 else 0.0
        )

    def _fix_code(self, code: str, error: str, _attempt: int) -> str | None:
        prompt = (
            f"The following Python code was executed but raised an error:\n\n"
            f"```python\n{code}\n```\n\n"
            f"Error:\n```\n{error[-self._truncation:]}\n```\n\n"
            f"Fix the code. Return ONLY valid Python code inside ```python...``` "
            f"or as a raw code block. No explanation."
        )
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        self._record_usage(resp, self._fix_usage)
        raw = resp.choices[0].message.content or ""
        if not raw:
            return None
        match = re.search(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw.strip()

    def _execute_and_fix(
        self, code: str, console: Console
    ) -> tuple[str, str, list[str]]:
        fixed_code = code
        fix_attempts = 0
        output = ""
        fs_keys: list[str] = []

        while fix_attempts <= self.max_fixes_per_turn:
            console.print("[dim cyan]  ↻ Executing in sandbox…[/]")
            t0 = time.time()
            output = self.sandbox.execute(fixed_code)
            elapsed_exec = time.time() - t0
            fs_keys = self.sdk.fs.list()

            searches = self.sdk.search.total_queries
            results_count = self.sdk.search.total_results
            console.print(
                f"  [dim]Executed in {elapsed_exec:.2f}s · "
                f"Searches: {searches} · "
                f"Results: {results_count} · "
                f"FS keys: {fs_keys}[/]"
            )
            lines = output.strip().split("\n")
            tail = "\n".join(lines[-10:]) if len(lines) > 10 else output
            if tail:
                console.print(
                    Panel(
                        tail[:800],
                        title="[dim]Output[/]",
                        border_style="dim",
                        padding=(0, 1),
                    )
                )

            if "--- ERROR ---" not in output:
                if self.library:
                    self.library.collect(fixed_code, self.task)
                break

            fix_attempts += 1
            if fix_attempts > self.max_fixes_per_turn:
                console.print(
                    f"[red]  ✗ {fix_attempts} fix attempts failed, giving up.[/]"
                )
                break

            console.print(
                f"[yellow]  ↻ Fix attempt {fix_attempts}/{self.max_fixes_per_turn}…[/]"
            )
            fixed = self._fix_code(fixed_code, output, fix_attempts)
            if fixed is None or fixed == fixed_code:
                console.print("[red]  ✗ Fixer returned no change, aborting.[/]")
                break
            fixed_code = fixed
            console.print(
                Syntax(
                    fixed_code,
                    "python",
                    theme="monokai",
                    line_numbers=False,
                    background_color="default",
                    word_wrap=True,
                )
            )
            console.print()

        return fixed_code, output, fs_keys

    def run(self) -> str:
        console = Console()
        self._start_time = time.time()
        console.print()
        console.print(
            Panel(
                f"[bold white]{self.task}[/]",
                title="[green]Search as Code — Task[/]",
                border_style="green",
            )
        )

        while self._turn < self.max_turns:
            self._turn += 1
            console.print(
                f"\n[dim]─── Turn {self._turn} / {self.max_turns} ───────────────────────────────[/]"
            )

            response = self._call_model()
            action = self._parse_response(response)

            if action is None:
                if self.library:
                    self.library.flush_all()
                return "Error: failed to parse agent response."

            reasoning = action.get("reasoning", "")
            console.print(f"[dim]  Reasoning:[/] [italic]{reasoning}[/]")

            if action.get("turn_type") == "synthesis":
                answer = action.get("answer", "No answer provided.")
                elapsed = time.time() - self._start_time
                console.print()
                console.print(
                    Panel(
                        answer,
                        title=(
                            f"[bold yellow]Answer[/] [dim]("
                            f"{self.sdk.search.total_queries} searches · "
                            f"{self.sdk.search.total_results} results · "
                            f"{self._usage.prompt:,} ctx · "
                            f"{elapsed:.1f}s)[/]"
                        ),
                        border_style="yellow",
                        padding=(1, 2),
                    )
                )
                if self.library:
                    self.library.flush_all()
                return cast("str", answer)

            code = action.get("code", "")
            if not code:
                self._history.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(action),
                    }
                )
                continue

            console.print()
            console.print(
                Syntax(
                    code,
                    "python",
                    theme="monokai",
                    line_numbers=False,
                    background_color="default",
                    word_wrap=True,
                )
            )
            console.print()

            _, output, fs_keys = self._execute_and_fix(code, console)

            self._history.append(
                {
                    "role": "assistant",
                    "content": json.dumps(action),
                }
            )
            turns_left = self.max_turns - self._turn
            ctx_pct = self._context_used_pct
            ctx_bar = "▓" * int(ctx_pct * 20) + "░" * (20 - int(ctx_pct * 20))
            self._history.append(
                {
                    "role": "user",
                    "content": (
                        f"Turn {self._turn} executed ({turns_left} turns left).\n"
                        f"Context: {ctx_bar} {self._usage.prompt:,}/{self._context_limit:,} "
                        f"({ctx_pct:.0%})\n"
                        f"Output:\n{output[-self._truncation:]}\n"
                        f"Persisted keys: {fs_keys}\n\n"
                        f"Keep tracking source URLs in sdk.fs — the final "
                        f"answer must include a ## References section."
                    ),
                }
            )
            if ctx_pct >= self._context_force_threshold:
                console.print(f"[red]Context at {ctx_pct:.0%} — forcing synthesis.[/]")
                break

        console.print("\n[yellow]Max turns reached — requesting final synthesis.[/]")
        fallback = self._force_synthesis()
        if self.library:
            self.library.flush_all()
        elapsed = time.time() - self._start_time
        console.print(
            f"[dim]Usage: {self._usage.prompt:,} prompt · "
            f"{self._usage.completion:,} completion · "
            f"{self._fix_usage.total:,} fix · "
            f"{self._synthesis_usage.total:,} synthesis · "
            f"{elapsed:.1f}s · "
            f"{self.sdk.search.total_queries} searches "
            f"({self.sdk.search.total_results} results)[/]"
        )
        return fallback

    def _call_model(self) -> str:
        messages: list[dict[str, Any]] = []
        if not self._history:
            text = (
                f"Research task: {self.task}\n\n"
                f"Available persisted keys: {self.sdk.fs.list()}\n\n"
                "Start by fanning out web searches. Respond with a code turn."
            )
            if self._images:
                msg_content: str | list[dict[str, Any]] = [
                    {"type": "text", "text": text},
                    *[
                        {"type": "image_url", "image_url": {"url": img}}
                        for img in self._images
                    ],
                ]
            else:
                msg_content = text
            messages.append({"role": "user", "content": msg_content})
        else:
            messages = self._history
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": self._system_prompt},
                *messages,
            ],
        )
        self._record_usage(resp)
        msg = resp.choices[0].message
        content = msg.content or ""
        if not content and hasattr(msg, "reasoning_content") and msg.reasoning_content:
            content = msg.reasoning_content
        return content or ""

    def _parse_response(self, raw: str) -> dict[str, Any] | None:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            return cast("dict[str, Any] | None", json.loads(cleaned))
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    return cast("dict[str, Any] | None", json.loads(match.group(0)))
                except json.JSONDecodeError:
                    return None
            return None

    def _force_synthesis(self) -> str:
        fs_keys = self.sdk.fs.list()
        context_parts = []
        for key in fs_keys:
            try:
                val = self.sdk.fs.read(key)
                context_parts.append(
                    f"{key}: {json.dumps(val, default=str, indent=2)[:self._truncation]}"
                )
            except Exception:
                pass
        prompt = (
            f"Research task: {self.task}\n\n"
            f"Available persisted data:\n"
            + "\n".join(context_parts)
            + "\n\nSynthesize a final answer based on the available evidence. "
            "Your answer MUST end with a ## References section listing every "
            "URL used as evidence (formatted as markdown bullets). "
            "Return only a JSON synthesis turn with turn_type 'synthesis'."
        )
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self._max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": "You are a research synthesizer. Return JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        self._record_usage(resp, self._synthesis_usage)
        msg = resp.choices[0].message
        raw = msg.content or ""
        if not raw and hasattr(msg, "reasoning_content") and msg.reasoning_content:
            raw = msg.reasoning_content
        parsed = self._parse_response(raw)
        if parsed and parsed.get("turn_type") == "synthesis":
            return cast("str", parsed.get("answer", "No synthesis could be generated."))
        return f"Research completed after {self.max_turns} turns. Check sdk.fs data."
