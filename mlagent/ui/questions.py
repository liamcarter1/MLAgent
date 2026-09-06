"""Ask the user questions. Console version works in Colab via the inline input() box."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class Questioner(Protocol):
    def choice(self, question: str, options: list[str], allow_other: bool = True) -> str: ...
    def text(self, prompt: str, default: str | None = None) -> str: ...
    def confirm(self, question: str, default: bool = True) -> bool: ...


class ConsoleQuestioner:
    def __init__(self, input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print):
        self._input = input_fn
        self._print = print_fn

    def choice(self, question: str, options: list[str], allow_other: bool = True) -> str:
        self._print(question)
        for i, opt in enumerate(options, 1):
            self._print(f"  {i}) {opt}")
        hint = "number or text" + (", or type your own answer" if allow_other else "")
        while True:
            raw = self._input(f"[{hint}] > ").strip()
            if raw.isdigit():
                if 1 <= int(raw) <= len(options):
                    return options[int(raw) - 1]
                self._print(f"Please enter a number from 1 to {len(options)}.")
                continue
            for opt in options:
                if raw.lower() == opt.lower():
                    return opt
            if allow_other and raw:
                return raw
            self._print(f"Please enter a number from 1 to {len(options)}.")

    def text(self, prompt: str, default: str | None = None) -> str:
        suffix = f" [{default}]" if default is not None else ""
        while True:
            raw = self._input(f"{prompt}{suffix} > ").strip()
            if raw:
                return raw
            if default is not None:
                return default
            self._print("Please enter a value.")

    def confirm(self, question: str, default: bool = True) -> bool:
        suffix = " [Y/n]" if default else " [y/N]"
        raw = self._input(f"{question}{suffix} > ").strip().lower()
        if not raw:
            return default
        return raw in {"y", "yes"}


class ScriptedQuestioner:
    def __init__(self, answers: list[str]):
        self._answers = list(answers)
        self.asked: list[str] = []

    def _next(self, question: str) -> str:
        self.asked.append(question)
        if not self._answers:
            raise RuntimeError(f"ScriptedQuestioner has no answer for: {question}")
        return self._answers.pop(0)

    def choice(self, question: str, options: list[str], allow_other: bool = True) -> str:
        return self._next(question)

    def text(self, prompt: str, default: str | None = None) -> str:
        answer = self._next(prompt)
        return answer if answer or default is None else default

    def confirm(self, question: str, default: bool = True) -> bool:
        answer = self._next(question).strip().lower()
        if not answer:
            return default
        return answer in {"y", "yes", "true"}
