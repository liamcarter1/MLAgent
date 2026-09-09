"""Ask the user questions. Console version works in Colab via the inline input() box."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class Questioner(Protocol):
    def choice(
        self,
        question: str,
        options: list[str],
        allow_other: bool = True,
        key: str | None = None,
    ) -> str: ...
    def text(self, prompt: str, default: str | None = None, key: str | None = None) -> str: ...
    def confirm(self, question: str, default: bool = True, key: str | None = None) -> bool: ...
    def number(
        self,
        prompt: str,
        default: float | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
        key: str | None = None,
    ) -> float: ...


class ConsoleQuestioner:
    def __init__(
        self, input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print
    ):
        self._input = input_fn
        self._print = print_fn

    def choice(
        self,
        question: str,
        options: list[str],
        allow_other: bool = True,
        key: str | None = None,
    ) -> str:
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

    def text(self, prompt: str, default: str | None = None, key: str | None = None) -> str:
        suffix = f" [{default}]" if default is not None else ""
        while True:
            raw = self._input(f"{prompt}{suffix} > ").strip()
            if raw:
                return raw
            if default is not None:
                return default
            self._print("Please enter a value.")

    def confirm(self, question: str, default: bool = True, key: str | None = None) -> bool:
        suffix = " [Y/n]" if default else " [y/N]"
        raw = self._input(f"{question}{suffix} > ").strip().lower()
        if not raw:
            return default
        return raw in {"y", "yes"}

    def number(
        self,
        prompt: str,
        default: float | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
        key: str | None = None,
    ) -> float:
        bounds = ""
        if minimum is not None or maximum is not None:
            lo = "" if minimum is None else f"{minimum:g}"
            hi = "" if maximum is None else f"{maximum:g}"
            bounds = f" ({lo} to {hi})"
        suffix = f" [{default:g}]" if default is not None else ""
        while True:
            raw = self._input(f"{prompt}{bounds}{suffix} > ").strip()
            if not raw and default is not None:
                return float(default)
            try:
                value = float(raw)
            except ValueError:
                self._print("Please enter a number.")
                continue
            if (minimum is not None and value < minimum) or (
                maximum is not None and value > maximum
            ):
                self._print(f"Please enter a number{bounds}.")
                continue
            return value


class ScriptedQuestioner:
    def __init__(self, answers: list[str]):
        self._answers = list(answers)
        self.asked: list[str] = []

    def _next(self, question: str) -> str:
        self.asked.append(question)
        if not self._answers:
            raise RuntimeError(f"ScriptedQuestioner has no answer for: {question}")
        return self._answers.pop(0)

    def choice(
        self,
        question: str,
        options: list[str],
        allow_other: bool = True,
        key: str | None = None,
    ) -> str:
        return self._next(question)

    def text(self, prompt: str, default: str | None = None, key: str | None = None) -> str:
        answer = self._next(prompt)
        return answer if answer or default is None else default

    def confirm(self, question: str, default: bool = True, key: str | None = None) -> bool:
        answer = self._next(question).strip().lower()
        if not answer:
            return default
        return answer in {"y", "yes", "true"}

    def number(
        self,
        prompt: str,
        default: float | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
        key: str | None = None,
    ) -> float:
        answer = self._next(prompt).strip()
        if not answer and default is not None:
            return float(default)
        return float(answer)


_MISSING = object()
TRUTHY = {"y", "yes", "true", "1", "on"}
FALSEY = {"n", "no", "false", "0", "off"}


def _silent(_message: str) -> None:
    return None


class FormQuestioner:
    """Answers fixed questions from a Colab `#@param` form, falling back to `fallback`.

    A key that is absent from `answers` falls back silently: the form simply does not
    cover that question. A key that is present but empty or unusable falls back with a
    one-line note, because the user did fill the form in and deserves to know why they
    are being asked again. Exception: `text()` treats a blank value as the answer
    (the default) when the question has a non-None `default`, since a blank field on an
    optional question is a legitimate "use the default" answer, not a skipped one.
    """

    def __init__(
        self,
        answers: dict[str, object],
        fallback: Questioner,
        note: Callable[[str], None] = _silent,
    ):
        self.answers = dict(answers or {})
        self.fallback = fallback
        self.note = note
        self.used: list[str] = []

    def _raw(self, key: str | None):
        if key is None or key not in self.answers:
            return _MISSING
        value = self.answers[key]
        if value is None or (isinstance(value, str) and not value.strip()):
            self.note(f"The form field for '{key}' is empty; asking instead.")
            return _MISSING
        return value

    def _reject(self, key: str, value) -> None:
        self.note(f"'{value}' is not a valid answer for '{key}'; asking instead.")

    def _accept(self, key: str, value):
        self.used.append(key)
        return value

    def choice(
        self,
        question: str,
        options: list[str],
        allow_other: bool = True,
        key: str | None = None,
    ) -> str:
        raw = self._raw(key)
        if raw is not _MISSING:
            text = str(raw).strip()
            for option in options:
                if text.lower() == option.lower():
                    return self._accept(key, option)
            if allow_other:
                return self._accept(key, text)
            self._reject(key, text)
        return self.fallback.choice(question, options, allow_other=allow_other, key=key)

    def text(self, prompt: str, default: str | None = None, key: str | None = None) -> str:
        if key is not None and key in self.answers:
            value = self.answers[key]
            blank = value is None or (isinstance(value, str) and not value.strip())
            if blank and default is not None:
                # The question is optional by construction (it has a default), so a
                # blank form field is a real answer, not a sign the user skipped it.
                return self._accept(key, default)
        raw = self._raw(key)
        if raw is not _MISSING:
            return self._accept(key, str(raw).strip())
        return self.fallback.text(prompt, default=default, key=key)

    def confirm(self, question: str, default: bool = True, key: str | None = None) -> bool:
        raw = self._raw(key)
        if raw is not _MISSING:
            if isinstance(raw, bool):
                return self._accept(key, raw)
            text = str(raw).strip().lower()
            if text in TRUTHY:
                return self._accept(key, True)
            if text in FALSEY:
                return self._accept(key, False)
            self._reject(key, raw)
        return self.fallback.confirm(question, default=default, key=key)

    def number(
        self,
        prompt: str,
        default: float | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
        key: str | None = None,
    ) -> float:
        raw = self._raw(key)
        if raw is not _MISSING:
            try:
                value = float(str(raw).strip())
            except (TypeError, ValueError):
                self._reject(key, raw)
            else:
                too_low = minimum is not None and value < minimum
                too_high = maximum is not None and value > maximum
                if too_low or too_high:
                    self._reject(key, raw)
                else:
                    return self._accept(key, value)
        return self.fallback.number(
            prompt, default=default, minimum=minimum, maximum=maximum, key=key
        )
