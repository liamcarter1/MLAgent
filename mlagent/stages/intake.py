"""Intake: fixed interview -> draft -> Claude checks it and writes spec.json."""

from __future__ import annotations

import json

from mlagent import config
from mlagent.llm import LLMError, ToolSpec
from mlagent.prompts_io import audience, load_prompt
from mlagent.spec import (
    DATA_SOURCES,
    GPU_CHOICES,
    LEARNING_LEVELS,
    METRICS_FOR_TASK,
    TASK_TYPES,
    Spec,
    SpecError,
)
from mlagent.stages.base import StageContext
from mlagent.ui.questions import Questioner

TASK_LABELS = {
    "Tabular classification": "tabular_classification",
    "Tabular regression": "tabular_regression",
    "Image classification": "image_classification",
}
SOURCE_LABELS = {
    "Synthetic data": "synthetic",
    "Upload or Google Drive path": "drive",
    "HuggingFace Hub dataset": "huggingface",
}
GPU_LABELS = {
    "No GPU (CPU only)": "none",
    "T4 GPU": "T4",
    "Any available GPU": "any",
}
LEVEL_LABELS = {
    "Beginner - explain everything as we go": "beginner",
    "Intermediate - explain the key ideas": "intermediate",
    "Expert - just the numbers": "expert",
}


def _label_to_code(answer: str, mapping: dict[str, str], allowed: tuple[str, ...]) -> str:
    if answer in mapping:
        return mapping[answer]
    if answer in allowed:
        return answer
    return mapping[next(iter(mapping))]


def collect_draft(q: Questioner) -> dict:
    goal = q.text(
        "In one or two sentences, what do you want the model to do?", key="intake.goal"
    )
    learning_level = _label_to_code(
        q.choice(
            "How much explanation do you want as we go?",
            list(LEVEL_LABELS),
            allow_other=False,
            key="intake.learning_level",
        ),
        LEVEL_LABELS, LEARNING_LEVELS,
    )
    task_type = _label_to_code(
        q.choice("What kind of task is it?", list(TASK_LABELS), allow_other=False,
                 key="intake.task_type"),
        TASK_LABELS, TASK_TYPES,
    )
    metric = q.choice("Which metric defines success?", METRICS_FOR_TASK[task_type],
                      allow_other=False, key="intake.metric")
    target_value = q.number(f"What {metric} value would count as good enough?", default=0.9,
                            key="intake.target_value")
    data_source = _label_to_code(
        q.choice("Where will the data come from?", list(SOURCE_LABELS), allow_other=False,
                 key="intake.data_source"),
        SOURCE_LABELS, DATA_SOURCES,
    )
    minutes = int(q.number("Roughly how many minutes per training run are acceptable?",
                           default=10, minimum=1, key="intake.minutes_per_run"))
    rounds = int(q.number("How many tuning rounds at most?", default=5, minimum=1,
                          key="intake.max_rounds"))
    gpu = _label_to_code(q.choice("GPU preference?", list(GPU_LABELS), allow_other=False,
                                  key="intake.gpu"),
                         GPU_LABELS, GPU_CHOICES)
    return {
        "goal": goal,
        "learning_level": learning_level,
        "task_type": task_type,
        "metric": metric,
        "target_value": target_value,
        "data_source": data_source,
        "minutes_per_run": minutes,
        "max_rounds": rounds,
        "gpu": gpu,
        "notes": "",
    }


SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "task_type": {"type": "string", "enum": list(TASK_TYPES)},
        "metric": {"type": "string"},
        "target_value": {"type": "number"},
        "data_source": {"type": "string", "enum": list(DATA_SOURCES)},
        "minutes_per_run": {"type": "integer"},
        "max_rounds": {"type": "integer"},
        "gpu": {"type": "string", "enum": list(GPU_CHOICES)},
        "learning_level": {"type": "string", "enum": list(LEARNING_LEVELS)},
        "notes": {"type": "string"},
    },
    "required": ["goal", "task_type", "metric", "target_value", "data_source",
                 "minutes_per_run", "max_rounds", "gpu"],
}


class IntakeStage:
    name = "intake"
    MAX_FOLLOWUPS = 2

    def is_complete(self, ctx: StageContext) -> bool:
        data = ctx.project.read_json(config.SPEC_FILE)
        if not data:
            return False
        try:
            return Spec.from_dict(data).validate() == []
        except (SpecError, TypeError, ValueError):
            return False

    def run(self, ctx: StageContext) -> None:
        draft = collect_draft(ctx.questioner)
        ctx.project.write_json("draft_spec.json", draft)
        written: dict = {}
        followups = {"n": 0}

        def ask_user(inp: dict) -> str:
            if followups["n"] >= self.MAX_FOLLOWUPS:
                return "No more follow-up questions allowed; call write_spec now."
            followups["n"] += 1
            options = inp.get("options")
            if options:
                return ctx.questioner.choice(inp["question"], [str(o) for o in options])
            return ctx.questioner.text(inp["question"])

        def write_spec(inp: dict) -> str:
            if not str(inp.get("learning_level") or "").strip():
                inp = {**inp, "learning_level": draft["learning_level"]}
            try:
                spec = Spec.from_dict(inp)
            except SpecError as exc:
                return f"invalid: {exc}"
            problems = spec.validate()
            if problems:
                return "invalid: " + "; ".join(problems)
            ctx.project.write_json(config.SPEC_FILE, spec.to_dict())
            written.update(spec.to_dict())
            return "ok"

        tools = [
            ToolSpec(
                name="ask_user",
                description="Ask the user one clarifying question. Provide options when sensible.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["question"],
                },
                handler=ask_user,
            ),
            ToolSpec(
                name="write_spec",
                description="Save the final project specification. Returns 'ok' or 'invalid: ...'.",
                input_schema=SPEC_SCHEMA,
                handler=write_spec,
            ),
        ]
        prompt = "Interview answers (draft spec):\n" + json.dumps(draft, indent=2, sort_keys=True)
        result = None
        try:
            result = ctx.llm.run(
                system=load_prompt("intake", audience=audience(draft["learning_level"])),
                messages=[{"role": "user", "content": prompt}],
                tools=tools,
            )
        except LLMError as exc:
            ctx.display(f"Couldn't reach Claude ({exc}); saved your answers as the spec.")
        if not written:
            spec = Spec.from_dict(draft)
            problems = spec.validate()
            if problems:
                raise SpecError("draft spec invalid: " + "; ".join(problems))
            ctx.project.write_json(config.SPEC_FILE, spec.to_dict())
        if result is not None:
            if result.text:
                ctx.display(result.text)
            else:
                ctx.display("Spec saved. Next: obtaining the [[training data]].")
