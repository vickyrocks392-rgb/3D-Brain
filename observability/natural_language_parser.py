"""Natural-Language Experiment Interface & Constrained Instruction Parser.

Provides a deterministic, constrained natural-language parser for human-driven
experiment specification in the 3D-Brain Observatory.

SCIENTIFIC INTEGRITY RULE:
-------------------------
DO NOT PUT AN LLM INSIDE THE BRAIN.
The natural-language layer is strictly an experiment/task interface that translates
human text instructions into structured task specifications via a deterministic grammar.
It executes externally to the neural substrate and performs zero neuromorphic computation.

Supported Instruction Grammar Examples:
- "Learn that red means left, then test the rule after 20 distracting events."
- "Train on 50 examples of temporal classification with delay 10, then evaluate frozen."
- "Learn rule blue -> right for 25 trials, test after 15 distractors with frozen weights."
- "Evaluate MIN_E on static classification with budget 100."
- "Test compositional XOR rule for 50 examples with frozen evaluation."
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple

from benchmarks.phase23a_sample_efficiency import (
    BenchmarkTrial,
    TaskType,
    SystemArchitecture,
    get_task_datasets,
    EVAL_TRIALS_COUNT,
)


@dataclass
class ParsedTaskSpec:
    """Structured experiment/task specification parsed from natural-language input."""
    raw_instruction: str
    task_type: TaskType
    rule_description: str
    stimulus_name: str
    response_action: int
    distractor_count: int
    training_budget: int
    evaluation_mode: str
    learning_mode: str
    target_model: str
    valid: bool = True
    parse_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_instruction": self.raw_instruction,
            "task_type": self.task_type.value,
            "rule_description": self.rule_description,
            "stimulus_name": self.stimulus_name,
            "response_action": self.response_action,
            "distractor_count": self.distractor_count,
            "training_budget": self.training_budget,
            "evaluation_mode": self.evaluation_mode,
            "learning_mode": self.learning_mode,
            "target_model": self.target_model,
            "valid": self.valid,
            "parse_notes": self.parse_notes,
        }

    def format_display_card(self) -> str:
        """Format the interpreted task into the standard human-readable display card."""
        return (
            "INTERPRETED TASK\n\n"
            f"Rule:\n{self.rule_description}\n\n"
            f"Distractors:\n{self.distractor_count}\n\n"
            f"Evaluation:\n{self.evaluation_mode}\n\n"
            f"Learning:\n{self.learning_mode}\n\n"
            f"Budget:\n{self.training_budget} trials\n\n"
            f"Model:\n{self.target_model}"
        )


class NaturalLanguageTaskParser:
    """Deterministic parser for human experiment instructions."""

    STIMULUS_COLOR_MAP = {
        "red": 0,
        "blue": 1,
        "green": 0,
        "yellow": 1,
        "cue_a": 0,
        "cue_b": 1,
        "pattern_a": 0,
        "pattern_b": 1,
    }

    ACTION_MAP = {
        "left": 0,
        "right": 1,
        "action 0": 0,
        "action 1": 1,
        "0": 0,
        "1": 1,
        "port 0": 0,
        "port 1": 1,
    }

    @classmethod
    def parse(cls, instruction: str) -> ParsedTaskSpec:
        """Parse natural language instruction into structured task specification."""
        text = instruction.strip().lower()
        if not text:
            return ParsedTaskSpec(
                raw_instruction=instruction,
                task_type=TaskType.STATIC_CLASSIFICATION,
                rule_description="default: neutral → left",
                stimulus_name="neutral",
                response_action=0,
                distractor_count=0,
                training_budget=25,
                evaluation_mode="frozen",
                learning_mode="enabled during training only",
                target_model="MIN_E",
                valid=False,
                parse_notes="Empty instruction",
            )

        # 1. Detect target model
        target_model = "MIN_E"
        if "transformer" in text:
            target_model = "CAUSAL_TRANSFORMER"
        elif "gru" in text:
            target_model = "GRU"
        elif "episodic" in text or "memory" in text:
            target_model = "MIN_E_PLUS_EPISODIC"
        elif "typed" in text or "probabilistic" in text or "system one" in text or "jev" in text:
            target_model = "TYPED_PROBABILISTIC_DECISION"
        elif "rnn" in text or "simple recurrent" in text:
            target_model = "SIMPLE_RECURRENT"

        # 2. Extract distractor count / delay events
        distractors = 0
        dist_match = re.search(r'(\d+)\s*(?:distracting|distractor|intervening|delay|step)', text)
        if dist_match:
            distractors = int(dist_match.group(1))

        # 3. Extract training budget / examples
        budget = 25
        budget_match = re.search(r'(?:train|budget|for|learn\s*(?:for)?)\s*(?:on\s*)?(\d+)\s*(?:example|trial|sample|event)', text)
        if budget_match:
            budget = int(budget_match.group(1))
        else:
            # Check standalone budget numbers
            b2 = re.search(r'budget\s*[:=]?\s*(\d+)', text)
            if b2:
                budget = int(b2.group(1))

        # 4. Extract rule and mapping
        stimulus = "red"
        action = 0
        rule_desc = "red → left"
        task_type = TaskType.EPISODE_DEPENDENT_DECISION if distractors > 0 else TaskType.STATIC_CLASSIFICATION

        # Check for compositional keywords
        if "compositional" in text or "xor" in text or "parity" in text:
            task_type = TaskType.SIMPLE_COMPOSITIONAL_DECISION
            rule_desc = "C1 ^ C2 (XOR) → action"
            stimulus = "composite_cue"
            action = 1
        elif "temporal" in text or "sequence" in text or "order" in text:
            task_type = TaskType.TEMPORAL_CLASSIFICATION
            rule_desc = "order [A ... B] → left, [B ... A] → right"
            stimulus = "sequence"
            action = 0
        elif "delayed" in text:
            task_type = TaskType.DELAYED_CONTEXT_DECISION
            rule_desc = "context cue → persist across delay → match action"
            stimulus = "context_cue"
            action = 0
        else:
            # Pattern: "X means Y" or "X -> Y" or "X gives Y"
            rule_match = re.search(r'(\w+)\s*(?:means|->|gives|produces|to)\s*(\w+)', text)
            if rule_match:
                s_cand = rule_match.group(1)
                a_cand = rule_match.group(2)
                stimulus = s_cand
                if a_cand in cls.ACTION_MAP:
                    action = cls.ACTION_MAP[a_cand]
                    rule_desc = f"{s_cand} → {a_cand}"
                else:
                    action = 0
                    rule_desc = f"{s_cand} → left"
            else:
                if "blue" in text:
                    stimulus = "blue"
                    action = 1
                    rule_desc = "blue → right"
                elif "red" in text:
                    stimulus = "red"
                    action = 0
                    rule_desc = "red → left"

        # 5. Evaluation & Learning Modes (Strictly frozen test)
        eval_mode = "frozen"
        learning_mode = "enabled during training only"
        if "online" in text:
            learning_mode = "online learning"

        return ParsedTaskSpec(
            raw_instruction=instruction,
            task_type=task_type,
            rule_description=rule_desc,
            stimulus_name=stimulus,
            response_action=action,
            distractor_count=distractors,
            training_budget=max(1, budget),
            evaluation_mode=eval_mode,
            learning_mode=learning_mode,
            target_model=target_model,
            valid=True,
            parse_notes=f"Parsed successfully into {task_type.value} for {target_model}",
        )


def build_executable_trials_from_spec(
    spec: ParsedTaskSpec,
    seed: int = 42,
) -> Tuple[List[BenchmarkTrial], List[BenchmarkTrial]]:
    """Generate executable train and evaluation trial sets from parsed specification."""
    return get_task_datasets(
        task=spec.task_type,
        train_budget=spec.training_budget,
        eval_count=EVAL_TRIALS_COUNT,
        input_dim=16,
        seed=seed,
    )
