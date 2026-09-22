from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
import re

from gcw.data._vendor.reasoning_gym.algorithmic.ab import (
    ABConfig,
    ABDataset,
    compute_steps,
)
from gcw.data._vendor.reasoning_gym.algorithmic.manipulate_matrix import (
    ManipulateMatrixConfig,
    ManipulateMatrixDataset,
)
from gcw.data._vendor.reasoning_gym.algorithmic.string_manipulation import (
    StringManipulationConfig,
    StringManipulationDataset,
)
from gcw.data._vendor.reasoning_gym.algorithmic.string_splitting import (
    StringSplittingConfig,
    StringSplittingDataset,
)
from gcw.data._vendor.reasoning_gym.arithmetic.gcd import GCDConfig, GCDDataset
from gcw.data._vendor.reasoning_gym.cognition.color_cube_rotation import (
    Color,
    ColorCubeRotationConfig,
    ColorCubeRotationDataset,
    Cube,
    Side,
)

FORMAT_INSTRUCTION = (
    "Write a worked solution showing the operations and intermediate states. "
    "Then put only the final result inside <answer>...</answer> tags."
)
MAX_TRANSITIONS = 256
SPLIT_DELTAS = (
    (-1, 0, 0, 2, 1, 0),
    (0, -2, 0, 1, 0, 0),
    (0, 0, -2, 0, 1, 0),
    (1, -1, -1, 0, 0, 0),
    (0, 0, 0, -1, -1, 1),
)
SPLIT_RULES = ("A -> 2X + Y", "2B -> X", "2C -> Y", "B + C -> A", "X + Y -> Z")


class AdmissionError(ValueError):
    """An expected, counted exclusion rather than a correctness failure."""


class BoundedStringManipulationDataset(StringManipulationDataset):
    def _get_all_transforms(self, string, selected_rules):
        states = [string]
        seen = {string}
        for _ in range(MAX_TRANSITIONS + 1):
            result, rule = self._apply_rule(states[-1], selected_rules)
            if not rule or result in seen:
                return states
            if len(states) > MAX_TRANSITIONS:
                raise AdmissionError("generation_transition_limit")
            states.append(result)
            seen.add(result)
        raise AssertionError("unreachable transition bound")


TASKS = {
    "string_manipulation": (
        BoundedStringManipulationDataset,
        StringManipulationConfig,
    ),
    "string_splitting": (StringSplittingDataset, StringSplittingConfig),
    "ab": (ABDataset, ABConfig),
    "color_cube_rotation": (ColorCubeRotationDataset, ColorCubeRotationConfig),
    "manipulate_matrix": (ManipulateMatrixDataset, ManipulateMatrixConfig),
    "gcd": (GCDDataset, GCDConfig),
}
TASK_CONFIGS = {
    "string_manipulation": dict(
        min_string_length=8,
        max_string_length=28,
        min_num_rules=6,
        max_num_rules=12,
    ),
    "string_splitting": dict(
        min_initial_machines=0, max_initial_machines=64, max_iterations=256
    ),
    "ab": dict(length=18),
    "color_cube_rotation": dict(min_rotations=2, max_rotations=8),
    "manipulate_matrix": dict(
        min_rows=3,
        max_rows=6,
        min_cols=3,
        max_cols=6,
        min_transforms=3,
        max_transforms=8,
    ),
    "gcd": dict(min_numbers=2, max_numbers=4, min_value=10, max_value=100000),
}
FAMILIES = {
    "string_manipulation": "rewriting",
    "string_splitting": "rewriting",
    "ab": "rewriting",
    "color_cube_rotation": "spatial_tracking",
    "manipulate_matrix": "matrix_transformations",
    "gcd": "arithmetic",
}


@dataclass(frozen=True)
class Trace:
    prompt: str
    solution: str
    final_answer: str
    steps: int
    rendered_steps: int
    termination: str
    problem: object
    difficulty: dict


def create_dataset(task: str, *, seed: int, size: int):
    dataset_cls, config_cls = TASKS[task]
    return dataset_cls(config_cls(seed=seed, size=size, **TASK_CONFIGS[task]))


def extract_final_answer(text: str) -> str | None:
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    return matches[0].strip() if len(matches) == 1 else None


def _matrix_operations(question: str) -> list[dict]:
    patterns = (
        (r"Rotate the matrix (90|180|270|360) degrees", "rotate", ("degrees",)),
        (r"Horizontally mirror the matrix", "hmirror", ()),
        (r"Vertically mirror the matrix", "vmirror", ()),
        (r"Mirror the matrix along the diagonal", "dmirror", ()),
        (r"Mirror the matrix along the counterdiagonal", "cmirror", ()),
        (r"Map each occurrence of (\d+) to (\d+)", "map", ("from", "to")),
        (
            r"Set all elements divisible by (\d+) to zero",
            "zero_divisible",
            ("k",),
        ),
        (
            r"Crop the matrix to rows (\d+)-(\d+) and columns (\d+)-(\d+) \(1-indexed\)",
            "crop",
            ("row_start", "row_end", "col_start", "col_end"),
        ),
        (
            r"Remove every (\d+)-(?:st|nd|th) row \(1-indexed\)",
            "remove_every_nth_row",
            ("n",),
        ),
        (
            r"Remove every (\d+)-(?:st|nd|th) column \(1-indexed\)",
            "remove_every_nth_col",
            ("n",),
        ),
    )
    operations = []
    instructions = question.split(
        "- Identity transformation, i.e. no change\n", 1
    )[1]
    for instruction in instructions.splitlines():
        for pattern, transform, keys in patterns:
            match = re.fullmatch(pattern, instruction.removeprefix("- "))
            if match:
                values = match.groups()
                if transform != "rotate":
                    values = tuple(map(int, values))
                operations.append(
                    {
                        "transform": transform,
                        "instruction": instruction,
                        **dict(zip(keys, values, strict=True)),
                    }
                )
                break
        else:
            raise ValueError(f"Unknown matrix instruction: {instruction}")
    return operations


def render_trace(task: str, dataset, entry: dict) -> Trace:
    question = entry["question"].strip()
    metadata = entry["metadata"]
    lines = []
    steps = 0
    termination = "completed"
    if task == "string_manipulation":
        state = question.rsplit("above list of rules:\n", 1)[1].strip()
        descriptions = re.findall(r"^\d+\. (.+)$", question, re.MULTILINE)
        rule_map = dict(dataset.rules)
        rules = [(text, rule_map[text]) for text in descriptions]
        problem = [state, descriptions]
        lines.append(f"Initial string: {state!r}.")
        seen = {state}
        while True:
            result, rule_id = dataset._apply_rule(state, rules)
            if not rule_id:
                lines.append(
                    "No listed rule applies. Stop at the current string."
                )
                termination = "no_applicable_rule"
                break
            position = next(
                i for i, (_, fn) in enumerate(rules, 1) if fn(state)[1]
            )
            if result in seen:
                lines.append(
                    f"Rule {position}, the first applicable rule, would produce "
                    f"{result!r}, a previously visited state. Discard this "
                    "repeated state and stop at the preceding string."
                )
                termination = "repeated_state_discarded"
                break
            if steps >= MAX_TRANSITIONS:
                raise AdmissionError("generation_transition_limit")
            lines.append(
                f"{steps + 1}. Rule {position} is the first applicable rule: "
                f"{descriptions[position - 1]} {state!r} -> {result!r}."
            )
            steps += 1
            state = result
            seen.add(state)
        final = state
        rendered_steps = steps
        question = question.replace(
            "Your output should be the final transformed string after applying all the rules.",
            "Find the final transformed string after applying all the rules.",
        )
        question += "\nFor an even-length string, the middle character means the right of the two middle characters."
        difficulty = {"initial_length": len(problem[0]), "rules": len(rules)}
    elif task == "string_splitting":
        match = re.search(
            r"Now, you have (\d+) machine A, (\d+) machine B, and (\d+) machine C",
            question,
        )
        initial = [int(value) for value in match.groups()]
        limit = int(
            re.search(r"Apply the rules at most (\d+) times", question)[1]
        )
        state = initial + [0, 0, 0]
        problem = [initial, limit]
        lines.append(
            "Use count order (A, B, C, X, Y, Z). Initial counts: "
            + str(state)
            + "."
        )
        transitions = []
        seen = {tuple(state)}
        for _ in range(limit):
            result = dataset._apply_rule(state[:])
            if result == state:
                termination = "no_applicable_rule"
                break
            if tuple(result) in seen:
                termination = "repeated_state_discarded"
                break
            delta = tuple(b - a for a, b in zip(state, result))
            rule = SPLIT_DELTAS.index(delta)
            transitions.append((rule, result))
            state = result
            seen.add(tuple(state))
        else:
            termination = "iteration_limit"
        for rule, group in groupby(transitions, key=lambda item: item[0]):
            run = list(group)
            lines.append(
                f"Apply rule {rule + 1} ({SPLIT_RULES[rule]}) {len(run)} "
                f"time(s), checking priority from rule 1 after each application. "
                f"Each application changes the counts by {SPLIT_DELTAS[rule]}. "
                f"Counts become {run[-1][1]}."
            )
        rendered_steps = len(lines) - 1
        steps = len(transitions)
        if termination == "no_applicable_rule":
            lines.append(
                "A = 0, B < 2, C < 2, B and C are not both positive, and X and Y are not both positive. No rule applies; stop."
            )
        elif termination == "iteration_limit":
            lines.append(
                f"The stated limit of {limit} applications has been reached; stop at the current counts."
            )
        else:
            lines.append(
                f"The next counts {result} repeat a previous state; discard the repeated state and stop."
            )
        final = " ".join(map(str, state))
        difficulty = {
            "initial_machine_total": sum(initial),
            "iteration_limit": limit,
        }
    elif task == "ab":
        program = (
            question.split("Now, consider the following program:\n\n")[1]
            .split("\n\n")[0]
            .split()
        )
        problem = program
        states, non_halting = compute_steps(program)
        if non_halting:
            raise AdmissionError("ab_step_limit")
        lines.append("Initial program: " + " ".join(program))
        for before, after in zip(states, states[1:]):
            position = next(
                i
                for i in range(len(before) - 1)
                if before[i].endswith("#") and before[i + 1].startswith("#")
            )
            pair = before[position : position + 2]
            replacement = [] if pair[0][0] == pair[1][1] else pair[::-1]
            if (
                before[:position] + replacement + before[position + 2 :]
                != after
            ):
                raise ValueError("Invalid AB transition")
            lines.append(
                f"{steps + 1}. The leftmost rewritable pair, at positions "
                f'{position + 1}-{position + 2}, is {" ".join(pair)}. '
                f'Replace it with {" ".join(replacement) or "nothing"}. '
                f'Program: {" ".join(after) or "(empty)"}.'
            )
            steps += 1
        lines.append(
            "No adjacent pair has its # symbols facing each other, so no rule applies. Stop."
        )
        final = " ".join(states[-1])
        rendered_steps = steps
        termination = "no_applicable_rule"
        question += "\nAlways rewrite the leftmost applicable pair first."
        difficulty = {"initial_tokens": len(program)}
    elif task == "color_cube_rotation":
        initial = {
            side: Color(color)
            for color, side in re.findall(r"- a (\w+) (\w+) side", question)
        }
        rotations = re.findall(
            r"(?:which was before at the|bring the|make the|place its|Next, the) "
            r"(right|front|left|back|bottom) "
            r"(?:is now at the top|side to the top|face the top|side at the top|side is rotated to become the top face)",
            question,
        )
        target = re.search(r"color of the (\w+) side", question)[1]
        if (
            initial != metadata["initial_state"]
            or rotations != metadata["rotations"]
        ):
            raise ValueError("Cube prompt and metadata differ")
        cube = Cube({Side(side): color for side, color in initial.items()})
        problem = [initial, rotations, target]
        for rotation in rotations:
            dataset._rotate_to_top(cube, Side(rotation))
            state = ", ".join(
                f"{side.value}={cube.colors[side].value}" for side in Side
            )
            lines.append(
                f"{steps + 1}. Bring the {rotation} face to the top: {state}."
            )
            steps += 1
        final = cube.colors[Side(target)].value
        lines.append(f"Read the {target} face: {final}.")
        rendered_steps = steps
        question = question.replace(
            "Provide only the color as your final answer.", ""
        )
        question += "\nEach front/back/left/right-to-top move is a quarter-turn with no extra twist. A bottom-to-top move is a half-turn about the left-right axis: top/bottom and front/back swap, while left/right stay fixed."
        difficulty = {"rotations": steps}
    elif task == "manipulate_matrix":
        matrix = [
            [int(x) for x in row.split()]
            for row in question.split("matrix:\n")[1]
            .split("\n\n")[0]
            .splitlines()
        ]
        operations = _matrix_operations(question)
        problem = [matrix, operations]
        difficulty = {
            "rows": len(matrix),
            "cols": len(matrix[0]),
            "operations": len(operations),
        }
        for index, op in enumerate(operations, 1):
            kind = op["transform"]
            if kind == "rotate":
                result = dataset._rotations[op["degrees"]](matrix)
            else:
                keys = {
                    "map": ("from", "to"),
                    "zero_divisible": ("k",),
                    "crop": ("row_start", "row_end", "col_start", "col_end"),
                    "remove_every_nth_row": ("n",),
                    "remove_every_nth_col": ("n",),
                }.get(kind, ())
                result = getattr(dataset, "_" + kind)(
                    matrix, *(op[key] for key in keys)
                )
            steps += result != matrix
            matrix = result
            lines.append(
                f'{index}. {op["instruction"].removeprefix("- ")}:\n{dataset._matrix_to_str(matrix)}'
            )
        final = dataset._matrix_to_str(matrix)
        lines.append(
            "All operations have been applied in the given order. The last matrix is the result."
        )
        rendered_steps = len(operations)
        question += "\nRotations are clockwise. Horizontal mirroring reverses row order; vertical mirroring reverses column order. Diagonal mirroring transposes the matrix; counterdiagonal mirroring transposes it after reversing both row and column order."
    elif task == "gcd":
        numbers = [
            int(x)
            for x in question.split("numbers: ")[1].split(".")[0].split(", ")
        ]
        problem = numbers
        result = numbers[0]
        for number in numbers[1:]:
            a, b = max(result, number), min(result, number)
            lines.append(
                f"Compute gcd({result}, {number}) using Euclidean division:"
            )
            while b:
                quotient, remainder = divmod(a, b)
                lines.append(f"{a} = {quotient} * {b} + {remainder}.")
                a, b = b, remainder
                steps += 1
            result = a
            lines.append(
                f"The last nonzero remainder (or divisor) is {result}, so the running GCD is {result}."
            )
        final = str(result)
        lines.append(
            f"All {len(numbers)} numbers have been included; their GCD is {final}."
        )
        rendered_steps = steps
        question = question.replace(
            " Give only the GCD as your final answer.", ""
        )
        difficulty = {
            "numbers": len(numbers),
            "max_digits": len(str(max(numbers))),
        }
    else:
        raise ValueError(f"Unsupported task: {task}")
    if not final.strip():
        raise AdmissionError("empty_final_answer")
    if final != entry["answer"] or dataset.score_answer(final, entry) != 1.0:
        raise ValueError(
            f"{task}: reconstructed answer disagrees with generator"
        )
    return Trace(
        prompt=question.strip() + "\n\n" + FORMAT_INSTRUCTION,
        solution="\n".join(lines),
        final_answer=final,
        steps=steps,
        rendered_steps=rendered_steps,
        termination=termination,
        problem=problem,
        difficulty=difficulty,
    )
