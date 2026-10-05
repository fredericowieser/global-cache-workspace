from __future__ import annotations

from dataclasses import dataclass
from math import gcd
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
from gcw.data._vendor.reasoning_gym.algorithmic.base_conversion import (
    BaseConversionConfig,
    BaseConversionDataset,
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


class AdmissionError(ValueError):
    """An expected, counted exclusion rather than a correctness failure."""


@dataclass
class BalancedGCDConfig(GCDConfig):
    max_common_factor: int = 100
    coprime_probability: float = 0.25

    def validate(self):
        super().validate()
        if not (
            self.min_numbers == self.max_numbers == 2
            and 2 <= self.max_common_factor <= self.max_value // 10
            and self.max_value - self.min_value >= 10 * self.max_common_factor
            and 0 < self.coprime_probability < 1
        ):
            raise ValueError("Invalid balanced GCD configuration")


class BalancedGCDDataset(GCDDataset):
    def _generate_numbers(self, rng):
        factor = (
            1
            if rng.random() < self.config.coprime_probability
            else rng.randint(2, self.config.max_common_factor)
        )
        lower = (self.config.min_value + factor - 1) // factor
        upper = self.config.max_value // factor
        for _ in range(256):
            a, b = (rng.randint(lower, upper) for _ in range(2))
            if gcd(a, b) == 1:
                return [factor * a, factor * b], factor
        raise RuntimeError("Could not generate coprime GCD cofactors")


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
    "base_conversion": (BaseConversionDataset, BaseConversionConfig),
    "ab": (ABDataset, ABConfig),
    "color_cube_rotation": (ColorCubeRotationDataset, ColorCubeRotationConfig),
    "manipulate_matrix": (ManipulateMatrixDataset, ManipulateMatrixConfig),
    "gcd": (BalancedGCDDataset, BalancedGCDConfig),
}
TASK_CONFIGS = {
    "string_manipulation": dict(
        min_string_length=8,
        max_string_length=28,
        min_num_rules=6,
        max_num_rules=12,
    ),
    "base_conversion": dict(
        min_base=2, max_base=16, min_value=256, max_value=1048575
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
    "gcd": dict(min_numbers=2, max_numbers=2, min_value=10, max_value=100000),
}
FAMILIES = {
    "string_manipulation": "rewriting",
    "base_conversion": "arithmetic",
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
    elif task == "base_conversion":
        match = re.search(
            r"Now, convert the (binary|hexadecimal|base-\d+) number "
            r"([0-9a-f]+) to (binary|hexadecimal|base-\d+)",
            question,
        )
        if match is None:
            raise ValueError("Unrecognised base conversion prompt")
        source_name, source_repr, target_name = match.groups()
        bases = {"binary": 2, "hexadecimal": 16}
        source_base = bases.get(source_name) or int(
            source_name.removeprefix("base-")
        )
        target_base = bases.get(target_name) or int(
            target_name.removeprefix("base-")
        )
        problem = [source_base, source_repr, target_base]
        lines.append(
            f"Read the base-{source_base} digits from left to right, starting at 0."
        )
        value = 0
        for character in source_repr:
            digit = int(character, 36)
            if digit >= source_base:
                raise ValueError("Digit outside the source base")
            updated = value * source_base + digit
            lines.append(
                f"Read {character!r} ({digit}): {value} * {source_base} + {digit} = {updated}."
            )
            value = updated
            steps += 1
        lines.append(
            f"The decimal value is {value}. Divide repeatedly by {target_base} and record each remainder."
        )
        digits = "0123456789abcdefghijklmnopqrstuvwxyz"
        remaining = value
        remainders = []
        while remaining:
            quotient, remainder = divmod(remaining, target_base)
            lines.append(
                f"{remaining} = {target_base} * {quotient} + {remainder}; record digit {digits[remainder]!r}."
            )
            remainders.append(digits[remainder])
            remaining = quotient
            steps += 1
        final = "".join(reversed(remainders)) or "0"
        lines.append(
            f"The quotient is 0. Read the recorded digits in reverse order: {final}."
        )
        rendered_steps = steps
        difficulty = {
            "source_base": source_base,
            "target_base": target_base,
            "source_digits": len(source_repr),
            "target_digits": len(final),
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
        difficulty.update(
            final_rows=len(matrix),
            final_cols=len(matrix[0]) if matrix else 0,
            distinct_cells=len({cell for row in matrix for cell in row}),
        )
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
