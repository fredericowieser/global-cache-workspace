from contextlib import contextmanager
from pathlib import Path
import json
import shutil
import time
from uuid import uuid4


class PublishedDatasetCleanupError(OSError):
    def __init__(
        self, output: Path, directory: Path, *, kind: str = "superseded backup"
    ):
        self.output = output
        self.directory = directory
        super().__init__(
            f"Complete dataset published at {output}; could not remove the "
            f"{kind} at {directory}. Close handles to that directory and remove it. "
            "The published dataset is valid; regeneration is unnecessary."
        )


def _remove_directory(path: Path, parent: Path):
    if path.is_symlink() or path.resolve().parent != parent.resolve():
        raise ValueError("Dataset cleanup escaped its destination parent")
    for delay in (0, 0.1, 0.5):
        if delay:
            time.sleep(delay)
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            if not path.exists():
                return
            raise
        except OSError:
            if delay == 0.5:
                raise


@contextmanager
def dataset_staging(output: Path):
    """Stage beside output; the caller must publish before normal context exit."""
    parent = output.parent.resolve()
    temporary = parent / (".reasoning-gym-" + uuid4().hex)
    temporary.mkdir()
    try:
        yield temporary
    except BaseException as error:
        try:
            _remove_directory(temporary, parent)
        except OSError as exc:
            raise error from OSError(
                f"Could not remove staging directory at {temporary}: {exc}"
            )
        raise
    else:
        try:
            _remove_directory(temporary, parent)
        except OSError as exc:
            raise PublishedDatasetCleanupError(
                output, temporary, kind="staging directory"
            ) from exc


def publish_dataset(staging: Path, output: Path, *, replace: bool = False):
    parent = staging.parent.parent.resolve()
    if output.is_symlink() or output.resolve().parent != parent:
        raise ValueError("Dataset publication escaped its destination parent")
    if not output.exists():
        staging.rename(output)
        return
    if not replace:
        raise FileExistsError(output)
    allowed = {"manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl"}
    if not (output / "manifest.json").is_file() or any(
        path.name not in allowed or not path.is_file() or path.is_symlink()
        for path in output.iterdir()
    ):
        raise ValueError(
            "Refusing to replace a directory containing unrelated files"
        )
    schema = json.loads((output / "manifest.json").read_bytes()).get("schema")
    if schema not in {
        "gcw.reasoning_gym.sft.v1",
        "gcw.reasoning_gym.subset.v1",
    }:
        raise ValueError("Refusing to replace an unrecognised dataset")
    backup = parent / (".reasoning-gym-" + uuid4().hex)
    if backup.resolve().parent != parent:
        raise ValueError("Dataset backup escaped its destination parent")
    output.rename(backup)
    try:
        staging.rename(output)
    except BaseException:
        backup.rename(output)
        raise
    try:
        _remove_directory(backup, parent)
    except OSError as exc:
        raise PublishedDatasetCleanupError(output, backup) from exc
