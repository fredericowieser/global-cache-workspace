from contextlib import contextmanager
from pathlib import Path
import shutil
from uuid import uuid4


@contextmanager
def dataset_staging(parent: Path):
    """Stage on the destination filesystem with normal inherited permissions."""
    parent = parent.resolve()
    temporary = parent / (".reasoning-gym-" + uuid4().hex)
    temporary.mkdir()
    try:
        yield temporary
    finally:
        if temporary.resolve().parent != parent:
            raise ValueError("Staging cleanup escaped its destination parent")
        shutil.rmtree(temporary)
