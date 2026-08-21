from __future__ import annotations

from importlib.metadata import version as package_version
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "Global Cache Workspace"
author = "GCW contributors"
release = package_version("gcw")
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinxcontrib.mermaid",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "substitution",
]
myst_heading_anchors = 3
autosummary_generate = False
autodoc_typehints = "description"
nitpicky = True
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"
html_title = "GCW documentation"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pytorch": ("https://docs.pytorch.org/docs/stable", None),
}
