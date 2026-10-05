## General guidelines

* Whenever you implement new features in the core codebase, it is highly important that you make smart choices about the efficient re-use of existing logic and structure.
* Read the README before making code changes, changing configs, or launching jobs. If anything is still unclear with regards to the task at hand, look at the docs as well.
* Make sure you update the docs when necessary, especially when adding new features or changing existing functionality.
* Keep to the user's explicit requests. Do not change code or launch jobs unless the user has explicitly asked for that action.
* Never stage, unstage, commit, pull, or push changes without the user's explicit request.

## Code practices

* Do not add useless helpers: a single-line helper that simply computes a value or delegates to another function without adding meaningful abstraction or logic is strongly discouraged.
* Prioritise logical ownership: a class and method should own the logic and state that are most relevant to them, rather than scattering responsibilities across unrelated components. This also greatly improves code maintainability and readability.
* Keep to the formatting conventions used throughout the codebase, including indentation, line breaks, and naming schemes.
* Avoid comments unless strictly necessary for future reference: for example, TODOs are acceptable, but avoid redundant or obvious comments.

## Contributions to the paper

This codebase is a development codebase designed to run the research experiments presented in the paper, ensuring reproducibility and facilitating further investigation into the proposed methods. A draft of the paper, in latex format, is available as a submodule: `src/Global-Cache-Workspace-Theory-ICLR-2027-`. Sometimes you may be asked to make changes to this paper. When this happens you should make sure to adapt to this setting: adopt a scientific writing style consistent with that of a high-impact machine learning research paper. Avoid all non-crucial details that belong in the codebase: include only what is necessary to understand and reproduce the experiments. Don't mention the version of the scoring or other implementation-specific details that are not relevant to the scientific content of the paper.

Some more specific points:
* NEVER use em-dashes for anything other than showing missing results in tables.
* If you are not writing a new paragraph, continue the text on the same line without adding unnecessary line breaks. The column-80 limit does not apply to the md/latex files.


