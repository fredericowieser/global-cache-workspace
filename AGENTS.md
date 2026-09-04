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