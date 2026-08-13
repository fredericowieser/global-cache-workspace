# Documentation policy

**Status:** Maintained

Use short declarative sentences. Use one stable term for one concept. Preserve every scientific condition, number, and uncertainty. Do not use marketing language.

Handwritten MyST explains mechanisms and decisions. Python signatures and docstrings generate API facts. Typed config schemas generate configuration facts. `build_parser()` generates CLI facts. Never edit `docs/_generated/` by hand.

Label experiment and proposal pages as Maintained, Experimental, Historical, or Proposal. Distinguish mathematical consequences, implementation facts, observations, interpretations, and hypotheses.

For an important mechanism, connect the idea, equation, symbol definitions, tensor shapes, short code, implementation owner, and parity tests. Comments explain intent or invariants. They do not narrate syntax.

Build locally:

```bash
uv run --extra cpu --group docs python scripts/docs/generate_reference.py
uv run --extra cpu --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
```
