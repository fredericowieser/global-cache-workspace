# LM Evaluation Harness

**Status:** Maintained

GCW adapts native models to `lm-eval` for task execution. Source-controlled task YAML files define GCW-specific prompts and scoring. The integration preserves prompt tokens, continuation boundaries, stop sequences, batch ordering, and partial-result authentication.

Do not copy upstream task documentation into this repository. Document only GCW overrides, pinned behavior, and scientific consequences. Evaluation W&B records and local result bundles must refer to the same authenticated task outputs.
