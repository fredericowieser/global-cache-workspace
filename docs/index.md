# Global Cache Workspace

GCW tests whether small learned processors can select, transform, and rewrite a causal language model's internal cache. The package supports native transformer backbones, BT, BGT, Icarus, Longshot, exact training resume, controlled evaluation, and scientific provenance.

Start with the CPU run. Read the concepts before changing cache semantics. Use campaign pages to reproduce scientific results.

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started/installation
getting-started/first-cpu-run
getting-started/first-training-run
getting-started/operate-repository
```

```{toctree}
:maxdepth: 2
:caption: Concepts

concepts/overview
concepts/transformer-cache
concepts/cache-processing
concepts/task-directed-diffusion
concepts/model-variants
concepts/data-and-tokenization
concepts/research-boundaries
```

```{toctree}
:maxdepth: 2
:caption: Guides

guides/configure-training
guides/training-and-resume
guides/resume-training
guides/evaluation-conversion-analysis
guides/evaluate
guides/inspect-runs
guides/add-a-backbone
guides/add-a-cache-processor
```

```{toctree}
:maxdepth: 2
:caption: Campaigns

campaigns/smollm-matched-total
```

```{toctree}
:maxdepth: 2
:caption: Integrations

integrations/weights-and-biases
integrations/hugging-face
integrations/lm-evaluation-harness
integrations/slurm
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/api
reference/configuration
reference/cli
reference/checkpoint-formats
reference/run-format
reference/provenance
reference/compatibility
reference/glossary
```

```{toctree}
:maxdepth: 2
:caption: Development

development/architecture
development/package-boundaries
development/testing
development/documentation
development/invariants
development/refactor-plan
development/migration-history
```
