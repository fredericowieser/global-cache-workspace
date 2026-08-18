# First CPU run

**Status:** Maintained

The maintained smoke program constructs TF and BGT, computes losses, admits Longshot, and rejects one incompatible Longshot composition.

```bash
uv run --extra cpu python scripts/quickstart/cpu_smoke.py
```

Success prints two finite losses, `Longshot admitted: longshot`, and one explicit compatibility rejection. CI executes the same file that this page references.

```{literalinclude} ../../scripts/quickstart/cpu_smoke.py
:language: python
:caption: scripts/quickstart/cpu_smoke.py
```
