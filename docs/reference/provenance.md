# Provenance and identity

**Status:** Maintained

A reproducible run binds:

- authored and resolved config identities;
- repository or authenticated build identity;
- exact model weights and revision;
- tokenizer and chat template;
- prepared data bytes;
- intervention schedule;
- checkpoint parents;
- runtime environment and relevant kernels.

Config plus Git hash alone cannot reconstruct external model, tokenizer, or dataset state. A provenance manifest identifies those artifacts. A frozen evidence bundle can make the run self-contained.

Historical identity schemas remain unchanged. A new schema verifies old records with an exact reader; it never refreshes old hashes.
