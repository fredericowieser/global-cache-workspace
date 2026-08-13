# Hugging Face

**Status:** Maintained

GCW uses Hugging Face for tokenizer access, checkpoint conversion, datasets, and selected teacher paths. Native GCW training and evaluation use GCW model classes after conversion.

Conversion must bind the source repository, revision, config, tokenizer, chat template, tensor names, vocabulary mapping, and dtype. Load converted state strictly. Do not use `strict=False` or broad key remapping to hide a conversion mismatch.

Generic writable-cache intervention is unsupported. GCW certifies cache mutation one model family and generation path at a time.
