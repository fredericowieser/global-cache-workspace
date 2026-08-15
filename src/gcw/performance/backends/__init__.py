"""Architecture-specific regional execution backends.

There is intentionally no generic backend registry. Each supported region owns
its admission checks, compiler options, provenance, and fail-closed dispatch.
"""
