# Unified `dev` History

The unified branch uses the former `experimental` implementation as its active baseline and retains the former `dev`
tip as the second parent of a history-only merge.

| Former branch | Commit | Tree |
|---|---|---|
| `experimental` | `abd8953e92509e840dd688e6cf4a7014f3ddf3a8` | `62901491ce0b0f838687aaed73d37fd71789a0ba` |
| `dev` | `80da1325fcaa0351f21961a71f328cd9a9f4e2c6` | `9a219a233eba572d02a24d6373950b1a603bd77d` |

Merge `124ea91735de70e820230286aae7bed2c7e7dee4` used Git's `ours` strategy. Its tree exactly matches the former
`experimental` tree. Later commits port the selected analysis workflows and add versioned training resume explicitly;
they do not mechanically combine conflicting source files.

`data/synergy` remains in the active tree. The large `docs/lrc-archive` result bundle remains reachable through the old
`dev` parent but does not appear in normal checkouts.

The `main` and `martin` branches are outside this integration. In particular, `martin` belongs to another contributor
and neither its commits nor its ref participate in the merge or promotion.

## Recovery

Both former tips remain ancestors of unified `dev`. Recreate the old experimental branch without object recovery:

```bash
git push origin abd8953e92509e840dd688e6cf4a7014f3ddf3a8:refs/heads/experimental
```

If a functional integration commit regresses behavior, revert that commit on `dev`. Keep the history-only merge so both
source histories remain reachable; do not force-push `dev`.
