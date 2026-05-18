# Semantic Index — ET-research Prior Art

This directory contains a structured semantic index of all downloaded prior art
in `inspiration/`. It is designed for fast lookup by a coding agent.

## File Layout

| File | Contents |
|------|----------|
| `master.yaml` | Top-level topic clusters with cross-references |
| `repos.yaml` | Structured metadata for all cloned repos |
| `papers-architectures.yaml` | Papers on segmentation model design |
| `papers-training.yaml` | Papers on SSL, active learning, label efficiency |
| `papers-datasets-robustness.yaml` | Papers on data, deployment, failure modes |
| `papers-boundary.yaml` | Boundary losses, post-processing, CPS fixes, active learning |
| `tags.yaml` | Inverted index: tag → list of entries |

## How to Use

1. **By topic**: Check `master.yaml` for thematic clusters
2. **By tag**: Check `tags.yaml` for tag-based lookup (e.g., "semi-supervised" → list of papers + repos)
3. **By file**: Each YAML file has `key_files` or `file` fields pointing to exact paths in `inspiration/`
4. **By relevance**: Every entry has a `segunet_takeaway` field summarizing direct applicability
