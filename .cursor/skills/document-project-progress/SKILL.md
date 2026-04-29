---
name: document-project-progress
description: Keeps this Mahjong AI project documentation organized. Use when the user asks to record progress, summarize work, add troubleshooting notes, update README/docs, document model development steps, or after substantial changes that should be reflected in docs.
---

# Document Project Progress

作成日: 2026/04/29

## Purpose

Keep README small and store detailed records under `docs/`.

Use this skill when:

- The user asks to record what was done.
- The user asks for troubleshooting notes.
- The user asks where to put documentation.
- Model, data, training, evaluation, or viewer behavior changes.
- A bug fix affects future dataset generation, training, or interpretation.

## Documentation Map

- `README.md`: project overview, current state, shortest commands, links to docs.
- `docs/index.md`: entry point for all documentation.
- `docs/progress-YYYY-MM-DD.md`: daily or milestone progress records.
- `docs/troubleshooting.md`: symptoms, causes, fixes, checks, regeneration notes.
- `docs/model-development-guide.md`: files and workflow for stronger models.
- `docs/project-structure.md`: where files live and where new files should go.
- `docs/viewer-guide.md`: web viewer and report JSON usage.
- `docs/work-log-policy.md`: documentation rules.

## Update Rules

Do not put long daily logs in `README.md`.

When documenting a completed change:

1. Add or update the relevant `docs/` file.
2. For every newly created Markdown document, including README and skill documents, write `作成日: YYYY/MM/DD` directly under the top-level title.
3. Keep entries concise but include enough context for a future agent.
4. Mention verification commands and whether they passed.
5. Mention when old datasets, checkpoints, or report JSON must be regenerated.
6. Update `README.md` only if the project overview, main commands, or doc links changed.

If an existing Markdown document is edited and it does not have `作成日: YYYY/MM/DD`, add it near the top while preserving the existing content.

## Troubleshooting Entry Template

```markdown
## YYYY/MM/DD Problem Title

作成日: YYYY/MM/DD

### 症状

What the user saw.

### 原因

Why it happened.

### 修正

What changed.

### 確認方法

Commands or checks used.

### 注意点

Regeneration, retraining, or remaining risks.
```

## Progress Entry Template

```markdown
# YYYY/MM/DD 成果記録

作成日: YYYY/MM/DD

## やったこと

- Main changes.

## 変更した主なファイル

- `path/to/file.py`: purpose.

## 確認したこと

- Command or manual check.

## 次にやること

- Follow-up tasks.
```
