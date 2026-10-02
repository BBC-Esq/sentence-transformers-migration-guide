# Sentence Transformers Migration Guide

What a program written for [sentence-transformers](https://github.com/huggingface/sentence-transformers) **v5.1.2** (October 22, 2025) has to account for when it upgrades to any later stable release.

## Contents

- [`Sentence Transformers Breaking Changes Comprehensive Guide.md`](Sentence%20Transformers%20Breaking%20Changes%20Comprehensive%20Guide.md): the guide. Changes are organized by subsystem, each with an explanation, before/after code, the upstream pull request, the first release whose code contains it, and what fails or silently changes. It ends with a per-version summary of every release after v5.1.2 and a quick reference of the most likely breakage points.
- [`release-notes/`](release-notes/): a verbatim archive of the upstream release notes for v5.1.2 (the baseline) and every stable release after it. Each file is kept exactly as `gh release view <TAG> --repo huggingface/sentence-transformers --json body --jq '.body'` returns it, byte for byte, and is refreshed if HuggingFace edits a published release. `.gitattributes` stops git from changing the files' line endings.

## How this guide is maintained

1. **List releases.** New stable releases (no pre-releases, no `rc` tags) of `huggingface/sentence-transformers` are checked daily.
2. **Archive the notes.** Each new release's notes are saved to `release-notes/<TAG>.md` unchanged, and the archived notes of earlier releases are re-fetched and compared, so edits made after publication are picked up.
3. **Find candidate changes.** The release notes, the linked pull requests and the project's own `docs/migration_guide.md` are scanned for breaking changes, deprecations, renames, new defaults and dependency changes. The source of each release is also diffed against the previous release (`pyproject.toml`, `__init__.py`, module paths, public signatures, training arguments, losses, evaluators), because not every change is flagged in the notes.
4. **Verify every change against the released code.** Release notes, PR descriptions and deprecation messages are treated as leads, not proof. Each change is confirmed in the source at the tag of the release it is attributed to, and the old code is confirmed at the previous tag. Where reading the source is not conclusive (exception types, warning text, outputs, defaults), the code is run against the published package. A change is attributed to the first release whose code contains it, and the guide notes where the release notes disagree.
5. **Add it to the guide** in the matching subsystem section and in the per-version summaries.
6. **Update the intro's version range and the Quick Reference.**

Patch releases (`vX.Y.Z` with `Z > 0`) are usually bug-fix-only. They still get a per-version summary entry, but they change the body of the guide only when they contain a genuine breaking or behavior change.
