# Sentence Transformers Breaking Changes Comprehensive Guide

This guide covers every change a program written for **sentence-transformers v5.1.2** (released October 22, 2025) must account for when it upgrades to a later stable release, through **v6.1.0** (released September 18, 2026), the newest release at the time of writing. That spans 15 releases: v5.2.0, v5.2.1, v5.2.2, v5.2.3, v5.3.0, v5.4.0, v5.4.1, v5.5.0, v5.5.1, v5.6.0, v5.6.1, v5.7.0, v6.0.0, v6.0.1 and v6.1.0.

Most of the risk is concentrated in two releases. **v5.4.0** restructured the package and rebuilt `CrossEncoder`; its notes say existing code keeps working "with warnings at worst", but several common patterns fail or silently change. **v6.0.0** raised the dependency floors to transformers v5, huggingface-hub v1 and torch 2.2, and requires `trust_remote_code=True` for custom module classes. If you are short on time, start with the [Quick Reference](#quick-reference-most-likely-breakage-points-when-upgrading-from-v512).

## How this guide is verified

Audited on **2026-10-02** against the released source of every tag from v5.1.2 to **v6.1.0**.

- **Leads, not proof.** Release notes (archived verbatim in [`release-notes/`](release-notes/)), the project's `docs/migration_guide.md`, PR descriptions and deprecation messages were used only to find candidate changes.
- **Source at both tags.** Every change was located in the code of the release it is attributed to (file and line) and the old code was confirmed at the previous tag. Every consecutive pair of tags was diffed (`pyproject.toml`, `__init__.py`, module layout, public signatures, training arguments, losses, evaluators, `docs/migration_guide.md`) to find changes the notes do not mention.
- **Run where reading wasn't enough.** Exception types and messages, warning text, outputs and defaults were checked by running the published packages in throwaway CPU environments with tiny models, mostly with both transformers 4.57 (the last version v5.1.2 allows) and transformers 5.18 (the newest). Error and warning messages are quoted from those runs. The Before/After snippets are based on those runs, and every import line in them was executed: Before imports on v5.1.2, After imports on v6.1.0.
- **Second audit.** After the draft was written, a separate, independent audit re-checked every claim against the code and by running it, without relying on the first pass; everything it confirmed has been corrected.
- **Attribution.** Each change is attributed to the first release whose code contains it. Where the release notes say otherwise, the entry says so ("Notes vs. code").
- **Limits.** A few behaviors need GPUs (flash attention, multi-GPU training) or optional services (Qdrant, Seismic); those are marked as verified from the code only, or attributed to the PR author. PyPI matches the tags for every release in the range, and no release is yanked.

## How to read an entry

Each entry gives the first release whose code contains the change, the pull request, and the impact:

- **hard error**: code that worked on v5.1.2 raises;
- **deprecation warning**: the old form still works and warns (the entry says until when; "still works in v6.1.0" means no removal has happened yet);
- **silent change**: no error, but outputs, training or saved files differ;
- **additive**: new, optional functionality.

"Before" code works on v5.1.2; "After" code is for the release that made the change and, unless the entry says otherwise, also works on v6.1.0. Import paths in the After blocks use the post-v5.4.0 module layout; the old paths still work with a `DeprecationWarning` (see [import paths](#import-paths-moved-old-paths-still-work-with-a-deprecationwarning-v540)).

## Contents

- [Installation and dependencies](#installation-and-dependencies)
- [Model loading and saving](#model-loading-and-saving)
- [SentenceTransformer inference](#sentencetransformer-inference)
- [CrossEncoder](#crossencoder)
- [SparseEncoder](#sparseencoder)
- [MultiVectorEncoder](#multivectorencoder)
- [Modules and import paths](#modules-and-import-paths)
- [Training](#training)
- [Losses](#losses)
- [Evaluators](#evaluators)
- [Utilities](#utilities)
- [Multimodal inputs](#multimodal-inputs)
- [Hidden breakage for custom modules, custom losses and remote code](#hidden-breakage-for-custom-modules-custom-losses-and-remote-code)
- [Per-version summary](#per-version-summary)
- [Quick Reference: most likely breakage points when upgrading from v5.1.2](#quick-reference-most-likely-breakage-points-when-upgrading-from-v512)

## Installation and dependencies

The dependency picture moved twice: v5.2.0 widened the `transformers` cap to `<6`, so fresh installs started pulling transformers v5; v6.0.0 then made transformers v5, huggingface-hub v1 and torch 2.2 hard floors.

| Requirement | v5.1.2 (baseline) | v5.7.0 (last 5.x) | v6.0.0 / v6.1.0 |
|---|---|---|---|
| Python (`requires-python`) | `>=3.9` | `>=3.10` | `>=3.10` |
| `transformers` | `>=4.41.0,<5.0.0` | `>=4.41.0,<6.0.0` | `>=5.0.0,<6.0.0` |
| `huggingface-hub` | `>=0.20.0` | `>=0.23.0` | `>=1.3.0,<2.0.0` |
| `torch` | `>=1.11.0` | `>=1.11.0` | `>=2.2` |
| `numpy` | not declared | `>=1.20.0` | `>=1.24.0` |
| `scikit-learn` | unpinned | `>=0.22.0` | `>=1.1.0` |
| `scipy` | unpinned | `>=1.0.0` | `>=1.0.0` |
| `tokenizers` | not declared | `>=0.19` | `>=0.19` |
| `typing_extensions` | `>=4.5.0` | `>=4.5.0` | `>=4.10.0` |
| `tqdm` | unpinned | `>=4.0.0` | `>=4.0.0` |
| `Pillow` | core dependency | `image` extra (`transformers[vision]`) | `image` extra (`transformers[vision]`) |
| `train` extra | `datasets`, `accelerate>=0.20.3` | `datasets>=2.0.0`, `accelerate>=0.20.3` | `datasets>=2.16.0`, `accelerate>=1.3.0` |
| `onnx` / `onnx-gpu` extra | `optimum[onnxruntime]>=1.23.1` | `optimum-onnx[onnxruntime]` | `optimum-onnx[onnxruntime]` (cannot currently be installed, see below) |
| `openvino` extra | `optimum-intel[openvino]>=1.20.0` | `optimum-intel[openvino]` | `optimum-intel[openvino]>=2.0.0` |
| `audio`, `video` extras | do not exist | `transformers[audio]`, `transformers[video]` | same |

Every row above was read from `pyproject.toml` at the tags, and the PyPI metadata of every release from 5.1.2 to 6.1.0 matches its tag. No release in the range is yanked.

### Python 3.10 or newer is required (v5.2.0)

**First release:** v5.2.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3573 · **Impact:** install-time hard error on Python 3.9

`requires-python` went from `>=3.9` to `>=3.10` and the 3.9 classifier was removed. Nothing warns at runtime; the restriction is enforced by pip.

**What happens:** on Python 3.9, `pip install "sentence-transformers==5.2.0"` (or any later version) fails with `ERROR: Could not find a version that satisfies the requirement sentence-transformers==5.2.0`, after listing every newer release as "Requires-Python >=3.10". An unpinned `pip install -U sentence-transformers` on 3.9 silently stays on 5.1.2.

**Notes vs. code:** the v5.2.0 release notes call this a "deprecation" of Python 3.9. The package metadata drops 3.9 outright.

Before (v5.1.2):
```bash
python3.9 -m pip install "sentence-transformers==5.1.2"
```

After (v5.2.0):
```bash
python3.10 -m pip install "sentence-transformers==5.2.0"   # any Python >= 3.10
```

### Fresh installs of v5.2.x get transformers v5 (v5.2.0)

**First release:** v5.2.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3586 · **Impact:** environment change

v5.2.0 raised the cap from `transformers<5.0.0` to `<6.0.0`. The v5.2.0 package contains no transformers-v5-specific code; the compatibility fixes arrived in the v5.2.1, v5.2.2 and v5.2.3 patches. When v5.2.0 shipped, transformers v5 existed only as release candidates, but a fresh `pip install "sentence-transformers==5.2.x"` today resolves transformers 5.x and huggingface-hub 1.x. An existing 5.1.2 environment upgraded to a pinned 5.x release (`pip install -U "sentence-transformers==5.2.3"`) keeps its transformers 4.x unless you upgrade transformers too; an unversioned `pip install -U sentence-transformers` today installs 6.1.0 and upgrades transformers to 5.x and huggingface-hub to 1.x.

Three crashes hit v5.2.x on transformers v5. Use v5.2.3 or later, or pin `transformers<5`:

| Symptom | Affected | Fixed in |
|---|---|---|
| `ModuleNotFoundError: No module named 'requests'` at `import sentence_transformers` | 5.2.0, 5.2.1 on a requests-free install (transformers 5 / huggingface-hub 1 no longer pull in `requests`) | [v5.2.2](#v522) |
| `AttributeError: 'SentenceTransformerTrainingArguments' object has no attribute 'save_safetensors'` at the first checkpoint save | 5.2.0 with transformers 5.x | [v5.2.1](#v521) |
| `AttributeError: 'SentenceTransformerTrainer' object has no attribute '_nested_gather'` at the first logging step | 5.2.0 to 5.2.2 with transformers 5.2 or newer | [v5.2.3](#v523) |

All three were reproduced in clean environments and confirmed fixed in the listed release.

Before (v5.1.2):
```bash
pip install "sentence-transformers==5.1.2"                    # always transformers 4.x
```

After (v5.2.3):
```bash
pip install "sentence-transformers==5.2.3"                    # gets transformers 5.x today
pip install "sentence-transformers==5.2.3" "transformers<5"   # stay on transformers 4.x
```

With transformers 5, `TrainingArguments` no longer has `save_safetensors`, so passing `save_safetensors=...` to `SentenceTransformerTrainingArguments` raises `TypeError: ... got an unexpected keyword argument 'save_safetensors'`. That is a transformers change; drop the argument (safetensors is always used).

### Pillow is no longer installed by default (v5.2.0)

**First release:** v5.2.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3589 · **Impact:** hard error for image models on fresh installs

`Pillow` moved from the core dependencies to a new `image` extra. `import sentence_transformers` and `from sentence_transformers.models import CLIPModel` still work without it, but loading a CLIP-style model fails. Upgrading an existing environment does not uninstall Pillow, so this mostly affects fresh installs and regenerated lock files. In v5.4.0 the extra became `image = ["transformers[vision]"]`, which with transformers 5.1 or newer also installs `torchvision` (it must match your torch build, so install it from the same index as torch).

**What fails:** with transformers 4.x, `ImportError: CLIPImageProcessor requires the PIL library but it was not found in your environment. ...`; with transformers 5.x, `ValueError: Could not load any image processor class for <model>. ... Missing optional dependencies: torchvision, Pillow. ...`.

Before (v5.1.2):
```bash
pip install sentence-transformers==5.1.2            # Pillow included
```

After (v5.2.0):
```bash
pip install "sentence-transformers[image]==5.2.0"   # or: pip install sentence-transformers pillow
```

### New `audio` and `video` extras, and explicit floors for numpy, scikit-learn, scipy and tqdm (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** additive

v5.4.0 added `audio = ["transformers[audio]"]` and `video = ["transformers[video]"]`, changed `image` to `transformers[vision]`, and set floors on `numpy>=1.20.0` (declared without a floor since v5.2.1, https://github.com/huggingface/sentence-transformers/pull/3608), `scikit-learn>=0.22.0`, `scipy>=1.0.0`, `tqdm>=4.0.0` and `huggingface-hub>=0.23.0` (the last has no practical effect, because `transformers>=4.41.0` already requires it). What the extras pull in depends on your transformers version; with transformers 4.57 `[audio]` includes `kenlm`, which can fail to build on some platforms.

Nothing to change for text-only code. Install `[image]`, `[audio]` or `[video]` only if you pass those inputs.

### `onnx` / `openvino` extras now point to `optimum-onnx` and `optimum-intel` (v5.2.0); the `onnx` extra cannot be installed with v6

**First release:** v5.2.0 (extras re-pointed), v6.0.0 (transformers floor) · **PR:** https://github.com/huggingface/sentence-transformers/pull/3587 · **Impact:** install-time conflict

v5.2.0 changed `onnx = ["optimum[onnxruntime]>=1.23.1"]` to `["optimum-onnx[onnxruntime]"]` (same for `onnx-gpu`) and `openvino` to an unpinned `optimum-intel[openvino]`; v6.0.0 pins `optimum-intel[openvino]>=2.0.0`. On 5.x the extras resolve to the same packages as before.

The catch is outside Sentence Transformers: every `optimum-onnx` release on PyPI as of this audit (0.0.1 to 0.1.0) requires `transformers<4.58`. Resolved on the audit date:

- `sentence-transformers[onnx]==6.0.0`, `[onnx]==6.1.0` and `[onnx-gpu]==6.1.0` are **unsatisfiable** (`optimum-onnx` needs transformers <4.58, Sentence Transformers 6 needs >=5.0.0).
- An unpinned `pip install "sentence-transformers[onnx]"` silently resolves to **5.7.0** with transformers 4.57.
- `sentence-transformers[onnx]` on any 5.x release also keeps you on transformers 4.x.
- `sentence-transformers[openvino]==6.1.0` resolves (optimum-intel 2.2.0).

If you use `backend="onnx"`, stay on 5.7.0 until an `optimum-onnx` release supports transformers 5, and pin it explicitly so the resolver doesn't pick it for you:

```bash
pip install "sentence-transformers[onnx]==5.7.0"
```

### Source builds need newer setuptools; license metadata uses PEP 639 (v5.5.0, v5.6.0)

**First release:** v5.5.0, v5.6.0 · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3745, https://github.com/huggingface/sentence-transformers/pull/3809 · **Impact:** only for source builds and license tooling

Wheel installs and runtime dependencies are unaffected. Building from the sdist requires `setuptools>=64` from v5.5.0 and `setuptools>=77.0.3` from v5.6.0, which matters with `--no-build-isolation` or an old pinned setuptools. From v5.6.0 the installed metadata says `License-Expression: Apache-2.0` instead of `License: Apache 2.0` plus the `License :: OSI Approved :: Apache Software License` classifier, so license scanners keyed on the classifier stop detecting Apache.

```bash
pip install "setuptools>=77.0.3"   # before building v5.6.0+ from source without build isolation
```

### `tokenizers` is a declared dependency (v5.7.0)

**First release:** v5.7.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3844 · **Impact:** additive

`tokenizers>=0.19` is now listed explicitly. Every supported transformers version already required it, so this only matters for strict or minimal resolvers. No action needed.

### v6.0.0 requires transformers v5, huggingface-hub v1 and torch 2.2 (v6.0.0)

**First release:** v6.0.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3794 · **Impact:** install-time hard error if something pins the old versions; otherwise silent upgrades

v6.0.0 raised the floors in the table at the top of this section (`transformers>=5.0.0,<6.0.0`, `huggingface-hub>=1.3.0,<2.0.0`, `torch>=2.2`, `numpy>=1.24.0`, `scikit-learn>=1.1.0`, `typing_extensions>=4.10.0`; `train`: `datasets>=2.16.0`, `accelerate>=1.3.0`; `openvino`: `optimum-intel[openvino]>=2.0.0`). Python stays at `>=3.10`.

**What happens:**
- Without conflicting pins, pip upgrades transformers 4.x to 5.x and huggingface-hub 0.x to 1.x (and datasets/accelerate with `[train]`). Those upgrades bring their own behavior changes, such as transformers v5 deprecating `warmup_ratio` (removed in transformers 5.15) (see [`warmup_ratio`](#warmup_ratio-is-kept-as-a-sentence-transformers-argument-v530)).
- With a pin such as `transformers<5` or `huggingface-hub<1.0`, `pip install "sentence-transformers==6.0.0" ...` fails with `ResolutionImpossible`.
- With such a pin and an *unversioned* `pip install -U sentence-transformers`, pip silently installs 5.7.0 instead.
- There is no runtime version check: a forced `--no-deps` install of 6.0.0 on transformers 4.57 still imports and encodes; only `pip check` reports the mismatch. That combination is unsupported.

**Notes vs. code:** the v6.0 section of the project's migration guide omits the `typing_extensions>=4.10.0` and `optimum-intel[openvino]>=2.0.0` floors that the code and the release notes have. `huggingface_hub` 2.x is out, but 6.1.0 caps it below 2.0, so 1.33.0 is the newest that installs.

Before (v5.1.2):
```text
# requirements.txt
sentence-transformers==5.1.2
transformers==4.57.6
huggingface-hub==0.36.2
accelerate==0.34.2
datasets==2.15.0
```

After (v6.0.0):
```text
# requirements.txt
sentence-transformers[train]==6.0.0
transformers>=5.0.0,<6.0.0
huggingface-hub>=1.3.0,<2.0.0
torch>=2.2
accelerate>=1.3.0
datasets>=2.16.0
```

## Model loading and saving

The upgrade direction is safe: models saved by v5.1.2 (SentenceTransformer with Normalize and prompts, CrossEncoder, SparseEncoder, Router, CLIP) load in v5.4.0 and v6.x with identical outputs. The exceptions are custom module classes, which need `trust_remote_code=True` from v6.0.0. The downgrade direction is not safe: models saved by newer releases often cannot be read by older ones.

### Custom module classes require `trust_remote_code=True`, including local directories (v5.6.0 warning, v6.0.0 error)

**First release:** v5.6.0 (`FutureWarning` for local custom code), v5.7.0 (`FutureWarning` for third-party classes from Hub models), v6.0.0 (`ValueError`) · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3807, https://github.com/huggingface/sentence-transformers/pull/3860, https://github.com/huggingface/sentence-transformers/pull/3935 · **Impact:** hard error from v6.0.0

A model's `modules.json` names each module class by import path. In v5.1.2, any class outside `sentence_transformers.*` was imported without a trust check whenever the path was a local directory, and classes from installed packages were imported silently in every case. The timeline:

| Version | Custom code file in a **local** model directory | Class from another installed package (your own project, a library), referenced by a local model | Class from an installed package, referenced by a **Hub** repo |
|---|---|---|---|
| v5.1.2 – v5.5.1 | loads silently | loads silently | loads silently |
| v5.6.x | loads, `FutureWarning` | loads silently | loads silently |
| v5.7.0 | loads, `FutureWarning` | loads silently (if the model is a local directory) | loads, `FutureWarning` |
| v6.0.0 + | `ValueError` | `ValueError` | `ValueError` |

Code files hosted in a Hub repository always needed `trust_remote_code=True`: without it, every version fails (with `ModuleNotFoundError` up to v5.7.0, after the v5.7.0 `FutureWarning`, and with the `ValueError` below from v6.0.0).

The v5.7.0 `FutureWarning` reads: `` The model '<repo>' references the module class '<class>', which is not part of Sentence Transformers. Importing it executes third-party code, and will start requiring `trust_remote_code=True` from Sentence Transformers v6.0. ... ``. The v5.7.0 change also routes the `tokenizer_class` of a `WordEmbeddings` module through the same check.

**What fails (v6.0.0, also v6.1.0):** `ValueError: The model <path> references the module class '<module.Class>', which is not part of Sentence Transformers. Importing it executes third-party code. You can inspect the repository content at <absolute path or https://hf.co/<repo>>.` followed by `` Please pass the argument `trust_remote_code=True` to allow custom code to be run. `` It is raised by `SentenceTransformer(...)`, `CrossEncoder(...)`, `SparseEncoder(...)` and `MultiVectorEncoder(...)`. With `trust_remote_code=True` such models load with outputs identical to v5.1.2.

**Notes vs. code:** the v6.0.0 notes say this "completes the deprecation cycle announced in v5.6 and v5.7". The middle column never warned: a class from your own installed package, referenced by a model saved in a local directory, loaded silently on v5.7.0 and raises on v6.0.0.

Trainer checkpoint reloading has its own wrinkle; see [Trainer reload of custom modules](#trainer-checkpoint-reload-of-custom-modules-v600).

Before (v5.1.2):
```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer
from my_project.modules import MyCustomPooling  # your own nn.Module

model = SentenceTransformer(modules=[Transformer("bert-base-uncased"), MyCustomPooling()])
model.save("my-model")
model = SentenceTransformer("my-model")  # loaded silently
```

After (v6.0.0):
```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Transformer
from my_project.modules import MyCustomPooling

model = SentenceTransformer(modules=[Transformer("bert-base-uncased"), MyCustomPooling()])
model.save("my-model")
model = SentenceTransformer("my-model", trust_remote_code=True)  # required, local directories included
```

`trust_remote_code=True` is accepted on every version back to v5.1.2, so the After line works on both.

### A file path passed as the model raises `NotADirectoryError` (v5.7.0)

**First release:** v5.7.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3860 · **Impact:** exception type changes

Passing a file (for example `model.safetensors`) instead of the model directory used to fall through to Hub loading. On v5.1.2, `SentenceTransformer` raised `huggingface_hub`'s `HFValidationError` (a `ValueError`; for example `Repo id must be in the form 'repo_name' or 'namespace/repo_name': ...`, or `Repo id must use alphanumeric chars ...`, depending on the path) or transformers' `OSError` for a bare file name, and `CrossEncoder` and `SparseEncoder` raised `OSError: It looks like the config file at '<path>' is not a valid JSON file.` Now all three raise `` NotADirectoryError: Path <path> is a file, not a directory. `model_name_or_path` must be a local directory containing a model, or a Hugging Face Hub model ID. `` `NotADirectoryError` subclasses `OSError`, so `except OSError` catches it, but an `except ValueError` (or `except HFValidationError`) around `SentenceTransformer(...)` no longer does.

Before (v5.1.2):
```python
model = SentenceTransformer("models/my-model/model.safetensors")   # confusing Hub error
```

After (v5.7.0):
```python
model = SentenceTransformer("models/my-model")                     # pass the directory
```

### Hub errors propagate as `huggingface_hub` exceptions (v5.5.0)

**First release:** v5.5.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3765 · **Impact:** exception type and message change

Missing, private or gated repositories and bad revisions used to be swallowed during the `modules.json` probe and then failed inside transformers with a plain `OSError` (in v5.1.2, after a misleading "Creating a new one with mean pooling" warning). They now raise the `huggingface_hub` error directly: `RepositoryNotFoundError`, `GatedRepoError` or `RevisionNotFoundError`. All of these subclass `OSError`. If `model_name_or_path` is an existing local directory, the Hub is no longer consulted for missing files. Offline mode with a warm cache is unaffected.

**What breaks:** `except OSError` keeps working; code that matches transformers' old message text (for example `"is not a valid model identifier"`) or checks `type(e) is OSError` does not.

Before (v5.1.2):
```python
try:
    model = SentenceTransformer(name)
except OSError as e:
    if "is not a valid model identifier" in str(e):
        ...
```

After (v5.5.0):
```python
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError

try:
    model = SentenceTransformer(name)
except (RepositoryNotFoundError, RevisionNotFoundError, GatedRepoError):
    ...
except OSError:
    ...
```

### Models saved by v5.4.0 or later cannot be loaded by v5.3.x or earlier (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** hard error when downgrading or mixing versions

The v5.4.0 restructure changed the saved format:

- `modules.json` records the new class paths, for example `sentence_transformers.base.modules.transformer.Transformer` and `sentence_transformers.sentence_transformer.modules.pooling.Pooling`.
- `sentence_bert_config.json` no longer stores `max_seq_length` and `do_lower_case`; it stores `transformer_task`, `modality_config` and `module_output_name`. `max_seq_length` is persisted only as `model_max_length` in `tokenizer_config.json`, and `do_lower_case=True` is baked into the fast tokenizer's normalizer.
- `1_Pooling/config.json` uses `embedding_dimension` and `pooling_mode` instead of `word_embedding_dimension` and six `pooling_mode_*` booleans. Dense configs gain `module_input_name`/`module_output_name`; Router configs gain `route_mappings`.

**What fails:** loading a v5.4+ SentenceTransformer or SparseEncoder save with v5.1.2 or v5.3.0 raises `ModuleNotFoundError: No module named 'sentence_transformers.base'` (or `'sentence_transformers.sentence_transformer'`, depending on the first module, for example for CLIP). CrossEncoder saves are different; see the next entry.

A second trap affects CLIP-style models whose modality config uses `get_text_features`/`get_image_features`. The inferred `method_output_name` (`null` under transformers 4, `"pooler_output"` under transformers 5) is saved, so a CLIP-style model saved with Sentence Transformers 5.4+ on transformers 4.x fails to load on transformers 5.x, including every v6 release (`AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'ndim'`), and a save made on transformers 5 fails on transformers 4 (`IndexError: too many indices for tensor of dimension 2`). CLIP models saved by v5.1.2 load fine on both.

Before (v5.1.2):
```python
model.save("out")   # loadable by any 5.x
```

After (v5.4.0):
```python
model.save("out")   # loadable only by sentence-transformers >= 5.4.0
# Keep an untouched copy for services still on <= 5.3, and save CLIP-style models
# under the transformers major version that will load them.
```

### CrossEncoder save format: `modules.json` and `config_sentence_transformers.json`; activation moved out of `config.json` (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** silent score change when an older release loads a newer save

v5.1.2 saved a CrossEncoder as plain Hugging Face files and stored the activation in `config.json` (`"sentence_transformers": {"activation_fn": ..., "version": ...}`). v5.4.0 also writes `modules.json`, `sentence_bert_config.json` and `config_sentence_transformers.json` (`model_type`, `activation_fn`, `prompts`, `default_prompt_name`), and no longer writes the activation into `config.json`. As in v5.1.2, the legacy `sbert_ce_default_activation_function` key that some Hub rerankers carry (for example `cross-encoder/ms-marco-MiniLM-L6-v2`, whose activation is `Identity`) is read and dropped on load, so re-saving such a model moves its activation into `config_sentence_transformers.json` only.

**What happens:**
- Upgrading is safe: v5.1.2-saved folders load in v5.4.0 and v6.1.0 with identical scores, and the activation is still read from `config.json`.
- Downgrading silently loses a non-default activation. A v5.4+ save loads in v5.1.2 without error (old CrossEncoder ignores `modules.json`), but falls back to `Sigmoid` for one-label models, so logits become probabilities. Rankings are unchanged; absolute thresholds break. v5.1.2 emits no warning, because the version key it checks is no longer written.

After (v5.4.0), if an older release must read the folder with the same activation:
```python
from sentence_transformers.util import fullname

model.model.config.sentence_transformers = {"activation_fn": fullname(model.activation_fn)}
model.save("out")   # v5.1.2 now loads it with the same activation
```

### `Normalize` moved to `sentence_transformers.base.modules`; v6 saves containing it cannot be loaded by older releases (v6.0.0)

**First release:** v6.0.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3794 · **Impact:** downgrade break for new saves; deprecation warning for one import path

`Normalize` now lives in `sentence_transformers/base/modules/normalize.py` and writes a `config.json` with `module_input_name`/`module_output_name`. Old references in `modules.json` (`sentence_transformers.models.Normalize` from v5.1.2, `sentence_transformers.sentence_transformer.modules.normalize.Normalize` from v5.4 to v5.7) are aliased, so loading old models is silent.

**What fails:** a v6 save that contains `Normalize` raises `ModuleNotFoundError: No module named 'sentence_transformers.base.modules.normalize'` on v5.4 to v5.7 (and `No module named 'sentence_transformers.base'` on v5.3 and earlier, as for other v5.4+ SentenceTransformer saves). A CrossEncoder saved by v6.0.0 still loads in v5.1.2. Importing the submodule path `sentence_transformers.sentence_transformer.modules.normalize` gives a `DeprecationWarning`; `from sentence_transformers.sentence_transformer.modules import Normalize` does not.

Before (v5.1.2):
```python
from sentence_transformers.models import Normalize
```

After (v6.0.0):
```python
from sentence_transformers.base.modules import Normalize
```

### Subclasses of the model classes inherit the base `model_type` (v5.5.0)

**First release:** v5.5.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3763 · **Impact:** silent change in what a subclass loads and saves

`model_type` used to come from the instance's class name, so `class MyST(SentenceTransformer)` saved `"model_type": "MyST"`. It is now a class attribute fixed on `SentenceTransformer`, `SparseEncoder` and `CrossEncoder` and inherited by subclasses. When the saved `model_type` differs from the loading class, the loader ignores `modules.json` and builds default modules (Transformer + mean pooling).

**What changes for `class MyST(SentenceTransformer)`:**
1. Fix: `MyST("<normal checkpoint>")` used to build Transformer + mean pooling (with a warning in v5.1.2). From v5.5.0 it loads the real pipeline, so embeddings change and become correct.
2. Regression: a checkpoint saved through the subclass on v5.4.1 or earlier has `"model_type": "MyST"`. From v5.5.0, `MyST(...)` silently drops its `Normalize`/`Dense` modules and saved pooling mode. v5.5.0 logs this only at INFO; v6.x logs `Converting MyST model <path> to SentenceTransformer.` at WARNING.
3. Saving through a subclass now writes `"model_type": "SentenceTransformer"`, and `model.model_type` returns the base name.

Before (v5.1.2):
```python
from sentence_transformers import SentenceTransformer

class MyST(SentenceTransformer):
    pass

model = MyST("path/saved-by-MyST-on-v5.1.2")   # loaded the full pipeline
```

After (v5.5.0), a one-time fix for such checkpoints:
```python
import json, pathlib

cfg = pathlib.Path("path/saved-by-MyST-on-v5.1.2/config_sentence_transformers.json")
data = json.loads(cfg.read_text())
data["model_type"] = "SentenceTransformer"   # was "MyST"
cfg.write_text(json.dumps(data, indent=2))
model = MyST("path/saved-by-MyST-on-v5.1.2")  # full pipeline again
```

Declaring `model_type = "MyST"` on the subclass also restores the old loading for those checkpoints, but then normal checkpoints take the conversion path for that subclass.

### `model.config` exists and is read-only until v5.7.0 (v5.5.0)

**First release:** v5.5.0 (read-only property), v5.7.0 (setter added) · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3764, https://github.com/huggingface/sentence-transformers/pull/3831 · **Impact:** hard error on assignment in v5.5.0–v5.6.1

`SentenceTransformer` and `SparseEncoder` had no `config` attribute in v5.1.2, and a common workaround for DeepSpeed and Hugging Face `Trainer` integrations was `model.config = model[0].auto_model.config`. v5.5.0 added a `config` property that returns the backbone's `PretrainedConfig` (or `None`), without a setter.

**What fails:** in v5.5.0 to v5.6.1, assignment raises `AttributeError: property 'config' of 'SentenceTransformer' object has no setter` (Python 3.12 wording). From v5.7.0 assignment works again, but it replaces the backbone's config rather than storing a separate attribute. Code that uses `hasattr(model, "config")` to detect a Hugging Face model now gets `True` for Sentence Transformers models.

Before (v5.1.2):
```python
model.config = model[0].auto_model.config   # so integrations find model.config.hidden_size
```

After (v5.5.0):
```python
hidden = model.config.hidden_size            # built in; drop the assignment
```

### `device_map` controls placement and `device` is ignored when both are given (v5.6.0)

**First release:** v5.6.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3823 · **Impact:** silent change in device placement (multi-GPU)

Previously the backbone was placed according to `model_kwargs={"device_map": ...}` and then the whole model was moved to `device` (or the auto-detected default), which defeated `device_map`. Now, with a `device_map`, the backbone stays where accelerate put it and the other modules follow it. Passing `device` as well logs `` Both `device` and `model_kwargs['device_map']` were provided. `device_map` controls device placement, so the `device` argument is ignored. `` The placement itself was verified from the code only (needs more than one GPU). Applies to all model types.

Before (v5.1.2):
```python
model = SentenceTransformer(name, device="cuda:1", model_kwargs={"device_map": "auto"})   # ended up on cuda:1
```

After (v5.6.0):
```python
model = SentenceTransformer(name, model_kwargs={"device_map": "cuda:1"})   # or only device="cuda:1"
```

### Raw causal-LM checkpoints get last-token pooling and a chat template (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** silent, completely different embeddings

This applies when a SentenceTransformer is built from a checkpoint without `modules.json` (for example `SentenceTransformer("Qwen/Qwen2-0.5B")`) or when you construct a `Transformer` module directly:

1. If `config.architectures[0]` ends in `ForCausalLM`, the automatically added `Pooling` uses `"lasttoken"` instead of `"mean"`.
2. Separately, if the tokenizer has a `chat_template`, a `"message"` modality is added and plain text is wrapped in the chat template before encoding. For `Qwen/Qwen2-0.5B` this prepends the template's default system turn (`You are a helpful assistant`; Qwen2.5-Instruct templates use `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`); for Llama-2-style templates it adds `[INST] ... [/INST]`.

Both are saved into the model's configs. Models saved by v5.3.0 or earlier (they have `modules.json`) are unaffected and still load text-only. v5.1.2's WARNING "No sentence-transformers model found ... Creating a new one with mean pooling." is replaced by an INFO log (`No modules.json found for <path>, initializing a new SentenceTransformer model.`), so nothing signals the change.

**Notes vs. code:** the v5.4.0 notes and migration guide mention only the last-token default, and the migration guide's "restore mean pooling" recipe still applies the chat template. The recipe below reproduced v5.3.0 embeddings exactly.

Before (v5.1.2):
```python
model = SentenceTransformer("Qwen/Qwen2-0.5B")   # Transformer + mean pooling over the raw text
```

After (v5.4.0), to reproduce v5.1.2:
```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling, Transformer

transformer = Transformer(
    "Qwen/Qwen2-0.5B",
    modality_config={"text": {"method": "forward", "method_output_name": "last_hidden_state"}},
    module_output_name="token_embeddings",
)
model = SentenceTransformer(
    modules=[transformer, Pooling(transformer.get_embedding_dimension(), pooling_mode="mean")]
)
```

Related: from v5.6.0, inputs rendered through a chat template that are truncated keep the template's closing tokens; see [chat-template truncation](#truncated-chat-template-inputs-keep-the-template-suffix-v560).

### Models can declare required package versions (v6.0.0)

**First release:** v6.0.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3934 · **Impact:** hard error for models that declare unmet requirements

`config_sentence_transformers.json` may contain a `requirements` dictionary of PEP 440 specifiers (for example `{"transformers": ">=5.15"}`); a value may also be `{"specifier": ..., "reason": ...}`, and the reason is added to the error. It is checked before any module loads.

**What fails:** an unmet requirement raises `ImportError` listing each requirement (`- transformers>=99.0, but transformers==5.18.0 is installed.`) and a `pip install -U ...` line. Unparsable specifiers or a non-dictionary value only log a WARNING. Older releases ignore the key, `save()` does not write it, and re-saving a loaded model drops it.

This already matters in practice: on 2026-08-31 `lightonai/ColBERT-Zero`, `ColBERT-Zero-supervised` and `ColBERT-Zero-unsupervised` (not their `-noprompts` variants) added a `sentence-transformers` requirement with specifier `>=6.0.1` and a reason, so v6.0.0 refuses to load them (see [the v6.0.1 MultiVectorEncoder fix](#pylate-query--document-prefixes-are-combined-with-saved-prompts-v601)).

### New warnings when loading (v6.0.0)

**First release:** v6.0.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3794 · **Impact:** log output only

- Passing both `model_name_or_path` and `modules=` now logs `` Both `model_name_or_path` and `modules` were provided. The modules are loaded from `model_name_or_path`, so the `modules` argument is ignored. `` The behavior (modules ignored) is the same as in v5.1.2.
- Loading a checkpoint of one model type as another (for example `CrossEncoder(<SentenceTransformer checkpoint>)`) logs `Converting <type> model <path> to <type>.` at WARNING instead of INFO.
- `Dense` configs with unknown keys load with a WARNING instead of raising `TypeError`.

**Notes vs. code:** the v6.0.0 notes list "cross-family conversion no longer inherits inference settings" among the breaking changes. Compared with v5.7.0 nothing changed: the inheriting code existed only on the unreleased development branch, and converting checkpoints gave identical settings on v5.7.0 and v6.0.0. Only the log level changed. Contrary to the notes, `SparseEncoder(<SentenceTransformer checkpoint>)` still inherits the checkpoint's `prompts`, `default_prompt_name` and `similarity_fn_name` on v5.7.0 through v6.1.0 (a code comment says this is intended), so a saved default prompt is still prepended to every `encode`; set `model.default_prompt_name = None` after loading if you don't want that.

## SentenceTransformer inference

The v5.4.0 notes say "there are no hard deprecations, all existing code should continue to work with warnings at worst." Several common v5.1.2 patterns in this section fail on v5.4.0 and are still broken in v6.1.0: positional constructor arguments, assigning `model.tokenizer`, pandas `Series` inputs and `encode(output_value=None)` on larger batches.

### Constructor arguments after the model name are keyword-only (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** hard error

v5.1.2 accepted `SentenceTransformer(name, modules, device, prompts, ...)` positionally. v5.4.0 made every argument after `model_name_or_path` keyword-only (and moved `similarity_fn_name` and `truncate_dim` to the end). Unlike `CrossEncoder`, there is no deprecation shim.

**What fails:** `TypeError: SentenceTransformer.__init__() takes from 1 to 2 positional arguments but 4 were given`. Still the case in v6.1.0.

Before (v5.1.2):
```python
model = SentenceTransformer("all-MiniLM-L6-v2", None, "cuda")
model = SentenceTransformer(None, [word_embedding_model, pooling_model])
```

After (v5.4.0; also valid on v5.1.2):
```python
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
```

### `encode(sentences=...)` becomes `inputs=`, and `tokenizer_kwargs` becomes `processor_kwargs` (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** deprecation warning; still works in v6.1.0

The first parameter of `encode`, `encode_query` and `encode_document` is now `inputs` (it also accepts dicts, images and chat messages). Passing it positionally is unaffected. The constructor's `tokenizer_kwargs` is now `processor_kwargs`, forwarded to `AutoProcessor.from_pretrained`. Both old names still work and log a warning once per process, through `logging` rather than `warnings`, so `-W error` and `pytest.warns` do not catch it: `` The `sentences` argument was renamed and is now deprecated. Please use `inputs` instead. `` / `` The `tokenizer_kwargs` argument was renamed and is now deprecated. Please use `processor_kwargs` instead. `` No removal version is announced.

`processor_kwargs` is not accepted by v5.1.2. Code that must run on both should keep `tokenizer_kwargs` and accept the warning. Note the truncation side effect in the next entry.

Before (v5.1.2):
```python
emb = model.encode(sentences=texts)
model = SentenceTransformer("all-MiniLM-L6-v2", tokenizer_kwargs={"padding_side": "left"})
```

After (v5.4.0):
```python
emb = model.encode(texts)   # or inputs=texts
model = SentenceTransformer("all-MiniLM-L6-v2", processor_kwargs={"padding_side": "left"})
```

### `max_seq_length` now follows `tokenizer.model_max_length` (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** silent change in truncation length

In v5.1.2, `Transformer.max_seq_length` was a plain attribute and the tokenizer was always called with `max_length=max_seq_length`, so `tokenizer.model_max_length` did not affect truncation. From v5.4.0, `max_seq_length` is a property that reads and writes `tokenizer.model_max_length`, and the tokenizer is called without `max_length`.

**What changes (verified on v5.4.0 and v6.1.0):**
- For a model saved with a `max_seq_length` (any Sentence Transformers model), `tokenizer_kwargs={"model_max_length": 8}` used to be ignored for truncation (52 tokens encoded); it now truncates to 8. Raw Hugging Face checkpoints already used it in v5.1.2.
- With both `max_seq_length=64` and `tokenizer_args={"model_max_length": 48}` on `Transformer`, the result is 48 (was 64).
- Setting `model.tokenizer.model_max_length` now changes truncation, and setting `model.max_seq_length` changes the tokenizer.

**Notes vs. code:** the migration guide presents `tokenizer_kwargs={"model_max_length": ...}` to `processor_kwargs={...}` as a plain rename.

Before (v5.1.2):
```python
# model_max_length was effectively ignored; truncation used the saved max_seq_length
model = SentenceTransformer("my-model", tokenizer_kwargs={"model_max_length": 128})
```

After (v5.4.0):
```python
model = SentenceTransformer("my-model")   # keep the saved max_seq_length
model.max_seq_length = 128                 # change it in exactly one place, if needed
```

### `model.tokenizer` can no longer be assigned, and assigning `auto_model` is silently ignored (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** hard error (`tokenizer`); silent (`auto_model`)

The `Transformer` module now stores the processor in `.processor` and the Hugging Face model in `.model`. `tokenizer` and `auto_model` are read-only properties derived from them.

**What fails:**
- `model.tokenizer = tok` raises a misleading `AttributeError: The first module (Transformer) does not have a 'tokenizer' attribute.`; `model[0].tokenizer = tok` raises `AttributeError: property 'tokenizer' of 'Transformer' object has no setter`.
- `model[0].auto_model = new_model` raises nothing: `nn.Module` registers `new_model` as an extra, unused submodule named `auto_model`, the forward pass keeps using the old model, the parameter count and `model.state_dict()` double (extra `0.auto_model.*` keys), and `model.save()` writes the original model, not the one you assigned. Patterns such as `model[0].auto_model = torch.compile(model[0].auto_model)` silently stop working.

Both still behave this way in v6.1.0.

Before (v5.1.2):
```python
model.tokenizer = AutoTokenizer.from_pretrained("my-tokenizer")
model[0].auto_model = my_modified_hf_model
```

After (v5.4.0):
```python
max_len = model.max_seq_length
model[0].processor = AutoTokenizer.from_pretrained("my-tokenizer")
model.max_seq_length = max_len            # truncation now follows the new tokenizer's model_max_length
model[0].model = my_modified_hf_model
```

### `tokenize()` is deprecated and its output gains a `"modality"` string (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** hard error for code that treats every value as a tensor; deprecation warning

`model.tokenize()` now calls `model.preprocess()` and logs `` The `tokenize` method is deprecated, please use `preprocess` instead. `` once. The returned dict also contains `"modality": "text"` (a `str`), and `"prompt_length"` when a prompt is used.

**What fails:** `{k: v.to(device) for k, v in model.tokenize(texts).items()}` raises `AttributeError: 'str' object has no attribute 'to'`. Passing the dict straight to the Hugging Face model, `model[0].auto_model(**features)`, raises `TypeError: BertModel.forward() got an unexpected keyword argument 'modality'` with transformers 4.x (transformers 5 models accept extra keyword arguments). `model(features)` and `util.batch_to_device` still work.

Before (v5.1.2):
```python
features = model.tokenize(texts)
features = {k: v.to(device) for k, v in features.items()}
```

After (v5.4.0):
```python
from sentence_transformers.util import batch_to_device

features = batch_to_device(model.preprocess(texts), device)   # skips non-tensor entries
hf_inputs = {k: v for k, v in features.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
```

### `encode(output_value=None)` fails for batches of five or more inputs (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** hard error (regression, still present in v6.1.0)

With `output_value=None`, `encode` splits each feature per input with `value[idx]` and only catches `TypeError`. The new `"modality": "text"` string is indexable, so input *i* receives `"text"[i]`.

**What fails:** `IndexError: string index out of range` as soon as a batch holds more than four inputs. With four or fewer, each result dict has a junk `"modality"` entry (`'t'`, `'e'`, `'x'`, ...).

Before (v5.1.2):
```python
outs = model.encode(sentences, output_value=None)
```

After (v5.4.0 to v6.1.0), as a workaround:
```python
outs = model.encode(sentences, output_value=None, batch_size=4)   # ignore the "modality" key
# or request what you need separately:
sentence_emb = model.encode(sentences)
token_emb = model.encode(sentences, output_value="token_embeddings")
```

### `encode()` rejects pandas Series and non-string items (v5.4.0)

**First release:** v5.4.0 (break); v5.4.1 (numpy arrays fixed) · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3554, https://github.com/huggingface/sentence-transformers/pull/3720 · **Impact:** hard error

v5.1.2 accepted any indexable sequence (including a pandas `Series` with a default index) as a batch and, as long as each batch's longest item was a string, converted every item with `str(s).strip()`, so stray `None`, `NaN` or numbers were embedded as `"None"`, `"nan"` or digits. From v5.4.0 only `list`, `tuple` and `datasets.Column` are batches, and each item must be a `str`, dict, PIL image, numpy array or tensor.

**What fails (v5.4.0 to v6.1.0):**
- `model.encode(df["text"])` raises `ValueError: Unsupported input type: Series. Expected one of: str, dict, PIL.Image.Image, np.ndarray, torch.Tensor` (and `StringArray`, or `ArrowStringArray` when pyarrow is installed, for `series.values` with pandas 3).
- Lists containing `None`, `NaN` or numbers raise the same error with `int` or `float`.
- numpy string arrays failed only in v5.4.0 and work again from v5.4.1. A two-element array was taken as a text pair (`` ValueError: text input must be of type `str` ... `` from transformers); longer arrays raised `ValueError: Modality 'audio' is not supported by this SentenceTransformer model. Supported modalities: text`.

Before (v5.1.2):
```python
emb = model.encode(df["text"])            # pandas Series, may contain NaN or numbers
```

After (v5.4.0):
```python
emb = model.encode(df["text"].fillna("").astype(str).tolist())
```

### Inputs are no longer stripped of surrounding whitespace (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** silent embedding change for some tokenizers

v5.1.2 called `str(s).strip()` on every text before tokenizing. v5.4.0 passes text unchanged. WordPiece tokenizers (BERT, MiniLM, MPNet) drop leading and trailing whitespace anyway, and so does the XLM-RoBERTa tokenizer used by common multilingual models (paraphrase-multilingual-*, multilingual-e5-*, bge-m3), so their embeddings are identical. Byte-level BPE tokenizers (RoBERTa, GPT-2, Qwen) and Llama-style SentencePiece tokenizers now produce different tokens and embeddings for text with leading or trailing spaces or newlines. Check yours with `tok("  x\n").input_ids == tok("x").input_ids`. This also applies to models saved by v5.1.2.

The same applies to prompts. v5.1.2 measured the prompt length on the stripped prompt; v5.4.0 measures the prompt as given. With a byte-level BPE or Llama-style tokenizer, a prompt that ends in a space (`"query: "`) is now one token longer than the part of the text it covers, so `Pooling(include_prompt=False)` also excludes the first token of the input (on a tiny RoBERTa model the prompt length went from 5 to 6). To reproduce v5.1.2 there, move the space into the texts: `model.encode([" " + t for t in texts], prompt="query:")`.

Before (v5.1.2):
```python
emb = model.encode(["  some text\n"])          # stripped internally
```

After (v5.4.0), to reproduce v5.1.2:
```python
emb = model.encode([t.strip() for t in texts])
```

### Dict inputs: `{"query": ...}` fails on non-Router models; `{"text": ...}` broke in v5.4.0 and works again from v5.5.1

**First release:** v5.4.0 (break), v5.5.1 (`{"text": ...}` fixed) · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3554, https://github.com/huggingface/sentence-transformers/pull/3779 · **Impact:** hard error

v5.1.2 accepted any single-key dict and ignored the key on models without an `Asym`/`Router` module. From v5.4.0 dict keys are modality names (`text`, `image`, `audio`, `video`).

**What fails:**
- On a model without a `Router`, `model.encode([{"query": "..."}])` raises `ValueError: Multimodal dict input contains unrecognized modality keys: {'query'} ...` from v5.4.0 through v6.1.0 (printed as `['query']` from v5.7.0). On `Router` models the legacy `{"query": text}` form still routes (equivalent to `task="query"`).
- `model.encode([{"text": "..."}])` raised `ValueError: Modality 'text' is not supported by this SentenceTransformer model. ...` on v5.4.0 to v5.5.0 and works again from v5.5.1.

Before (v5.1.2):
```python
emb = model.encode([{"query": "hello world"}])   # key ignored on a plain model
```

After (v5.4.0):
```python
emb = model.encode(["hello world"])               # works on every version
emb = model.encode_query(["hello world"])         # if you meant a query prompt or route
```

### Embedding-dimension and truncation methods renamed (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** deprecation warning (`FutureWarning`, shown by default); still works in v6.1.0

| v5.1.2 | v5.4.0 |
|---|---|
| `model.get_sentence_embedding_dimension()` | `model.get_embedding_dimension()` |
| `module.get_word_embedding_dimension()` | `module.get_embedding_dimension()` |
| `model.truncate_sentence_embeddings(dim)` | `model.truncate_embeddings(dim)` |

The old names emit `` FutureWarning: The `get_sentence_embedding_dimension` method has been renamed to `get_embedding_dimension`. `` (and the equivalents). `get_embedding_dimension()` does not exist on v5.1.2 or v5.3.0, so code that must run on both needs a fallback.

Two subtle changes: a model made of only a `Transformer` module now reports the hidden size instead of `None`, and a custom subclass of a built-in module that overrides only `get_sentence_embedding_dimension` is bypassed, because the inherited `get_embedding_dimension` is found first.

Before (v5.1.2):
```python
dim = model.get_sentence_embedding_dimension()
with model.truncate_sentence_embeddings(256):
    emb = model.encode(texts)
```

After (v5.4.0):
```python
dim = model.get_embedding_dimension()
with model.truncate_embeddings(256):
    emb = model.encode(texts)
```

### `Pooling(include_prompt=False)` handles left padding correctly (v5.3.0)

**First release:** v5.3.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3598 · **Impact:** silent embedding change

With `include_prompt=False` and a prompt, v5.1.2 zeroed the first `prompt_length` positions of every row. For left-padded rows those positions are padding plus part of the prompt, so prompt tokens leaked into mean, max, mean-sqrt and weighted-mean pooling. v5.3.0 excludes padding and prompt per row. Only batches with at least one left-padded row change; right-padding models and `include_prompt=True` are unaffected.

**Notes vs. code:** listed in the v5.3.0 notes as a minor feature, not as an output change.

No code change. Re-encode stored corpora built with a left-padding model, `include_prompt=False` and prompts.

### CLS pooling uses the first unmasked token (v5.5.0)

**First release:** v5.5.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3767 · **Impact:** silent embedding change

`pooling_mode="cls"` used to take `token_embeddings[:, 0]`. It now takes the first position whose attention mask is 1. Two cases change:

1. Left-padded inputs: position 0 was a padding token for shorter rows; the first real token is now used (the intended fix). Right-padded BERT-style models are unaffected.
2. `Pooling("cls", include_prompt=False)` called with a prompt: the prompt positions, which include the leading `[CLS]`, are masked before pooling, so CLS pooling now returns the first token after the prompt instead of `[CLS]`.

**Notes vs. code:** the notes describe only the left-padding case.

Before (v5.1.2):
```python
model = SentenceTransformer(modules=[Transformer(name), Pooling(dim, "cls", include_prompt=False)])
emb = model.encode(texts, prompt="query: ")   # pooled [CLS]
```

After (v5.5.0), to keep the `[CLS]` vector with a prompt:
```python
model[1].include_prompt = True                 # CLS is position 0 either way
emb = model.encode(texts, prompt="query: ")   # identical to the v5.1.2 output
```

### Truncated chat-template inputs keep the template suffix (v5.6.0)

**First release:** v5.6.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3787 · **Impact:** silent score and embedding change for over-long chat-template inputs

When inputs are rendered through a chat template and exceed `max_length`, truncation used to cut off the template's closing tokens (such as `<|im_end|>` or an assistant prefill), so last-token pooling and causal-LM rerankers read a token in the middle of the document. v5.6.0 writes the template's suffix back onto the end of truncated rows. Inputs shorter than `max_length` are unchanged.

This only concerns inputs that go through a chat template, which v5.1.2 never applied: raw chat-model checkpoints loaded with v5.4+ and `Transformer` modules constructed directly from a chat model (see [raw causal-LM checkpoints](#raw-causal-lm-checkpoints-get-last-token-pooling-and-a-chat-template-v540)), and v5.4+ models saved with a `"message"` modality, such as causal-LM rerankers. Models saved before v5.4 load text-only and never take this path.

To reproduce the v5.4 to v5.5 truncation:
```python
emb = model.encode(texts, processing_kwargs={"chat_template": {"restore_suffix": False}})
```

### Flash Attention 2 inputs are flattened, and RoBERTa-family position IDs were wrong until v5.6.1 (v5.4.0)

**First release:** v5.4.0 (flattening), v5.6.1 (position fix) · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3554, https://github.com/huggingface/sentence-transformers/pull/3879 · **Impact:** silent numeric change; hard error for some `output_value` settings

From v5.4.0, text-only batches are concatenated into a single row without padding when all of these hold: transformers 5, a flash-attention `attn_implementation` with a variable-length kernel (`kernels` or `flash-attn`), the torch backend, and a feature-extraction `Transformer`. A v5.1.2 setup cannot hit this without also moving to transformers 5.

**What changes:**
- Results are numerically close to, but not bit-identical with, padded flash attention.
- The flattened batch has no `attention_mask`, so `encode(output_value="token_embeddings")` raises `KeyError: 'attention_mask'` and `output_value=None` raises `IndexError` (reproduced by forcing the flattened path on CPU; not run on a GPU). Custom modules placed after `Transformer` that read `attention_mask` break the same way. Disable it with `model[0].unpad_inputs = False`.
- For the RoBERTa family (XLM-R models such as `bge-m3` and `multilingual-e5-*`, RoBERTa, CamemBERT), the position IDs of flattened inputs lacked the `padding_idx + 1` offset until v5.6.1, which degraded embeddings. The v5.6.1 notes report bge-m3 STSb Spearman 0.7239 before and 0.8485 after (not reproduced here; needs a GPU). Re-encode corpora built that way.

**Notes vs. code:** #3879 appears in the v5.7.0 "All Changes" list, but it first shipped in v5.6.1, and the v5.7.0 code is identical. The v5.6.1 notes say the bug affected v5.5.0 to v5.6.0; the flattening code without the offset is identical in v5.4.0 and v5.4.1, so by the code v5.4.x is affected too (the quality effect was not measured here; it needs a GPU).

Before (v5.1.2):
```python
model = SentenceTransformer(name, model_kwargs={"attn_implementation": "flash_attention_2"})
tok = model.encode(texts, output_value="token_embeddings")
```

After (v5.4.0, transformers 5):
```python
model = SentenceTransformer(name, model_kwargs={"attn_implementation": "flash_attention_2"})
model[0].unpad_inputs = False    # needed for token_embeddings, output_value=None or custom modules
tok = model.encode(texts, output_value="token_embeddings")
```

### Per-call `processing_kwargs` (v5.5.0)

**First release:** v5.5.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3753 · **Impact:** additive

`encode`, `encode_query`, `encode_document`, `predict` and `preprocess` accept `processing_kwargs` through `**kwargs` (the signatures are unchanged), merged per top-level key over the model's own processing settings for that call only. On v5.4.x, `encode`, `encode_query` and `encode_document` raised `ValueError: SentenceTransformer.encode() has been called with additional keyword arguments that this model does not use: ['processing_kwargs'] ...`, while `CrossEncoder.predict` and `preprocess` accepted the keyword and silently ignored it.

```python
emb = model.encode(texts, processing_kwargs={"text": {"max_length": 256, "truncation": True}})
```

### `model.compile()` takes effect, and forward hooks fire during inference (v5.7.0)

**First release:** v5.7.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3848 · **Impact:** silent change; can become a hard error at the first `encode()`

`SentenceTransformer.encode` and `SparseEncoder.encode` used to call `self.forward(...)` directly, and `CrossEncoder.predict` called `self.forward(...)` (v5.4.0 to v5.6.1) or the inner Hugging Face model (v5.1.2). They now go through the top-level model's `nn.Module.__call__`, so:

1. A `model.compile()` call (torch's `nn.Module.compile`) that silently did nothing for inference now compiles on the first `encode()`. That adds a compile delay and recompiles for new shapes unless `dynamic=True`, and fails where torch.compile cannot build kernels. On Windows CPU without MSVC, `encode()` raised `InductorError: InvalidCxxCompiler: Compiler: cl is not found.`
2. Hooks registered on the top-level model with `register_forward_hook`/`register_forward_pre_hook` now fire on every inference batch.

Eager outputs are unchanged.

**Notes vs. code:** the notes present this as a speed-up only.

Before (v5.1.2):
```python
model.compile()            # silently a no-op for encode()
emb = model.encode(texts)
```

After (v5.7.0):
```python
model.compile(dynamic=True)   # really compiles now; drop the call if the toolchain is missing
emb = model.encode(texts)
```

### `similarity` and `similarity_pairwise` are methods (v6.0.0)

**First release:** v6.0.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3794 · **Impact:** silent change

`model.similarity` and `model.similarity_pairwise` (also on `SparseEncoder`) were properties returning the resolved function, such as `cos_sim`. They are now methods that dispatch on `similarity_fn_name` when called. `model.similarity(a, b)` works as before. Differences:

- `model.similarity.__name__` is `"similarity"` instead of `"cos_sim"`, `"dot_score"` and so on. Use `model.similarity_fn_name`, or `util.similarity_fct_name(model.similarity)` for the function name.
- `fn = model.similarity` is now late-bound: after `model.similarity_fn_name = "dot"`, `fn` computes dot products (v5 kept the cosine function it captured).
- `model.similarity = my_fn` used to raise `AttributeError: property 'similarity' of 'SentenceTransformer' object has no setter`. It now silently creates an instance attribute that shadows the method, while `similarity_fn_name`, `similarity_pairwise` and the saved config keep the old function.
- `SimilarityFunction` gained `"maxsim"` and `"meanmaxsim"`, so `SimilarityFunction.possible_values()` returns six names. `SentenceTransformer` and `SparseEncoder` reject them: `ValueError: SentenceTransformer only supports similarity_fn_name in ('cosine', 'dot', 'euclidean', 'manhattan'), got 'maxsim'. Use MultiVectorEncoder for MaxSim late-interaction scoring.` Applied to ordinary 2-D embeddings, the MaxSim function silently returns a 1×1 matrix. Invalid names already raised `ValueError` in v5.1.2; only the message changed.

Before (v5.1.2):
```python
name = model.similarity.__name__                     # "cos_sim"
for fn_name in SimilarityFunction.possible_values():  # four dense functions
    model.similarity_fn_name = fn_name
```

After (v6.0.0):
```python
name = model.similarity_fn_name                      # "cosine"
for fn_name in model.SUPPORTED_SIMILARITY_FN_NAMES:   # excludes "maxsim"/"meanmaxsim"
    model.similarity_fn_name = fn_name               # never assign model.similarity
```

### Scores are computed in float32 for half-precision embeddings (v6.0.0)

**First release:** v6.0.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3892 · **Impact:** silent change in dtype and values

`util.cos_sim`, `dot_score`, `manhattan_sim`, `euclidean_sim`, their pairwise variants, and everything built on them (`model.similarity*`, `util.semantic_search`, `paraphrase_mining`, the IR and triplet evaluators, losses that call `cos_sim`) now upcast float16, bfloat16 and fp8 inputs to float32 before scoring. float32 and float64 inputs are unchanged. `encode()` output dtypes are unchanged: a bf16 model still returns bf16 tensors with `convert_to_tensor=True`.

**What changes:** score tensors are float32 instead of the input's half dtype. Values differ in the low bits and far fewer scores tie (10,000 fp16 cosine scores had 489 distinct values before and 9,977 after), so tie-dependent top-k results can change. `manhattan_sim` on CPU half tensors no longer raises `NotImplementedError: "cdist" not implemented for 'Half'`. The score matrix takes twice the memory for half inputs.

Before (v5.1.2):
```python
model = SentenceTransformer("all-MiniLM-L6-v2", model_kwargs={"torch_dtype": torch.float16})
emb = model.encode(["a", "b"], convert_to_tensor=True)   # float16
scores = model.similarity(emb, emb)                        # float16
```

After (v6.0.0):
```python
emb = model.encode(["a", "b"], convert_to_tensor=True)   # still float16
scores = model.similarity(emb, emb)                        # float32
scores_half = scores.half()                                # only if you need the old dtype
```

### A bare list of chat messages is one conversation (v6.0.0)

**First release:** v6.0.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3794 · **Impact:** silent shape change

A list whose first element is a `{"role": ..., "content": ...}` dict is now treated as one conversation, so `encode([msg1, msg2])` returns a single 1-D embedding instead of a batch of two. This only matters for models with a chat template (the `"message"` modality, from v5.4.0). For a text-only model such input raises `ValueError` from v5.4.0 on (`Modality 'message' is not supported by this SentenceTransformer model. ...` on v5.4.0 to v5.5.1, `This SentenceTransformer model does not support chat-style 'message' inputs ...` from v5.6.0); in v5.1.2 it silently embedded the role strings, so no correct v5.1.2 program depends on it.

Before (v5.4 to v5.7):
```python
embs = model.encode([{"role": "user", "content": "q1"}, {"role": "user", "content": "q2"}])  # (2, dim)
```

After (v6.0.0):
```python
embs = model.encode([[{"role": "user", "content": "q1"}], [{"role": "user", "content": "q2"}]])  # (2, dim)
```

### Multi-process `encode` quantizes once after merging (v6.0.0)

**First release:** v6.0.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3794 · **Impact:** silent change in int8/uint8 values

With a process pool and `precision="int8"` or `"uint8"`, each worker used to quantize its own chunk with calibration ranges from that chunk, so the same text got different codes depending on chunking. Workers now return float32 and the parent quantizes the merged result once, matching single-process output. In a 64-text test, 7,695 int8 elements differed from single-process encoding on v5.1.2 and 2 on v6.0.0. Peak memory is higher. Binary precisions are unaffected.

No code change. Regenerate int8/uint8 indexes built with multi-process encoding, or quantize yourself with fixed calibration:

```python
emb = model.encode(texts, pool=pool)                                  # float32
q = quantize_embeddings(emb, "int8", calibration_embeddings=calib)    # independent of chunking and version
```

### `Pooling(include_prompt=False)` leaves `attention_mask` intact (v6.0.0)

**First release:** v6.0.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3944 · **Impact:** silent change in returned features

Up to v5.6.1, `Pooling` zeroed the prompt positions of `features["attention_mask"]` in place; v5.7.0 stored a zeroed copy under the same key instead. In both cases the mask returned by `encode(output_value=None)` had the prompt positions zeroed, later modules saw the zeroed mask, and a second forward pass over the same features gave different embeddings. Now the prompt is excluded inside pooling only. `sentence_embedding` and `token_embeddings` values are unchanged. This also fixes a v5.7.0-only training regression: with prompts and `include_prompt=False`, `AdaptiveLayerLoss`/`Matryoshka2dLoss` computed a different loss on v5.7.0 than on v5.1.2 and v6.0.0 (0.920 vs. 0.752 in a seeded test).

Before (v5.1.2):
```python
out = model.encode(texts, prompt_name="query", output_value=None)
mask = out[0]["attention_mask"]               # prompt positions already zeroed
```

After (v6.0.0):
```python
out = model.encode(texts, prompt_name="query", output_value=None, batch_size=4)   # see the output_value=None entry
mask = out[0]["attention_mask"].clone()
mask[: int(out[0]["prompt_length"])] = 0      # rebuild it (assumes right padding)
```

### Smaller inference changes (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** mostly additive

- `encode()` raises `ValueError: batch_size must be a positive integer, got 0.` for `batch_size <= 0`, and `SentenceTransformer(modules=[])` raises `ValueError` for an empty module list.
- An explicit empty prompt now disables a saved prompt: `prompts={"query": ""}` used to be overwritten by the saved prompt.
- Strings that look like media URLs or paths (`https://.../x.jpg`, an existing `*.png`, `data:image/...`) are detected as images, video or audio, but only on models that support that modality. Text-only models still treat them as text.
- Private methods used by some subclasses were renamed or removed: `_text_length`, `_get_prompt_length`, `_load_auto_model`, `_load_sbert_model`, `_encode_multi_process`, `_encode_multi_process_worker` and the `_model_config` attribute. Overrides of these hooks are no longer called.
- New public API: `modalities`, `supports()`, `preprocess()`, `processor`, `get_embedding_dimension()`, `truncate_embeddings()`.

## CrossEncoder

v5.4.0 rebuilt `CrossEncoder` on the same module chain as `SentenceTransformer`. Scores from standard rerankers are unchanged: the tiny test reranker returned identical `predict` output on v5.1.2, v5.4.0 and v6.1.0. Most of the risk is in positional arguments, attributes that used to be plain Hugging Face objects, and custom training code.

### CrossEncoder is now a module chain: what happened to its attributes (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** mostly silent

`CrossEncoder` used to subclass `nn.Module` and hold a Hugging Face model in `self.model` and a tokenizer in `self.tokenizer`. It is now an `nn.Sequential` of modules: a `Transformer(transformer_task="sequence-classification")`, or `Transformer` plus `LogitScore` for causal-LM rerankers.

**Still available (as properties, also in v6.1.0):** `model.model` (the same `...ForSequenceClassification` object), `model.tokenizer`, `model.config`, `model.num_labels`, `model.activation_fn`, `model.device`. New: `model.max_seq_length`, `model.processor`, `model.preprocess()`, `model.prompts`, `model[0]`.

**What changes silently:**
- `model.config.sentence_transformers` is no longer populated on load. v5.1.2 wrote `{"activation_fn": ...}` into the Hugging Face config at init, so reading `model.config.sentence_transformers["activation_fn"]` raises `AttributeError: 'BertConfig' object has no attribute 'sentence_transformers'` unless the checkpoint already had it.
- `repr(model)` and `named_children()` show `0` (the `Transformer` module) instead of `model` and `activation_fn`, and `isinstance(model, nn.Sequential)` is now `True`.
- `model.model = new_hf_model` no longer replaces the model: it appends `new_hf_model` to the chain as an extra module named `model`, `model.model` keeps returning the old one, and the next `predict` fails with an unrelated `TypeError` or `AttributeError`. Assign `model[0].model = new_hf_model` instead (and `model[0].processor = tok` for the tokenizer; `model.tokenizer = tok` raises `AttributeError`).

Before (v5.1.2):
```python
hf_model, tok = model.model, model.tokenizer
act_path = model.config.sentence_transformers["activation_fn"]
```

After (v5.4.0):
```python
from sentence_transformers.util import fullname

hf_model, tok = model.model, model.tokenizer     # still fine
act_path = fullname(model.activation_fn)
```

### `set_activation_fn()` and `set_config_value()` were removed (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** hard error

Both public methods were removed without a deprecation period; neither the release notes nor the migration guide mention it. `get_default_activation_fn()` still exists. The activation is now a plain attribute that `save()` writes to `config_sentence_transformers.json`.

**What fails:** `AttributeError: 'CrossEncoder' object has no attribute 'set_activation_fn'` (or `'set_config_value'`), also in v6.1.0.

Before (v5.1.2):
```python
model.set_activation_fn(torch.nn.Identity())
```

After (v5.4.0):
```python
model.activation_fn = torch.nn.Identity()
```

### `predict(activation_fn=...)` applies to that call only (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** silent score change

In v5.1.2, passing `activation_fn` (or the older `activation_fct`) to `predict()` or `rank()` permanently replaced `model.activation_fn`, so later calls without the argument used it too. From v5.4.0 it affects only that call. Code that relied on the side effect, for example one call with `activation_fn=nn.Identity()` followed by calls expecting logits, now gets sigmoid scores again. This is documented in the v5.4.0 migration guide but not in the release notes.

Before (v5.1.2):
```python
model.predict(pairs, activation_fn=torch.nn.Identity())   # also set model.activation_fn
logits = model.predict(other_pairs)                       # raw logits
```

After (v5.4.0):
```python
model.activation_fn = torch.nn.Identity()                 # set it explicitly to make it stick
logits = model.predict(other_pairs)
```

### `predict()` and `rank()` take `prompt_name` and `prompt` before `batch_size` (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** hard error for most positional calls

Two parameters were inserted: `predict(inputs, prompt_name, prompt, batch_size, show_progress_bar, activation_fn, ...)` and `rank(query, documents, top_k, return_documents, prompt_name, prompt, batch_size, ...)`. A positional `batch_size` lands in `prompt_name`. Neither the notes nor the migration guide mention this.

**What fails (v5.4.0 to v6.1.0):**
- `predict(pairs, 64)` and `rank(q, docs, 10, True, 64)` raise `ValueError: Prompt name '64' not found in the configured prompts dictionary with keys [].`
- `predict(pairs, 64, True)` raises `TypeError: unsupported operand type(s) for +: 'bool' and 'str'` (`'bool' and 'list'` when the pairs are lists).
- `predict(pairs, 64, False)` logs a warning, silently ignores the batch size and returns correct scores.

`query`, `documents`, `top_k` and `return_documents` are still safe to pass positionally.

Before (v5.1.2):
```python
scores = model.predict(pairs, 64, False)
hits = model.rank(query, docs, 10, True, 64)
```

After (v5.4.0; also valid on v5.1.2):
```python
scores = model.predict(pairs, batch_size=64, show_progress_bar=False)
hits = model.rank(query, docs, top_k=10, return_documents=True, batch_size=64)
```

### `predict(sentences=...)` becomes `predict(inputs=...)` (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** deprecation warning; still works in v6.1.0

The first parameter of `predict` is now `inputs` and accepts multimodal pairs. `sentences=` still works and logs `` The `sentences` argument was renamed and is now deprecated. Please use `inputs` instead. `` once. `activation_fct=` is still mapped to `activation_fn=`.

Before (v5.1.2):
```python
scores = model.predict(sentences=pairs)
```

After (v5.4.0):
```python
scores = model.predict(pairs)   # or inputs=pairs
```

### Constructor: positional arguments past the fifth are silently ignored (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** deprecation warning for positions 2 to 5; silent for later positions

Arguments after `model_name_or_path` are keyword-only. A shim remaps up to four legacy positional arguments (`num_labels`, `max_length`, `activation_fn`, `device`) and logs `` Passing `num_labels` as a positional argument to CrossEncoder is deprecated. Please use `num_labels=1` as a keyword argument instead. `` for each. Any further positional arguments (`cache_folder`, `trust_remote_code`, `revision`, `local_files_only`, `token`, `model_kwargs`, `tokenizer_kwargs`, `config_kwargs`, `model_card_data`, `backend`) are **dropped without a warning**. A positional `revision` that doesn't exist loads the default revision instead of raising `OSError`.

**Notes vs. code:** the migration guide says positional arguments "still work but emit a deprecation warning", which is only true for the first four after the name. Still the case in v6.1.0.

Before (v5.1.2):
```python
model = CrossEncoder("my/model", 1, 512, None, "cuda", None, False, "v1.0")
```

After (v5.4.0; also valid on v5.1.2):
```python
model = CrossEncoder("my/model", num_labels=1, max_length=512, device="cuda", revision="v1.0")
```

### `tokenizer_kwargs` becomes `processor_kwargs`, and `max_length` takes precedence (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** deprecation warning; silent truncation change when both are given

The constructor now takes `processor_kwargs` (forwarded to `AutoProcessor.from_pretrained`). `tokenizer_kwargs` and the older `tokenizer_args` still work and log `` The CrossEncoder `tokenizer_kwargs` argument was renamed and is now deprecated. Please use `processor_kwargs` instead. `` `processor_kwargs` is not accepted by v5.1.2.

If both `max_length=` and `tokenizer_kwargs={"model_max_length": ...}` are given, v5.1.2 kept the `tokenizer_kwargs` value; v5.4.0 uses `max_length` (64 vs. 128 in the test). Pass only one.

Before (v5.1.2):
```python
model = CrossEncoder("my/model", tokenizer_kwargs={"model_max_length": 256})
```

After (v5.4.0):
```python
model = CrossEncoder("my/model", max_length=256)   # or processor_kwargs={"model_max_length": 256}
```

### The `max_length` property is renamed `max_seq_length` (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** deprecation warning; still works in v6.1.0

Reading or setting `model.max_length` emits `` DeprecationWarning: The `max_length` property was renamed and is now deprecated. Please use `max_seq_length` instead. `` (hidden by default outside `__main__`) and behaves as before. The `max_length=` **constructor** argument is not deprecated, and `max_seq_length=` is not accepted by the constructor (`TypeError`). In v5.1.2, setting `model.max_seq_length` only created an unused attribute; it now truncates.

Before (v5.1.2):
```python
model.max_length = 256
```

After (v5.4.0):
```python
model.max_seq_length = 256
```

### `model(**features)` no longer runs the Hugging Face model (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** hard error

`CrossEncoder.forward(*args, **kwargs)` used to be `self.model(*args, **kwargs)` and returned a `SequenceClassifierOutput`; custom losses and loops did `model(**tokens)[0]`. It now takes one features dict and returns it with a `"scores"` entry of shape `[batch, num_labels]`.

**What fails:** `TypeError: BaseModel.forward() missing 1 required positional argument: 'input'`. No shim; also in v6.1.0. `model.model(**tokens)` still returns the raw Hugging Face output.

Before (v5.1.2):
```python
tokens = model.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(model.device)
logits = model(**tokens)[0].view(-1)
```

After (v5.4.0):
```python
features = model.preprocess(pairs).to(model.device)
logits = model(features)["scores"].view(-1)
```

### Rerankers whose tokenizer has a chat template receive chat-formatted input (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** silent score change or constant scores; hard error with many templates (including Qwen's from v5.7.0)

v5.1.2 always tokenized pairs as plain text pairs. From v5.4.0, if the tokenizer has a `chat_template`, pairs are converted to a `query` message and a `document` message and rendered with `apply_chat_template`. BERT, RoBERTa, DeBERTa and ModernBERT rerankers have no chat template and give identical scores. Decoder-based `*ForSequenceClassification` rerankers (Qwen, Llama, Gemma, Mistral) whose tokenizer ships a template now receive a different input string. Three outcomes, depending on the template:

1. Templates that pass any role through (generic ChatML, for example SmolLM2) change the scores silently (0.5072 vs. 0.4777 in a test).
2. Templates that only render system/user/assistant roles, including Qwen2.5 and Qwen3, drop both the query and the document: on v5.4.0 to v5.6.x every pair gets the same meaningless score, and from v5.7.0 `predict` raises `ValueError: The chat template of <model> cannot carry a 'query'/'document' pair (the 'query' and 'document' content does not reach the rendered prompt) ...`.
3. Templates that enforce user/assistant alternation (Llama-2, Mistral, Gemma) raise `jinja2.exceptions.TemplateError: Conversation roles must alternate user/assistant/...` on v5.4.0 to v5.6.x and the same `ValueError` from v5.7.0.

**Notes vs. code:** the notes present chat-template inputs only as a new multimodal feature.

Before (v5.1.2):
```python
model = CrossEncoder("org/qwen-based-seq-cls-reranker")
scores = model.predict(pairs)
```

After (v5.4.0), to keep plain text-pair tokenization (reproduced the v5.1.2 scores exactly):
```python
from sentence_transformers.cross_encoder.modules import Transformer

t = Transformer("org/qwen-based-seq-cls-reranker", transformer_task="sequence-classification",
                modality_config={"text": {"method": "forward", "method_output_name": "logits"}},
                module_output_name="scores")
model = CrossEncoder(modules=[t])
scores = model.predict(pairs)
```

### A raw causal-LM checkpoint becomes a generative reranker, not a classifier (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** silent architecture change; can raise

In v5.1.2, `CrossEncoder("Qwen/Qwen2.5-0.5B", num_labels=1)` (any checkpoint whose architecture ends in `ForCausalLM`) loaded `AutoModelForSequenceClassification` with a new, randomly initialized one-label head, the usual starting point for training a decoder-based reranker. From v5.4.0 the same call builds `Transformer(transformer_task="text-generation")` plus `LogitScore`, which scores `logit("yes") - logit("no")` at the last position, uses left padding and the chat template, and has no classification head. If the tokenizer has no unknown token and lacks `yes`/`no`, it raises `ValueError: The tokenizer does not have 'yes' and/or 'no' tokens ...`; with an unknown token, missing `yes`/`no` silently map to its id. Because the chat template is applied, a raw Qwen2.5 or Qwen3 checkpoint scores an empty prompt on v5.4.0 to v5.6.x (every pair gets the same score) and raises the chat-template `ValueError` from v5.7.0 (see the previous entry). The default activation is `Sigmoid` when `num_labels=1` is passed, as above. Without `num_labels` it is `Identity` on v5.4.x (the code reads `config.num_labels`, which is 2 for a causal-LM config) and `Sigmoid` from v5.5.0 (it reads the model's `num_labels`, which is 1 when the last module is `LogitScore`). Checkpoints with `modules.json` or a `*ForSequenceClassification` architecture are not affected.

Before (v5.1.2):
```python
model = CrossEncoder("Qwen/Qwen2.5-0.5B", num_labels=1)   # new classification head, then CrossEncoderTrainer
```

After (v5.4.0), to build the v5.1.2 architecture explicitly:
```python
from sentence_transformers.cross_encoder.modules import Transformer

t = Transformer("Qwen/Qwen2.5-0.5B", transformer_task="sequence-classification", config_kwargs={"num_labels": 1},
                modality_config={"text": {"method": "forward", "method_output_name": "logits"}},
                module_output_name="scores")
model = CrossEncoder(modules=[t])
```

### `save()` no longer forwards Hugging Face keyword arguments, and `push_to_hub(tags=...)` was removed (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** hard error

`CrossEncoder.save(path, *, safe_serialization=True, **kwargs)` forwarded extra arguments to `PreTrainedModel.save_pretrained`. It now shares `SentenceTransformer.save(path, model_name=None, create_model_card=True, train_datasets=None, safe_serialization=True)`. `push_to_hub` lost `tags` and gained `local_model_path`, `replace_model_card` and `train_datasets`.

**What fails:** `TypeError: BaseModel.save() got an unexpected keyword argument 'max_shard_size'` and `TypeError: BaseModel.push_to_hub() got an unexpected keyword argument 'tags'`.

**Silent difference:** with the default `replace_model_card=False`, `push_to_hub` does not overwrite an existing README.md in the target repository; v5.1.2 always uploaded the generated card.

`SentenceTransformer.push_to_hub` never had a `tags` parameter, so the migration guide's "Removed `tags` parameter from `push_to_hub`" only concerns CrossEncoder.

Before (v5.1.2):
```python
model.save("out", max_shard_size="200MB")
model.push_to_hub("me/my-reranker", tags=["reranker"])
```

After (v5.4.0):
```python
model.save("out")                                   # or model.model.save_pretrained("out", max_shard_size="200MB")
model.model_card_data.add_tags(["reranker"])
model.push_to_hub("me/my-reranker", replace_model_card=True)
```

The saved folder format also changed; see [CrossEncoder save format](#crossencoder-save-format-modulesjson-and-config_sentence_transformersjson-activation-moved-out-of-configjson-v540).

### Multi-process `predict` and `rank` (v5.2.0)

**First release:** v5.2.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3580 · **Impact:** additive

`predict()` and `rank()` gained `device`, `pool` and `chunk_size` at the end of their signatures, and `CrossEncoder` gained `start_multi_process_pool()` and `stop_multi_process_pool()`. Passing a single `device="..."` moves the whole model to that device persistently, like `SentenceTransformer.encode(device=...)` (read from the code; not run on a GPU). `mine_hard_negatives(use_multi_process=...)` now also parallelizes the cross-encoder rescoring step.

```python
if __name__ == "__main__":
    pool = model.start_multi_process_pool(["cuda:0", "cuda:1"])
    scores = model.predict(pairs, batch_size=64, pool=pool)
    model.stop_multi_process_pool(pool)
```

### Chat-template rerankers must render both roles (v5.7.0)

**First release:** v5.7.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3902 · **Impact:** hard error (results were already wrong on v5.4.0 to v5.6.x)

When a model has a chat template, pairs become a `query` message and a `document` message. If the template ignores one of those roles, v5.7.0 raises `ValueError: The chat template of <model> cannot carry a 'query'/'document' pair ...` with a template you can copy, instead of crashing later or returning meaningless scores. This also affects v5.1.2 code: a decoder-based reranker whose tokenizer has such a template (for example Qwen2.5 or Qwen3, or Llama-2, Gemma and Mistral templates) worked on v5.1.2 and raises this `ValueError` from v5.7.0; see [rerankers with a chat template](#rerankers-whose-tokenizer-has-a-chat-template-receive-chat-formatted-input-v540) for the workaround. A related v5.7.0 change (#3869): causal rerankers now check for right padding in the actual attention mask, so single texts and equal-length batches no longer raise, while a `processing_kwargs` override that sets right padding now does.

### Logits are converted to float32 before the activation (v6.0.0)

**First release:** v6.0.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3893 · **Impact:** silent score and dtype change for half-precision rerankers

`predict` now casts float16, bfloat16 and fp8 logits to float32 before the sigmoid or softmax. In bf16 the sigmoid saturates near 1.0, ties the best candidates, and `rank` then ordered them by input position; in a saturation test, v5.7.0's top five all scored exactly 1.0 and came back in input order, while v6.0.0 separated them. `predict(..., convert_to_tensor=True)` and `convert_to_numpy=False` return float32 tensors instead of bf16/fp16; the default numpy output was already float32, but its values change. Every CrossEncoder evaluator inherits this.

Before (v5.1.2):
```python
model.model.to(torch.bfloat16)
scores = model.predict(pairs, convert_to_tensor=True)   # bfloat16, saturated sigmoid
```

After (v6.0.0):
```python
scores = model.predict(pairs, convert_to_tensor=True)   # float32, sigmoid computed in float32
```

### `rank()` returns Python floats (v6.0.0)

**First release:** v6.0.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3927 · **Impact:** silent type change; deprecation warning

Each `"score"` returned by `rank()` is now a Python `float`, so results serialize directly with `json.dumps` (v5 returned `numpy.float32`, which `json.dumps` rejects, or 0-d tensors with `convert_to_tensor=True`). `convert_to_numpy` and `convert_to_tensor` were removed from `rank`'s signature; passing them by keyword is accepted, ignored, and logs `` The CrossEncoder.rank `convert_to_tensor` argument is deprecated and has no effect, as `rank` always returns Python floats. It will be removed in a future version. `` once. `predict` keeps both parameters.

**What fails:** `score.item()`, `score.cpu()` and `torch.stack([r["score"] ...])` raise `AttributeError: 'float' object has no attribute ...` or `TypeError`. Positional arguments past `apply_softmax` now land in `device` and `pool`.

Before (v5.1.2):
```python
results = model.rank(query, docs, convert_to_tensor=True)
best = results[0]["score"].item()
```

After (v6.0.0):
```python
results = model.rank(query, docs)
best = results[0]["score"]                                        # already a float
scores = model.predict([(query, d) for d in docs], convert_to_tensor=True)  # for tensors
```

## SparseEncoder

`SparseEncoder` shares the v5.4.0 restructure with `SentenceTransformer`, so most items in [SentenceTransformer inference](#sentencetransformer-inference) also apply: whitespace is no longer stripped, `processor_kwargs={"model_max_length": N}` now overrides a saved `max_seq_length`, and models saved by v5.4.0 or later cannot be loaded by v5.3.x or earlier (`ModuleNotFoundError: No module named 'sentence_transformers.sparse_encoder.modules'` or `'sentence_transformers.base'`). Models saved by v5.1.2 load in v5.4.0 and v6.1.0 with identical outputs.

### `SparseEncoder` is no longer a `SentenceTransformer` subclass (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** hard error for removed methods; silent for `isinstance`

`SparseEncoder` now derives from the shared `BaseModel`, so `isinstance(sparse_model, SentenceTransformer)` is `False`. Methods it inherited or kept for compatibility were removed without a shim: `encode_multi_process` (deprecated since v5.0), `fit` and `old_fit`, `smart_batching_collate`, `get_sentence_features`, the static `SparseEncoder.load(path)`, and `truncate_sentence_embeddings` (which already raised `NotImplementedError`).

**What fails:** `AttributeError: 'SparseEncoder' object has no attribute 'encode_multi_process'` (and the same for the other names), also in v6.1.0.

Before (v5.1.2):
```python
pool = model.start_multi_process_pool(["cpu", "cpu"])
emb = model.encode_multi_process(texts, pool)
model.stop_multi_process_pool(pool)
```

After (v5.4.0):
```python
emb = model.encode(texts, device=["cpu", "cpu"])   # or pool=model.start_multi_process_pool([...])
```

Train sparse models with `SparseEncoderTrainer`; `SparseEncoder.fit` no longer exists.

### Constructor arguments after the model name are keyword-only, and `max_active_dims=0` raises (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** hard error

As for `SentenceTransformer` there is no shim for positional arguments: `SparseEncoder(None, [mlm, pooling])` and `SparseEncoder(name, None, "cuda")` raise `TypeError: SparseEncoder.__init__() takes from 1 to 2 positional arguments but N were given`. `tokenizer_kwargs=` is accepted with a logged warning in favor of `processor_kwargs=`. `max_active_dims=0` used to mean "no limit"; it now raises `ValueError: max_active_dims must be a positive integer, got 0.` `encode(..., max_active_dims=0)` also meant "no limit" on v5.1.2, silently returned all-zero vectors on v5.4.0 to v5.6.x, and raises the same `ValueError` from v5.7.0.

Before (v5.1.2):
```python
model = SparseEncoder(None, [mlm, pooling], "cuda")
```

After (v5.4.0; also valid on v5.1.2):
```python
model = SparseEncoder(modules=[mlm, pooling], device="cuda")
```

### `MLMTransformer` is deprecated in favor of `Transformer(transformer_task="fill-mask")` (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** deprecation warning; hard error for positional arguments

`MLMTransformer` is now a thin subclass of `Transformer` and logs `` The MLMTransformer module is deprecated. Please use the Transformer module with `transformer_task="fill-mask"` instead. `` when constructed directly (not when a saved model is loaded). Everything after the model name is keyword-only, `model_args`/`tokenizer_args`/`config_args` are renamed with warnings, and `max_seq_length` follows `tokenizer.model_max_length`.

**What fails:** `MLMTransformer(name, 128)` raises `TypeError: Transformer.__init__() takes 2 positional arguments but 3 positional arguments (and 1 keyword-only argument) were given`.

**Dimension trap:** v5.1.2's `MLMTransformer.get_sentence_embedding_dimension()` returned the vocabulary size (30522 for BERT). The renamed `get_embedding_dimension()` returns the hidden size (768 for BERT-base). The v5.1.2 recipe `SpladePooling("max", word_embedding_dimension=mlm.get_sentence_embedding_dimension())` therefore records the hidden size in the saved pooling config, and the model reports that dimension while `encode` still returns vocabulary-sized vectors. For a model loaded from a checkpoint whose pooling config already holds the vocabulary size, model-level `SparseEncoder.get_sentence_embedding_dimension()` still returns it (with a `FutureWarning`); for a model built with the recipe above it returns the hidden size too.

Before (v5.1.2):
```python
from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling

mlm = MLMTransformer("naver/splade-cocondenser-ensembledistil", 256)
pool = SpladePooling("max", word_embedding_dimension=mlm.get_sentence_embedding_dimension())
```

After (v5.4.0):
```python
from sentence_transformers.sparse_encoder.modules import SpladePooling, Transformer

mlm = Transformer("naver/splade-cocondenser-ensembledistil", transformer_task="fill-mask", max_seq_length=256)
pool = SpladePooling("max", embedding_dimension=mlm.model.config.vocab_size)  # omitting it reports the hidden size until the first encode()
```

### Sparse module renames: `SpladePooling(embedding_dimension=)`, `SparseAutoEncoder.prepare()` (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** deprecation warning; one hard error

- `SpladePooling(word_embedding_dimension=...)` is renamed `embedding_dimension` (logged warning; still works in v6.1.0). The attribute is now `.embedding_dimension`, with no alias for `.word_embedding_dimension` (`AttributeError`). The saved config key changes too; old configs still load.
- `SparseAutoEncoder.preprocess(x)` (the input-normalization helper of CSR models) is renamed `prepare(x)`. Calling `preprocess` raises `AttributeError: 'SparseAutoEncoder' object has no attribute 'preprocess'`.
- `SparseStaticEmbedding.tokenize(texts)` became `preprocess(inputs, prompt=None, **kwargs)`; `tokenize` still works with a logged warning.

Before (v5.1.2):
```python
pool = SpladePooling(pooling_strategy="max", word_embedding_dimension=30522)
x_norm, info = sae.preprocess(x)   # info == {"mu": ..., "std": ...} when normalize=True
```

After (v5.4.0):
```python
pool = SpladePooling(pooling_strategy="max", embedding_dimension=30522)
x_norm, info = sae.prepare(x)
```

### `encode` with numpy string arrays failed in v5.4.0 only (v5.4.0, fixed in v5.4.1)

**First release:** v5.4.0 (break), v5.4.1 (fix) · **PR:** https://github.com/huggingface/sentence-transformers/pull/3720 · **Impact:** hard error in v5.4.0 only

`SparseEncoder.encode(df["text"].to_numpy())` raised `ValueError: Modality 'audio' is not supported by this SparseEncoder model. Supported modalities: text` in v5.4.0; v5.4.1 treats 1-D string or object arrays as batches again. `CrossEncoder.predict(np.array(pairs))`, which never worked in v5.1.2, also works from v5.4.1.

**Notes vs. code:** the v5.4.1 notes describe the v5.4.0 behavior as "silently treated as a single input"; on text-only models `SentenceTransformer.encode`, `SparseEncoder.encode` and `CrossEncoder.predict` raised instead.

### `encode_document(max_active_dims=k)` on Router models returns the true top-k (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3695 · **Impact:** silent score change (bug fix)

In v5.1.2, `max_active_dims` was passed to both tokenization and the forward pass, and for Router-based models (inference-free SPLADE) `encode_document(texts, max_active_dims=k)` kept dimensions that were not the top-k of the full document embedding. From v5.4.0 the kept dimensions are the true top-k. `encode_query` was unaffected. Re-index documents encoded this way with v5.1.2 if consistency matters.

### Sparse losses and search changes (v5.7.0)

**First release:** v5.7.0 · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3868, https://github.com/huggingface/sentence-transformers/pull/3852, https://github.com/huggingface/sentence-transformers/pull/3909, https://github.com/huggingface/sentence-transformers/pull/3907, https://github.com/huggingface/sentence-transformers/pull/3906 · **Impact:** silent changes; one hard error

- `SparseCoSENTLoss`'s default `similarity_fct` changed from `util.cos_sim` to `util.pairwise_cos_sim`, which matches its docstring and `CoSENTLoss`. The old default optimized a different objective (base loss 4.16 vs. 0.75 in the test). Pass `similarity_fct=util.cos_sim` only to reproduce old runs.
- `util.select_max_active_dims` no longer modifies its input in place, and `max_active_dims <= 0` raises `ValueError: max_active_dims must be a positive integer, got 0.`, also through `SparseEncoder.encode`.
- `semantic_search_qdrant` treats a 1-D sparse query as one query; it used to run one search per vocabulary entry and return that many result lists. `semantic_search_seismic` keeps results in query order and returns `[]` for a query with no matches instead of raising `IndexError`. Neither was run here (qdrant and seismic not installed); verified from the code.
- `euclidean_sim`, `manhattan_sim`, their pairwise variants and `pairwise_angle_sim` accept a sparse and a dense input together (previously `NotImplementedError` or `RuntimeError`).

## MultiVectorEncoder

`MultiVectorEncoder` is new in v6.0.0: a fourth model type for ColBERT-style late-interaction retrieval. Nothing in v5.1.2 needs to change because of it. Before v6, these models were used through separate libraries such as PyLate.

### `MultiVectorEncoder` added (v6.0.0)

**First release:** v6.0.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3794 · **Impact:** additive

`from sentence_transformers import MultiVectorEncoder` works from v6.0.0 (`ImportError` on v5.x). The constructor mirrors `SentenceTransformer` with everything after the model name keyword-only. `encode_query` and `encode_document` return a **list** of `(num_tokens, dim)` tensors (or numpy arrays with `convert_to_numpy=True`), and `similarity` computes MaxSim (`similarity_fn_name="maxsim"` or `"meanmaxsim"`). Loading a plain SentenceTransformer checkpoint as a `MultiVectorEncoder` appends a randomly initialized token-level `Dense` and logs `Converting SentenceTransformer model <name> to MultiVectorEncoder.`; a plain Hugging Face checkpoint gets the same random projection with only INFO-level messages, so nothing is shown at the default log level. Either must be trained before use. The package also adds `MultiVectorEncoderTrainer`, `MultiVectorEncoderTrainingArguments`, losses, evaluators and `util.maxsim`.

```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder("lightonai/LateOn")
q = model.encode_query(["Which planet is known as the Red Planet?"])   # list of (num_tokens, dim) tensors
d = model.encode_document(["Mars is often referred to as the Red Planet."])
scores = model.similarity(q, d)                                        # MaxSim
```

### PyLate query / document prefixes are combined with saved prompts (v6.0.1)

**First release:** v6.0.1 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3968 · **Impact:** silent embedding change for affected checkpoints

PyLate checkpoints can carry both a `query_prefix`/`document_prefix` (`"[Q] "`/`"[D] "`) and saved `prompts`. v6.0.0 used the prefix only when the prompt was empty, so a saved prompt silently dropped the marker. v6.0.1 composes them (`"[Q] search_query: "`). Among the 38 text checkpoints listed in the v6.0.1 documentation, only `lightonai/ColBERT-Zero` has both; the unlisted `lightonai/ColBERT-Zero-supervised` and `-unsupervised` do too. The author reports NanoBEIR NDCG@10 of 0.6569 before and 0.6824 after the fix (not reproduced here).

Side effects:
- An explicit empty prompt (`prompts={"query": "", "document": ""}`) now turns the legacy marker off, for PyLate and Stanford ColBERT checkpoints alike. On v6.0.0 it was overridden by the marker.
- A model loaded and re-saved with v6.0.0 keeps the prefix-less prompt; v6.0.1 does not repair it.
- On 2026-08-31 those three Hub repositories added a `requirements` entry for `sentence-transformers` with specifier `>=6.0.1` (and a `reason`), so v6.0.0 raises `ImportError: The model 'lightonai/ColBERT-Zero' requires: - sentence-transformers>=6.0.1, but sentence-transformers==6.0.0 is installed. ...`. That is a Hub change, not a code change. Pinning `revision=` to an earlier commit loads on v6.0.0 with the old behavior.

After (v6.0.1):
```python
model = MultiVectorEncoder("lightonai/ColBERT-Zero")
print(model.prompts)   # {'query': '[Q] search_query: ', 'document': '[D] search_document: '}
# reproduce the v6.0.0 behavior, e.g. for a model fine-tuned with 6.0.0:
legacy = MultiVectorEncoder("lightonai/ColBERT-Zero", prompts={"query": "search_query: ", "document": "search_document: "})
```

## Modules and import paths

### Import paths moved; old paths still work with a `DeprecationWarning` (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** deprecation warning; still works in v6.1.0 (removal announced only for "a future version")

v5.4.0 split the package into `sentence_transformers.base` (shared code) plus one sub-package per model type, and renamed TitleCase module files to snake_case. An import hook maps about 130 old module paths to their new locations and emits `DeprecationWarning: Importing from '<old>' is deprecated and will be removed in a future version. Please use '<new>' instead.` Python shows a `DeprecationWarning` only when the import is in `__main__`, so imports inside your own modules or libraries stay silent. Loading saved models whose `modules.json` contains old paths does not warn. Class and function imports from the top level (`from sentence_transformers import SentenceTransformer, CrossEncoder, SparseEncoder, util, LoggingHandler, SimilarityFunction, ...`) and package-level imports such as `from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss` are unchanged and do not warn. Importing the old sub-packages from the top level does warn: `from sentence_transformers import models, losses, evaluation` (also `datasets`, `readers`). None of the new import locations work on v5.3.0 or earlier (`sentence_transformers.util.similarity` exists there but does not contain `SimilarityFunction`), so code that must run on both should keep the old paths.

| v5.1.2 path | v5.4.0 path |
|---|---|
| `sentence_transformers.models` | `sentence_transformers.sentence_transformer.modules` (`Transformer`, `Dense`, `Router` live in `sentence_transformers.base.modules`) |
| `sentence_transformers.models.Normalize` | `from sentence_transformers.sentence_transformer.modules import Normalize` works on v5.4.0 to v6.1.0; the module file moved again in v6.0.0, so `import sentence_transformers.sentence_transformer.modules.normalize` (the path the v5.4.0 warning recommends) warns from v6.0.0; use `sentence_transformers.base.modules` |
| `sentence_transformers.losses` | `sentence_transformers.sentence_transformer.losses` |
| `sentence_transformers.evaluation` | `sentence_transformers.sentence_transformer.evaluation` (`SentenceEvaluator` is an alias of `sentence_transformers.base.evaluation.BaseEvaluator`) |
| `sentence_transformers.SentenceTransformer` (module) | `sentence_transformers.sentence_transformer.model` |
| `sentence_transformers.training_args`, `.trainer`, `.data_collator`, `.model_card` | `sentence_transformers.sentence_transformer.<same>` |
| `sentence_transformers.datasets`, `sentence_transformers.readers` | `sentence_transformers.sentence_transformer.datasets`, `.readers` |
| `sentence_transformers.peft_mixin` | `sentence_transformers.base.peft_mixin` |
| `sentence_transformers.sampler` | `sentence_transformers.base.sampler` |
| `sentence_transformers.similarity_functions` | `sentence_transformers.util.similarity` |
| `sentence_transformers.quantization` | `sentence_transformers.util.quantization` (`quantize_embeddings` is also importable from `sentence_transformers`) |
| `sentence_transformers.LoggingHandler` | `sentence_transformers.util.logging` |
| `sentence_transformers.cross_encoder.CrossEncoder` (module) | `sentence_transformers.cross_encoder.model` |
| `sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss` (module) | `sentence_transformers.cross_encoder.losses.binary_cross_entropy` (same pattern for every loss) |
| `sentence_transformers.sparse_encoder.models` | `sentence_transformers.sparse_encoder.modules` |
| `sentence_transformers.sparse_encoder.SparseEncoder` (module) | `sentence_transformers.sparse_encoder.model` |
| `sentence_transformers.sparse_encoder.losses.SpladeLoss` (module), `...evaluation.SparseNanoBEIREvaluator` (module) | snake_case module names, e.g. `...losses.splade`, `...evaluation.sparse_nano_beir` (same pattern for every sparse loss and evaluator) |

Before (v5.1.2):
```python
from sentence_transformers.models import Transformer, Pooling, Dense, Normalize
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling
```

After (v5.4.0, warning-free; v6.1.0 too):
```python
from sentence_transformers.sentence_transformer.modules import Transformer, Pooling, Dense, Normalize
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.util.similarity import SimilarityFunction
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from sentence_transformers.sparse_encoder.modules import Transformer as SparseTransformer, SpladePooling
# MLMTransformer(name) becomes SparseTransformer(name, transformer_task="fill-mask"); see the MLMTransformer entry
```

### Some old import paths were not kept (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** hard error

Three v5.1.2 module paths have no alias, some helpers left `sentence_transformers.model_card`, and the old sub-packages are no longer attributes of the top-level package:

| v5.1.2 | Error on v5.4.0 to v6.1.0 | Use instead |
|---|---|---|
| `import sentence_transformers as st` then `st.models.Transformer`, `st.losses...`, `st.evaluation...` without importing the sub-package | `AttributeError: module 'sentence_transformers' has no attribute 'models'` | `from sentence_transformers.sentence_transformer import modules, losses` (or import the old sub-package explicitly) |
| `sentence_transformers.fit_mixin` | `ModuleNotFoundError: No module named 'sentence_transformers.fit_mixin'` | `sentence_transformers.sentence_transformer.fit_mixin` |
| `sentence_transformers.model_card_templates` | `ModuleNotFoundError` | `sentence_transformers.sentence_transformer.deprecated_model_card_templates` |
| `sentence_transformers.evaluation.MSEEvaluatorFromDataFrame` (module) | `ModuleNotFoundError` | `from sentence_transformers.sentence_transformer.evaluation import MSEEvaluatorFromDataFrame` |
| `generate_model_card`, `get_versions`, `format_log`, `get_losses`, `YAML_FIELDS`, `IGNORED_FIELDS` from `sentence_transformers.model_card` | `ImportError: cannot import name ...` | `sentence_transformers.base.model_card` |

**Notes vs. code:** the migration guide says "all old import paths continue to work".

Before (v5.1.2):
```python
from sentence_transformers.fit_mixin import FitMixin
from sentence_transformers.model_card import generate_model_card
```

After (v5.4.0):
```python
from sentence_transformers.sentence_transformer.fit_mixin import FitMixin
from sentence_transformers.base.model_card import generate_model_card
```

### `Transformer` module: keyword-only arguments and renamed parameters (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** hard error for positional arguments; logged deprecation for renamed ones

`Transformer(model_name_or_path, max_seq_length, model_args, tokenizer_args, config_args, cache_dir, ...)` became `Transformer(model_name_or_path, *, transformer_task="feature-extraction", model_kwargs=None, processor_kwargs=None, config_kwargs=None, ..., max_seq_length=None, ...)`. The tokenizer is loaded with `AutoProcessor` and stored in `.processor`; the Hugging Face model is stored in `.model` (see [`tokenizer` and `auto_model`](#modeltokenizer-can-no-longer-be-assigned-and-assigning-auto_model-is-silently-ignored-v540)).

**What fails:** `Transformer("bert-base-uncased", 256)` raises `TypeError: Transformer.__init__() takes 2 positional arguments but 3 were given`. `model_args`, `tokenizer_args`, `config_args` and `cache_dir` still work and log `` The Transformer `tokenizer_args` argument was renamed and is now deprecated, please use `processor_kwargs` instead. `` (and `` The Transformer `cache_dir` argument is deprecated. Please pass `cache_dir` via `model_kwargs`, `processor_kwargs`, and/or `config_kwargs` instead. ``). Still accepted in v6.1.0.

Before (v5.1.2):
```python
from sentence_transformers import models

word = models.Transformer("bert-base-uncased", 256, tokenizer_args={"use_fast": True}, cache_dir="/cache")
```

After (v5.4.0):
```python
from sentence_transformers.sentence_transformer.modules import Transformer

word = Transformer(
    "bert-base-uncased",
    max_seq_length=256,
    processor_kwargs={"use_fast": True, "cache_dir": "/cache"},
    model_kwargs={"cache_dir": "/cache"},
    config_kwargs={"cache_dir": "/cache"},
)
```

### `Pooling`, `LSTM`, `CNN` and `WeightedLayerPooling` parameters and attributes renamed (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** deprecation warning for constructor arguments; hard error for removed attributes; silent dimension change for single-flag constructors

`Pooling(word_embedding_dimension, pooling_mode_cls_token=..., pooling_mode_mean_tokens=..., ...)` became `Pooling(embedding_dimension, pooling_mode="mean" | tuple_of_modes, include_prompt=True)`. The old keyword arguments still work: `word_embedding_dimension=` logs a warning and the six `pooling_mode_*` booleans emit `` FutureWarning: The `pooling_mode_mean_tokens` argument(s) are deprecated. Please use `pooling_mode` instead. `` (listing the flags you passed). Multi-mode concatenation order is preserved. But only the flags you actually pass are used: in v5.1.2 `pooling_mode_mean_tokens` defaulted to `True`, so `Pooling(768, pooling_mode_cls_token=True)` was CLS plus mean (1536 dimensions); from v5.4.0 it is CLS only (768 dimensions), with no error. Pass `pooling_mode=("cls", "mean")` to keep the old output. Saved models are unaffected, because their configs list all six flags. `LSTM(word_embedding_dimension=)`, `WeightedLayerPooling(word_embedding_dimension=)`, `SpladePooling(word_embedding_dimension=)` and `CNN(in_word_embedding_dimension=)` are renamed to `embedding_dimension` / `in_embedding_dimension` the same way, and so is `SoftmaxLoss(sentence_embedding_dimension=)`.

**What fails:** the instance attributes are gone: `pooling.word_embedding_dimension`, `pooling.pooling_mode_mean_tokens` (and the other five), `pooling.get_pooling_mode_str()`, and the `word_embedding_dimension` / `in_word_embedding_dimension` attributes of `LSTM`, `WeightedLayerPooling`, `SpladePooling` and `CNN` raise `AttributeError: 'Pooling' object has no attribute ...`. Use `pooling.embedding_dimension` and `pooling.pooling_mode`. Also: `pooling_mode=None` (the old default) raises `TypeError: 'NoneType' object is not iterable`; mode names are no longer lower-cased (`"CLS"` raises `ValueError: Invalid pooling mode: 'CLS'. ...`); and the boolean flags only work as keywords (the third positional parameter is now `include_prompt`). Saved pooling configs use the new keys; old configs load.

**Notes vs. code:** the migration guide lists the constructor renames but not the removed attributes or `get_pooling_mode_str()`.

Before (v5.1.2):
```python
pooling = models.Pooling(word_embedding_dimension=768, pooling_mode_cls_token=True, pooling_mode_mean_tokens=False)
if pooling.pooling_mode_cls_token:
    print(pooling.get_pooling_mode_str())
```

After (v5.4.0):
```python
from sentence_transformers.sentence_transformer.modules import Pooling

pooling = Pooling(embedding_dimension=768, pooling_mode="cls")
if pooling.pooling_mode == "cls":
    print(pooling.pooling_mode)
```

### Custom input modules: override `preprocess`, not `tokenize` (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** silent for `Transformer` subclasses; deprecation warning otherwise

Models now call `self[0].preprocess(...)` instead of `self[0].tokenize(...)`. For a custom `InputModule` that only implements `tokenize`, the base class forwards to it and logs `` <Class> overrides `tokenize` instead of `preprocess`. `tokenize` is deprecated, please override `preprocess` instead. `` But `Transformer` implements its own `preprocess`, so a **subclass of `Transformer` that only overrides `tokenize` is never called**, with no warning. The migration guide does not mention this.

Before (v5.1.2):
```python
class MyTransformer(models.Transformer):
    def tokenize(self, texts, padding=True):
        return super().tokenize([t.lower() for t in texts], padding)
```

After (v5.4.0):
```python
from sentence_transformers.sentence_transformer.modules import Transformer

class MyTransformer(Transformer):
    def preprocess(self, inputs, prompt=None, **kwargs):
        return super().preprocess([t.lower() for t in inputs], prompt=prompt, **kwargs)
```

### `Transformer._load_t5_model` and `_load_mt5_model` were removed (v5.3.0)

**First release:** v5.3.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3644 · **Impact:** hard error for subclasses that call them; silent for subclasses that override them

T5 and mT5 encoders are now loaded inside `_load_model`, and loading no longer permanently changes `T5EncoderModel._keys_to_ignore_on_load_unexpected`. The same PR added T5Gemma encoder support. Custom subclasses that call these methods fail with `AttributeError: ... has no attribute '_load_t5_model'`; subclasses that only override them are silently bypassed (the override is never called). Move the logic into an override of `_load_model`.

After (v5.3.0; also v6.1.0, where `_load_model`'s signature differs again):
```python
class MyTransformer(models.Transformer):
    def _load_model(self, *args, **kwargs):
        return super()._load_model(*args, **kwargs)
```

### `CLIPModel` is deprecated, and `model_name=` no longer works (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** deprecation warning; hard error for `model_name=`

`CLIPModel` is now a thin `Transformer` subclass: `CLIPModel(model_name_or_path="openai/clip-vit-base-patch32", **kwargs)`. Constructing it logs `The CLIPModel module is deprecated. Please use the Transformer module instead.` (not when a saved model is loaded). Embeddings of models saved by v5.1.2 are identical. `SentenceTransformer("openai/clip-vit-base-patch32")` on a raw CLIP checkpoint now works (v5.1.2 raised `AttributeError: 'CLIPConfig' object has no attribute 'hidden_size'`).

**What fails:** `CLIPModel(model_name="...")` raises `TypeError: Transformer.__init__() got an unexpected keyword argument 'model_name'`. A positional name still works in v6.1.0. See also the [CLIP save trap](#models-saved-by-v540-or-later-cannot-be-loaded-by-v53x-or-earlier-v540).

Before (v5.1.2):
```python
model = SentenceTransformer(modules=[models.CLIPModel(model_name="openai/clip-vit-base-patch32")])
```

After (v5.4.0):
```python
from sentence_transformers.sentence_transformer.modules import Transformer

model = SentenceTransformer(modules=[Transformer("openai/clip-vit-base-patch32")])
```

### `Dense` only loads `torch.*` activation functions unless the model is trusted (v5.4.1)

**First release:** v5.4.1 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3714 · **Impact:** silent embedding change (logged)

A saved `Dense` module stores its activation as an import path. v5.1.2 imported any path. From v5.4.1, paths outside `torch.` (including legitimate ones such as `transformers.activations.GELUActivation`) are refused unless the model is loaded with `trust_remote_code=True`, and the module falls back to its default `Tanh`.

**What happens:** a WARNING `` Activation function path 'transformers.activations.GELUActivation' is not trusted, falling back to the default activation function (Tanh). Please load the model with `trust_remote_code=True` to allow loading custom activation functions via the configuration. `` and different embeddings. Models whose Dense layers use `torch.nn.*` activations (`Tanh`, `Identity`, `GELU`, ...) are unaffected.

Before (v5.1.2):
```python
model = SentenceTransformer("org/model-with-transformers-gelu-dense")
```

After (v5.4.1):
```python
model = SentenceTransformer("org/model-with-transformers-gelu-dense", trust_remote_code=True)  # repos you trust
```

### `Router` passes `task` to the routed module (v6.0.1)

**First release:** v6.0.1 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3967 · **Impact:** silent fix; hard error for non-conforming custom modules

`Router.preprocess` used the `task` to pick a route but did not pass it on, so per-task settings of the routed `Transformer` (`query_length`/`document_length`, query expansion, a task-aware chat template) had no effect behind a Router; in a test, a query capped at 4 tokens came out as 12. v6.0.1 passes `task=` (including `task=None` for plain `encode()`).

**What fails:** a custom input module behind a Router whose `tokenize`/`preprocess` does not accept `**kwargs` now raises `TypeError: <Module>.tokenize() got an unexpected keyword argument 'task'`, even for plain `encode()`. It worked on v5.1.2 and v6.0.0. The documented signature has always been `tokenize(self, texts, **kwargs)`.

Before (v5.1.2):
```python
class MyInput(InputModule):
    def tokenize(self, texts):           # no **kwargs
        ...
```

After (v6.0.1):
```python
from sentence_transformers.base.modules import InputModule

class MyInput(InputModule):
    def preprocess(self, inputs, prompt=None, task=None, **kwargs):
        ...
```

## Training

The trainers, `*TrainingArguments` and the deprecated `fit()` APIs of v5.1.2 still exist in v6.1.0: `model.fit(train_objectives=[(dataloader, loss)], ...)` and `CrossEncoder.fit(train_dataloader=...)` still train. The three existing `*TrainingArguments` classes have the same 118 fields and defaults in v5.7.0 and v6.0.0 (compared at runtime with transformers 5.18). Training with transformers v5 needs v5.2.3 or later; see [fresh installs of v5.2.x](#fresh-installs-of-v52x-get-transformers-v5-v520).

### Columns named `labels` or `scores` are treated as labels (v5.2.0)

**First release:** v5.2.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3506 · **Impact:** silent change; hard error for text columns with these names

The SentenceTransformer and SparseEncoder data collator's label columns grew from `["label", "score"]` to `["label", "labels", "score", "scores"]` (the CrossEncoder collator already had this list). The first matching column is removed from the inputs and passed to the loss as labels.

**What changes:** a dataset with a float `scores` column now trains (v5.1.2 failed with `TypeError: 'float' object is not subscriptable`). A dataset that used a text column literally named `labels` or `scores` as an input fails with `ValueError: too many dimensions 'str'`.

Before (v5.1.2):
```python
train_dataset = Dataset.from_dict({"anchor": anchors, "positive": positives, "labels": negatives_as_text})
```

After (v5.2.0):
```python
train_dataset = train_dataset.rename_column("labels", "negative")   # text columns must not be named label(s)/score(s)
```

### `warmup_ratio` is kept as a Sentence Transformers argument (v5.3.0)

**First release:** v5.3.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3645 · **Impact:** fixes a hard error with newer transformers; logged deprecation on transformers v5

transformers v5 deprecated `warmup_ratio` and removed it in 5.15. On Sentence Transformers 5.2.3 or earlier with transformers 5.15 or newer, `SentenceTransformerTrainingArguments(warmup_ratio=0.1)` raises `TypeError: ... got an unexpected keyword argument 'warmup_ratio'`. From v5.3.0 the `*TrainingArguments` classes declare `warmup_ratio` themselves: on transformers v5 it is converted to a fractional `warmup_steps` with a logged warning (`` The `warmup_ratio` argument is deprecated in Transformers v5+, and will also be removed from Sentence Transformers once support for Transformers v4 is dropped. ... ``); on transformers v4 a float `0 < warmup_steps < 1` is converted to a ratio. v6.x still accepts `warmup_ratio`, and because it requires transformers v5 the warning always appears. With transformers 5.15 or newer, if both `warmup_ratio` and a non-zero `warmup_steps` are given, `warmup_ratio` is ignored.

Before (v5.1.2):
```python
args = SentenceTransformerTrainingArguments(output_dir="out", warmup_ratio=0.1)
```

After (v5.3.0; preferred on transformers v5 and v6.x):
```python
args = SentenceTransformerTrainingArguments(output_dir="out", warmup_steps=0.1)   # a float below 1 is a ratio
```

### `GroupByLabelBatchSampler` was rewritten (v5.3.0)

**First release:** v5.3.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3668 · **Impact:** hard error for small batches or single-label data; silent change in batches and steps per epoch

`BatchSamplers.GROUP_BY_LABEL` used to concatenate whole label groups. It now interleaves two samples per label per round, reshuffling the label order every round, and stops when fewer than two labels have samples left.

**What changes:**
- `batch_size` must be even and at least 4: `ValueError: batch_size must be an even number >= 4, but got 2.` (2 used to work).
- At least two labels need two or more samples each: `ValueError: GroupByLabelBatchSampler requires at least 2 distinct labels with >= 2 samples each, but only 1 label(s) qualified.`
- Batches mix labels, and on imbalanced data most samples of the dominant label are skipped in each epoch. In a test with labels of size 20/6/4/1 and batch size 8, an epoch went from 4 batches covering 30 of 31 samples to 2 batches covering 16.
- `len()` of the sampler, and therefore steps per epoch, learning-rate schedule length and ratio-based `eval_steps`, shrink accordingly.

**Notes vs. code:** the notes describe this as a fix without mentioning the new errors or the shorter epochs.

Before (v5.1.2):
```python
args = SentenceTransformerTrainingArguments(output_dir="out", per_device_train_batch_size=2,
                                            batch_sampler=BatchSamplers.GROUP_BY_LABEL)
```

After (v5.3.0):
```python
# even batch size >= 4; at least 2 labels with >= 2 samples; expect fewer steps per epoch
args = SentenceTransformerTrainingArguments(output_dir="out", per_device_train_batch_size=4,
                                            batch_sampler=BatchSamplers.GROUP_BY_LABEL)
```

### `NO_DUPLICATES_HASHED` sampler needs `xxhash<4` before v6.0.0 (v5.3.0)

**First release:** v5.3.0 (option added), v6.0.0 (xxhash 4 fix) · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3611, https://github.com/huggingface/sentence-transformers/pull/3658, https://github.com/huggingface/sentence-transformers/pull/3928 · **Impact:** additive; hard error with xxhash 4 on v5.3.0–v5.7.0

The default `BatchSamplers.NO_DUPLICATES` was reimplemented in v5.3.0 and produces exactly the same batches as before for the same seed. The new opt-in `BatchSamplers.NO_DUPLICATES_HASHED` precomputes hashes with `xxhash` (installed with `datasets`) using multiprocessing, so on Windows and macOS the script needs an `if __name__ == "__main__":` guard. With xxhash 4.0 (released 2026-08-12) the hashed sampler fails on v5.3.0 to v5.7.0 with `TypeError: Strings must be encoded before hashing`; v6.0.0 fixes it. v6.0.0 also deduplicates images, audio, video and arrays by content.

### Trainer `tokenizer=` becomes `processing_class=` (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** logged deprecation; still works in v6.1.0

`SentenceTransformerTrainer`, `CrossEncoderTrainer` and `SparseEncoderTrainer` renamed `tokenizer` to `processing_class`, matching the Hugging Face `Trainer`. The old name logs `` The `tokenizer` argument was renamed and is now deprecated. Please use `processing_class` instead. `` When neither is given, the trainer uses `model.processor`. A new `optimizer_cls_and_kwargs` parameter sits before `preprocess_logits_for_metrics`, so passing the latter positionally now binds to the wrong parameter.

Before (v5.1.2):
```python
trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss, tokenizer=model.tokenizer)
```

After (v5.4.0):
```python
trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss)   # defaults to model.processor
```

### CrossEncoder and SparseEncoder trainers no longer subclass the SentenceTransformer ones (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** silent change for `isinstance` checks and overridden hooks

`CrossEncoderTrainer`, `SparseEncoderTrainer` and their `*TrainingArguments` used to subclass `SentenceTransformerTrainer` and `SentenceTransformerTrainingArguments`. They now share `sentence_transformers.base.trainer.BaseTrainer` and `BaseTrainingArguments`, so `issubclass(CrossEncoderTrainer, SentenceTransformerTrainer)` is `False`, and code that patches or `isinstance`-checks `SentenceTransformerTrainer` no longer reaches the CrossEncoder and SparseEncoder trainers. Separately, the `_include_prompt_length()` hook was removed from all trainers in v5.4.0, so overrides of it are silently never called. Evaluators are typed as `BaseEvaluator` (formerly `SentenceEvaluator`).

Before (v5.1.2):
```python
if isinstance(trainer, SentenceTransformerTrainer): ...   # True for CrossEncoderTrainer
```

After (v5.4.0):
```python
from sentence_transformers.base.trainer import BaseTrainer

if isinstance(trainer, BaseTrainer): ...
```

### Data collators take `preprocess_fn` instead of `tokenize_fn` (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** hard error for code that builds a collator itself

The data collators now share a base class with a required `preprocess_fn` (normally `model.preprocess`). `tokenize_fn`, `include_prompt_lengths` and `all_special_ids` (fields of the SentenceTransformer collator, inherited by the SparseEncoder one) were removed without an alias. The CrossEncoder collator now rejects nested per-dataset, per-column `prompts` and `router_mapping` dictionaries with `ValueError`. The trainers build their collator automatically, so only hand-built collators are affected.

**What fails:** `TypeError: SentenceTransformerDataCollator.__init__() got an unexpected keyword argument 'tokenize_fn'` (same for `CrossEncoderDataCollator` and `SparseEncoderDataCollator`).

Before (v5.1.2):
```python
from functools import partial
from sentence_transformers.cross_encoder.data_collator import CrossEncoderDataCollator

collator = CrossEncoderDataCollator(tokenize_fn=partial(model.tokenizer, padding=True, truncation=True, return_tensors="pt"))
```

After (v5.4.0):
```python
from sentence_transformers.cross_encoder import CrossEncoderDataCollator

collator = CrossEncoderDataCollator(preprocess_fn=model.preprocess)
```

### CrossEncoder training `prompts` are applied (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** silent change in training inputs

In v5.1.2, `CrossEncoderTrainingArguments(prompts=...)` was accepted but had no effect. From v5.4.0, a string prompt (or a dataset-name-to-string dictionary) is prepended during training, so the same arguments produce a different model (training loss 0.7024 vs. 0.7071 in a two-step test). A per-column dictionary such as `{"query": "query: "}` is read as per-dataset and ignored for a plain `Dataset`; a nested dictionary raises `ValueError: CrossEncoder prompts cannot be per-column. ...`.

Before (v5.1.2):
```python
args = CrossEncoderTrainingArguments(output_dir="out", prompts="query: ")   # silently ignored
```

After (v5.4.0):
```python
args = CrossEncoderTrainingArguments(output_dir="out")   # drop it to reproduce v5.1.2 training
# or keep it deliberately and use the same prompt at inference: model.predict(pairs, prompt="query: ")
```

### `old_fit(use_amp=True)` autocasts on CPU and MPS (v5.6.0)

**First release:** v5.6.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3818 · **Impact:** silent numeric change

The legacy `SentenceTransformer.old_fit` loop used CUDA-only autocast, which did nothing for a model on CPU and crashed on MPS. It now autocasts on the model's device type, so `old_fit(..., use_amp=True)` on CPU trains with bfloat16 linear layers instead of float32. CUDA is unchanged, and `fit()` (which runs on the Trainer) is not affected.

**Notes vs. code:** the notes describe an MPS fix to `model.fit(..., use_amp=True)`; the changed code is `old_fit`, and the CPU effect is not mentioned.

Pass `use_amp=False` to keep float32 on CPU, or move to `SentenceTransformerTrainer`.

### Trainer subclasses receive dataset names in `get_eval_dataloader` (v5.7.0)

**First release:** v5.7.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3895 · **Impact:** silent change for custom trainer subclasses

With a dictionary `eval_dataset`, the Hugging Face `Trainer` now recurses with the dataset **names**, so an overridden `get_eval_dataloader` receives `"a"`, `"b"`, ... instead of `Dataset` objects. Metric keys (`eval_<name>_loss`) are unchanged. With `dataloader_persistent_workers=True`, prepared evaluation dataloaders are cached per name.

Before (v5.1.2):
```python
class MyTrainer(SentenceTransformerTrainer):
    def get_eval_dataloader(self, eval_dataset=None):
        print(len(eval_dataset))                  # always a Dataset
        return super().get_eval_dataloader(eval_dataset)
```

After (v5.7.0):
```python
class MyTrainer(SentenceTransformerTrainer):
    def get_eval_dataloader(self, eval_dataset=None):
        ds = self.eval_dataset[eval_dataset] if isinstance(eval_dataset, str) else (eval_dataset or self.eval_dataset)
        print(len(ds))
        return super().get_eval_dataloader(eval_dataset)
```

### Trainer checkpoint reload of custom modules (v6.0.0)

**First release:** v6.0.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3935 · **Impact:** hard error for `resume_from_checkpoint`; silent for `load_best_model_at_end`

Checkpoint reloading now goes through the `trust_remote_code` check (see [custom module classes](#custom-module-classes-require-trust_remote_codetrue-including-local-directories-v560-warning-v600-error)). Only custom modules that subclass `sentence_transformers.base.modules.Module` are pre-registered and exempt. A plain `torch.nn.Module` custom module, the usual v5.1.2 style, fails unless the model was built with `trust_remote_code=True`.

**What fails:**
- `trainer.train(resume_from_checkpoint=...)` raises the `ValueError` `The model <checkpoint> references the module class '<module.Class>', which is not part of Sentence Transformers. ...`.
- With `load_best_model_at_end=True`, the same error is only logged (`ERROR ... Could not load the best model from <checkpoint>. Error: ...`), `train()` returns normally, and the model keeps the **last** weights instead of the best ones.

**Notes vs. code:** the v6.0.0 notes say checkpoint reloading "keeps working for programmatically built models without the flag". That holds only for `Module` subclasses.

Before (v5.1.2):
```python
class AddOne(torch.nn.Module):          # plain nn.Module custom module
    def forward(self, features):
        features["sentence_embedding"] = features["sentence_embedding"] + 1.0
        return features
    def save(self, output_path): pass
    @staticmethod
    def load(input_path): return AddOne()

model = SentenceTransformer(modules=[transformer, pooling, AddOne()])
args = SentenceTransformerTrainingArguments(output_dir="out", eval_strategy="steps", save_strategy="steps",
                                            load_best_model_at_end=True)
```

After (v6.0.0):
```python
# Option A: opt in explicitly
model = SentenceTransformer(modules=[transformer, pooling, AddOne()], trust_remote_code=True)

# Option B: subclass Sentence Transformers' Module base class
from sentence_transformers.base.modules import Module

class AddOne(Module):
    config_keys = []
    def forward(self, features, **kwargs):
        features["sentence_embedding"] = features["sentence_embedding"] + 1.0
        return features
    def save(self, output_path, *args, **kwargs):
        self.save_config(output_path)
```

### Losses embed candidate columns in one merged forward pass (v6.0.0)

**First release:** v6.0.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3938 · **Impact:** silent change; no new arguments or defaults

`MultipleNegativesRankingLoss`, `TripletLoss`, `MarginMSELoss`, `DistillKLDivLoss`, `CoSENTLoss`, `CosineSimilarityLoss`, `ContrastiveLoss`, `OnlineContrastiveLoss`, `SoftmaxLoss`, `MegaBatchMarginLoss`, `GISTEmbedLoss`, `GlobalOrthogonalRegularizationLoss`, `EmbedDistillLoss`/`MSELoss` and `SpladeLoss` keep a separate forward pass for the anchor column but now concatenate columns 2 to N into one forward pass when their features are compatible. Merging only happens with at least three input columns (anchor plus two or more candidates); two-column (pair) training, which is all that `CoSENTLoss`, `CosineSimilarityLoss`, `ContrastiveLoss` and `OnlineContrastiveLoss` accept, is unchanged. Exception: `SpladeLoss(use_document_regularizer_only=True)` merges all columns, including the anchor.

**What changes (three or more columns):** loss values in eval mode are identical. In training, dropout masks are drawn for one large batch instead of per column, so runs are not bit-reproducible against v5 with the same seed. The per-column feature dicts passed to the loss are no longer filled in place with `sentence_embedding` for merged columns, so custom wrapper losses or callbacks that read `features[i]["sentence_embedding"]` afterwards get `KeyError`. Peak memory can rise when candidate columns differ greatly in length. Cached (GradCache) losses are unchanged.

Before (v5.1.2):
```python
loss_value = loss(features, labels)
neg_emb = features[2]["sentence_embedding"]   # filled by the per-column forward
```

After (v6.0.0):
```python
from sentence_transformers.base.losses import column_merging_disabled

with column_merging_disabled():               # restore per-column forwards where you rely on them
    loss_value = loss(features, labels)
neg_emb = features[2]["sentence_embedding"]
```

### New training warnings (v6.0.0)

**First release:** v6.0.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3794 · **Impact:** log output only

- Creating `*TrainingArguments` with `dataloader_num_workers > 0` and `dataloader_persistent_workers=False` on Windows or macOS logs a recommendation to enable persistent workers.
- A column named `id` or ending in `_id`, `_ids` or `_idx` logs that it will be tokenized and trained on as text (v5 did the same silently).
- When a loss has to embed columns separately, it logs once why it could not merge them.

## Losses

Loss class names and constructor arguments are mostly stable; what moved are internals that custom subclasses rely on, several silent objective changes, and new input validation.

### `AdaptiveLayerLoss` and `Matryoshka2dLoss` train different layers with different weights (v5.2.0, v5.7.0)

**First release:** v5.2.0 (layer selection), v5.7.0 (`layer_weighting` default, detached KL teacher) · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3560, https://github.com/huggingface/sentence-transformers/pull/3901, https://github.com/huggingface/sentence-transformers/pull/3880 · **Impact:** silent change in the training objective

Three changes, none of which raises:

1. **v5.2.0, unflagged:** for a model with N transformer layers, v5.1.2 trained the "prior layer" losses on transformer layers 1 to N-1 with weights 1/1 to 1/(N-1). v5.2.0 includes the embedding-layer output (index 0) as a layer with weight 1 and shifts every other weight (layer k now gets 1/(k+1)). On a 5-layer model the layers used went from `[5, 1, 2, 3, 4]` to `[5, 0, 1, 2, 3, 4]`. The v5.2.0 notes only mention a fix to when layer embeddings are restored.
2. **v5.7.0:** the new `layer_weighting` argument defaults to `"uniform"` (every prior layer weighted 1.0). The other options are `"log"`, `"linear"` (the v5.2.0 to v5.6.1 `1/(1+i)` decay) and any callable mapping a layer index to a weight. `get_config_dict()`, and therefore generated model cards, gain a `layer_weighting` key.
3. **v5.7.0:** the KL-divergence term now treats the final layer as a fixed teacher (`.detach()`), so it no longer back-propagates into the final layer. The loss value is unchanged; the gradients change, and there is no switch to restore the old behavior.

**Notes vs. code:** the v5.7.0 notes say `layer_weighting="linear"` restores the previous behavior exactly. It restores the v5.2.0 to v5.6.1 layer weights, but the KL teacher stays detached (change 3), and nothing restores v5.1.2: with identical inputs and the KL term off, v5.1.2 gave 0.109 and v5.6.1 gave 0.383, because of change 1. There is no setting that reproduces v5.1.2 training.

Before (v5.1.2):
```python
loss = losses.AdaptiveLayerLoss(model, losses.MultipleNegativesRankingLoss(model))
```

After (v5.7.0):
```python
loss = losses.AdaptiveLayerLoss(model, losses.MultipleNegativesRankingLoss(model), layer_weighting="linear")
# closest to the old decay; or keep the new "uniform" default
```

### `TripletLoss` with the default Euclidean (or Manhattan) distance was inverted until v5.4.0

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3704 · **Impact:** silent change in the training objective

`TripletLoss` defaults to `distance_metric=TripletDistanceMetric.EUCLIDEAN`. In v5.1.2, `EUCLIDEAN` and `MANHATTAN` returned `pairwise_euclidean_sim` / `pairwise_manhattan_sim`, which are *negative* distances, so the loss `relu(d(a, p) - d(a, n) + margin)` rewarded pushing positives away and pulling negatives closer. v5.4.0 negates them back into distances. `COSINE` was correct and is unchanged. `SparseTripletLoss` subclasses `TripletLoss` with the same default and is fixed in the same way.

**What changes:** for a triplet with ||a-p|| = 3.77 and ||a-n|| = 15.16 (margin 5), v5.1.2 reported a loss of 16.39 and v5.4.0 and v6.1.0 report 0.0. Models trained with v5.1.2's default `TripletLoss` were trained against the inverted objective; retraining with the same script on v5.4.0 or later gives different, correct results.

Before (v5.1.2):
```python
loss = losses.TripletLoss(model)   # default EUCLIDEAN: inverted
```

After (v5.4.0):
```python
loss = losses.TripletLoss(model)   # same call; now a real distance
```

### `MultipleNegativesSymmetricRankingLoss` and its cached variant are deprecated (v5.3.0)

**First release:** v5.3.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3607 · **Impact:** deprecation warning; still works in v6.1.0

Both classes are now thin subclasses of `MultipleNegativesRankingLoss` / `CachedMultipleNegativesRankingLoss` that pass `directions=("query_to_doc", "doc_to_query")` and `partition_mode="per_direction"`. Loss values are the same. Creating one emits `` DeprecationWarning: The MultipleNegativesSymmetricRankingLoss is deprecated and will be removed in a future release. Please use MultipleNegativesRankingLoss with `directions=('query_to_doc', 'doc_to_query')` and `partition_mode='per_direction'` instead. ``, which Python shows only when the loss is created in `__main__`. Side effects: `isinstance(loss, MultipleNegativesRankingLoss)` is now `True`, `get_config_dict()` (and the model card) gains the new keys, and the classes gain a `citation`.

**Notes vs. code:** the release notes describe the new `directions` and `partition_mode` options but not the deprecation.

Before (v5.1.2):
```python
loss = MultipleNegativesSymmetricRankingLoss(model, scale=20.0)
cached = CachedMultipleNegativesSymmetricRankingLoss(model, mini_batch_size=32)
```

After (v5.3.0):
```python
loss = MultipleNegativesRankingLoss(model, scale=20.0, directions=("query_to_doc", "doc_to_query"),
                                    partition_mode="per_direction")
cached = CachedMultipleNegativesRankingLoss(model, mini_batch_size=32, directions=("query_to_doc", "doc_to_query"),
                                            partition_mode="per_direction")
```

### `CachedMultipleNegativesRankingLoss` and other cached losses: positional arguments shifted (v5.3.0, v5.7.0)

**First release:** v5.3.0, v5.7.0 · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3607, https://github.com/huggingface/sentence-transformers/pull/3862 · **Impact:** hard error or silent misconfiguration for positional calls

| Version | Signature after `model` |
|---|---|
| v5.1.2 | `scale, similarity_fct, mini_batch_size, gather_across_devices, show_progress_bar` |
| v5.3.0 | `scale, similarity_fct, mini_batch_size, gather_across_devices, directions, partition_mode, show_progress_bar, hardness_mode, hardness_strength` |
| v5.7.0 + | `scale, similarity_fct, mini_batch_size, mini_batch_num_tokens, gather_across_devices, directions, partition_mode, show_progress_bar, ...` |

**What fails:**
- On v5.3.0 to v5.6.1, a positional sixth argument (`show_progress_bar` in v5.1.2) lands in `directions`: `True` raises `TypeError: 'bool' object is not iterable`, `False` raises `ValueError: At least one direction must be specified.`
- From v5.7.0, everything from the fifth position shifts again. A positional fifth argument (`gather_across_devices`) lands in `mini_batch_num_tokens`: `False` raises `ValueError: mini_batch_num_tokens must be a positive integer or None.`, and `True` is accepted silently as `mini_batch_num_tokens=True`. The sixth lands in `gather_across_devices`, so the Before call below runs without error on v5.7.0 and later, but with `mini_batch_num_tokens=True`, `gather_across_devices=True` and `show_progress_bar=False`.
- v5.7.0 also inserted `mini_batch_num_tokens` right after `mini_batch_size` in `CachedMultipleNegativesSymmetricRankingLoss` (fifth positional argument, `gather_across_devices` in v5.1.2), `CachedGISTEmbedLoss` (fifth positional argument, `show_progress_bar` in v5.1.2) and `CachedSpladeLoss`, with the same effect: `False` raises the `ValueError` above, and `True` silently becomes `mini_batch_num_tokens=True`.

`MultipleNegativesRankingLoss`, `SparseMultipleNegativesRankingLoss`, the CrossEncoder `CachedMultipleNegativesRankingLoss` and `MegaBatchMarginLoss` only gained parameters at the end.

Before (v5.1.2):
```python
loss = CachedMultipleNegativesRankingLoss(model, 20.0, util.cos_sim, 32, True, True)
```

After (v5.3.0 and later; also valid on v5.1.2):
```python
loss = CachedMultipleNegativesRankingLoss(model, scale=20.0, similarity_fct=util.cos_sim, mini_batch_size=32,
                                          gather_across_devices=True, show_progress_bar=True)
```

### `MultipleNegativesRankingLoss` internals and validation (v5.3.0, v6.0.0)

**First release:** v5.3.0 (rewrite), v6.0.0 (NaN check) · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3607 and https://github.com/huggingface/sentence-transformers/pull/3667 (v5.3.0), https://github.com/huggingface/sentence-transformers/pull/3794 (v6.0.0) · **Impact:** hard error in edge cases; model-card change

The default loss value is unchanged (verified to float32 rounding for pairs, triplets and 4-tuples). What changed:

- `scale <= 0` raises `ValueError: Scale must be a positive value.` (v5.3.0; the cached variant from v5.7.0), and from v6.0.0 so does `scale=nan`. This also covers the symmetric and sparse variants.
- The `cross_entropy_loss` attribute is gone; custom subclasses that call `self.cross_entropy_loss(...)` raise `AttributeError`.
- `get_config_dict()` gains `directions`, `partition_mode`, `hardness_mode` and `hardness_strength`, and the default citation changes from Henderson et al. 2017 to van den Oord et al. (arXiv 1807.03748, cited as 2019), so generated model cards differ.
- New opt-in options: `directions`, `partition_mode`, `hardness_mode`, `hardness_strength`. The defaults reproduce the old loss.

Before (v5.1.2):
```python
class MyMNRL(MultipleNegativesRankingLoss):
    def compute_loss_from_embeddings(self, embeddings, labels):
        scores = self.similarity_fct(embeddings[0], torch.cat(embeddings[1:])) * self.scale
        return self.cross_entropy_loss(scores, torch.arange(len(scores), device=scores.device))
```

After (v5.3.0):
```python
class MyMNRL(MultipleNegativesRankingLoss):
    def compute_loss_from_embeddings(self, embeddings, labels):
        scores = self.similarity_fct(embeddings[0], torch.cat(embeddings[1:])) * self.scale
        return torch.nn.functional.cross_entropy(scores, torch.arange(len(scores), device=scores.device))
```

### Cached losses share a new GradCache engine (v5.7.0)

**First release:** v5.7.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3862 · **Impact:** hard error for code that uses internals; silent gradient fixes

`CachedMultipleNegativesRankingLoss`, `CachedMultipleNegativesSymmetricRankingLoss`, `CachedGISTEmbedLoss`, `CachedSpladeLoss`, the CrossEncoder `CachedMultipleNegativesRankingLoss` and the mini-batched `MegaBatchMarginLoss` were rebuilt on a shared engine. Normal trainer use is unaffected, apart from the new optional `mini_batch_num_tokens` (pack mini-batches by token count). The notes mention only the removed `.cache` and `.random_states`.

**What fails for code that reaches into the losses:**
- `.cache` and `.random_states` are gone from all of them. `calculate_loss_and_cache_gradients()` is gone from the SentenceTransformer CMNRL (and so from its symmetric variant) and from `CachedSpladeLoss`; on `CachedGISTEmbedLoss` and the CrossEncoder CMNRL it now returns a `(loss, cache)` tuple instead of a tensor. `MatryoshkaLoss.cached_losses` is replaced by `uses_gradient_cache`.
- `calculate_loss(reps, with_backward=False)` became `calculate_loss(reps, labels=None, *, with_backward=False)`, so a positional `calculate_loss(reps, True)` silently binds `True` to `labels` and skips the backward pass (`CachedSpladeLoss` gained a `calculate_loss` with the same signature). On `CachedGISTEmbedLoss`, `with_backward` became keyword-only (`TypeError` if positional).
- The CMNRL `state_dict` contains the model twice (`model.*` and `loss.model.*`).
- `_backward_hook` can no longer be imported from the CMNRL module; it is in `sentence_transformers.base.losses.gradcache`. `RandContext` is still importable from the old modules.

**Silent fixes** (training results change): cached losses with `Pooling(include_prompt=False)` used a different attention mask in their second pass, giving wrong gradients (max difference 2.27 vs. 1e-6 after the fix); `MatryoshkaLoss(GISTEmbedLoss)` trained only the largest dimension; the CrossEncoder cached loss could use different dropout masks between passes on GPU (the PR author reports logit differences up to 1.4; not reproduced here).

Before (v5.1.2):
```python
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=32)
value = loss.calculate_loss_and_cache_gradients(reps)   # custom loop
grads = loss.cache
```

After (v5.7.0):
```python
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=32)   # or mini_batch_num_tokens=16384
value = loss(features, labels)                                          # use forward()
value.backward()
```

### `MegaBatchMarginLoss` trains on the whole batch (v5.7.0)

**First release:** v5.7.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3862 · **Impact:** silent change in loss values and gradients

In v5.1.2 the default mini-batched path returned, and back-propagated, only the **last** mini-batch's loss, and `mini_batch_size` was meant to divide the batch size. On v5.4.0 to v5.6.1 the same path crashed training with `KeyError: 'te'`, because it sliced the new `"modality"` string; use `use_mini_batched_version=False` on those versions. Now the loss is the mean over the whole batch regardless of `mini_batch_size`, identical to `use_mini_batched_version=False`. Logged loss values jump; the PR notes learning rates may need retuning. A third column raises `ValueError: MegaBatchMarginLoss expects exactly 2 input columns, (anchor, positive), but got 3. ...`, and `use_mini_batched_version=True` with a `StaticEmbedding` model raises `ValueError`.

**Notes vs. code:** the notes say a third column used to be "silently ignored". v5.1.2 and v5.6.1 already raised `ValueError: too many values to unpack (expected 2)`; only the message changed.

### `ContrastiveTensionLossInBatchNegatives` reads one column (v5.3.0)

**First release:** v5.3.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3639 · **Impact:** silent change when two columns differ

The loss now feeds its first column to both models and ignores any second column. The documented v5.1.2 input, two identical columns, gives the same loss (the second column is still tokenized, wasting compute). Two different columns now give a different loss, and a single-column dataset, which used to fail, now works. In the same release, `ContrastiveTensionLoss` derives labels automatically when none are given.

Before (v5.1.2):
```python
train_dataset = Dataset.from_dict({"anchor": sentences, "anchor_copy": sentences})
```

After (v5.3.0):
```python
train_dataset = Dataset.from_dict({"sentence": sentences})
```

### TSDAE with transformers v5: `RuntimeError` in v5.3.0 to v5.5.1, fixed in v5.6.0

**First release:** v5.3.0 (error), v5.6.0 (fix) · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3619, https://github.com/huggingface/sentence-transformers/pull/3781 · **Impact:** hard error on v5.3.0–v5.5.1 with transformers 5

transformers v5 removed the private helper `DenoisingAutoEncoderLoss` used to tie encoder and decoder weights. With transformers 5 and the default `tie_encoder_decoder=True`, v5.2.x fails with `AttributeError: type object 'PreTrainedModel' has no attribute '_tie_encoder_decoder_weights'`, and v5.3.0 to v5.5.1 raise `RuntimeError: Tying encoder and decoder weights is not currently supported for transformers version >= 5.0.0. Please either install transformers<5.0.0 or set tie_encoder_decoder=False.` (the suggested `tie_encoder_decoder=False` also needs `decoder_name_or_path`). v5.6.0 ships its own tying routine with the same result as v5.1.2, so a v5.1.2 TSDAE script runs unchanged on v5.6.0 and v6.x. With transformers 4.x every v5.x release works (v6.x requires transformers 5).

**Notes vs. code:** the v5.6.0 notes say "v5.5 raised a RuntimeError". The `RuntimeError` is in the code from v5.3.0.

### `MSELoss` is a subclass of the new `EmbedDistillLoss` (v5.5.0)

**First release:** v5.5.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3665 · **Impact:** additive; hard error only for the removed `loss_fct` attribute

`MSELoss` (and `SparseMSELoss`) give identical loss values, but the internal `loss_fct` attribute is gone (`AttributeError`). New: an optional `projection_dim` (a learnable projection when student and teacher dimensions differ) and per-column teacher labels. `labels=None` raises `ValueError`. `get_config_dict()` now returns `{"projection_dim": None}`, which shows in generated model cards (v5.1.2's `MSELoss` had no `get_config_dict`; v5.4.x returned `{}`).

### Listwise CrossEncoder losses fail on batches with no documents (v5.5.0)

**First release:** v5.5.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3690 · **Impact:** hard error in an edge case

`ListNetLoss`, `LambdaLoss`, `PListMLELoss` and their subclasses `ListMLELoss` and `RankNetLoss` used to return 0 for a batch in which every sample had an empty document list. That guard was removed (unflagged): such a batch now raises `ValueError: torch.cat(): expected a non-empty list of Tensors` (`RuntimeError` on older torch). Filter empty rows out, for example `dataset.filter(lambda row: len(row["docs"]) > 0)`.

### `ListMLELoss` and `PListMLELoss` exclude padding from the normalizer (v5.7.0)

**First release:** v5.7.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3827 · **Impact:** silent change for batches with lists of different lengths

Padded list positions used to add roughly unit mass to the Plackett-Luce normalizer. They are now masked out, so the loss and gradients change whenever a batch mixes list lengths (0.996 vs. 0.897 for `ListMLELoss` in a test). Batches without padding are unchanged.

### GIST and hard-negative relative margins work for negative similarities (v5.6.0)

**First release:** v5.6.0 · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3821, https://github.com/huggingface/sentence-transformers/pull/3817 · **Impact:** silent change when positive-pair similarities are negative; multi-GPU fix

The relative threshold changed from `positive * (1 - margin)` to `positive - |positive| * margin`, in `GISTEmbedLoss`/`CachedGISTEmbedLoss` with `margin_strategy="relative"` and in `mine_hard_negatives(relative_margin=...)`. For positive scores nothing changes. For negative positive scores, the old threshold sat above the positive score, so candidates more similar than the true positive were kept (mining) or not suppressed (GIST). Separately, with `gather_across_devices=True` and a non-zero margin, the GIST losses masked out the true positive on GPUs other than the first, producing an infinite loss there (verified from the code; needs multiple GPUs).

### `DistillKLDivLoss` validates its inputs and gains per-side temperatures (v6.0.0)

**First release:** v6.0.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3794 · **Impact:** hard error where training was silently wrong

`DistillKLDivLoss` (and `SparseDistillKLDivLoss`, used inside `SpladeLoss` and `CSRLoss`) now raise `ValueError` for:
- fewer than three columns: `DistillKLDivLoss expects at least 3 columns (query, positive, negative_1, ...), but got 2. ...` (v5 trained to a constant zero loss when each label was a one-element list, and raised `IndexError` for a scalar label);
- teacher scores whose shape is not `(batch_size, number_of_candidate_columns)` (v5 raised an unclear `IndexError` or `RuntimeError`, or silently broadcast a `(batch_size, 1)` label and trained on it);
- a temperature that is zero, negative or not finite (v5 returned NaN or trained);
- NaN teacher scores in the first batch the loss sees (later batches are not checked).

New optional `student_temperature` and `teacher_temperature` default to `temperature`, so values are identical when unset. A one-time warning appears when teacher scores underflow to probability zero.

Before (v5.1.2):
```python
loss = DistillKLDivLoss(model)   # (query, positive) + one score trained to a zero loss
```

After (v6.0.0):
```python
# columns: query, positive, negative_1[, ...]; label: one teacher score per candidate column
loss = DistillKLDivLoss(model, temperature=1.0)
```

### Teacher labels are detached in distillation losses (v6.0.0)

**First release:** v6.0.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3794 · **Impact:** silent only for labels computed inside the autograd graph

`MarginMSELoss`, `EmbedDistillLoss`/`MSELoss` and `DistillKLDivLoss`, and the CrossEncoder `MSELoss` and `MarginMSELoss`, call `.detach()` on the labels. Dataset labels never carry gradients, so values are unchanged; custom setups that computed labels from a trainable teacher in the same graph no longer train the teacher.

### Custom CrossEncoder losses must call the model differently (v5.4.0)

**First release:** v5.4.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** hard error during training

The CrossEncoder trainer now calls `loss(features, labels, prompt=prompt, task=task)` and, on `TypeError`, retries without the new arguments, so an old `forward(self, inputs, labels)` signature is tolerated. But the v5.1.2 pattern inside the loss, `self.model(**tokens)`, fails because [`model(**features)` no longer runs the Hugging Face model](#modelfeatures-no-longer-runs-the-hugging-face-model-v540). Built-in CrossEncoder losses keep their constructor signatures.

**What fails:** `TypeError: BaseModel.forward() missing 1 required positional argument: 'input'` during `trainer.train()`.

**Notes vs. code:** the migration guide says existing custom losses "will continue to work"; that holds for the signature only.

Before (v5.1.2):
```python
class MyLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model, self.bce = model, torch.nn.BCEWithLogitsLoss()
    def forward(self, inputs, labels):
        tokens = self.model.tokenizer(list(zip(inputs[0], inputs[1])), padding=True, truncation=True,
                                      return_tensors="pt").to(self.model.device)
        return self.bce(self.model(**tokens)[0].view(-1), labels.float())
```

After (v5.4.0):
```python
class MyLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model, self.bce = model, torch.nn.BCEWithLogitsLoss()
    def forward(self, inputs, labels, prompt=None, task=None):
        features = self.model.preprocess(list(zip(inputs[0], inputs[1])), prompt=prompt, task=task).to(self.model.device)
        return self.bce(self.model(features)["scores"].view(-1), labels.float())
```

### CrossEncoder `MultipleNegativesRankingLoss` no longer samples a row's own positive as a negative (v5.3.0)

**First release:** v5.3.0 · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3636, https://github.com/huggingface/sentence-transformers/pull/3641 · **Impact:** silent change in small-batch edge cases

With hard-negative columns, `num_negatives=None` (or at least the number of candidates) crashed with `IndexError`, and a `num_negatives` between the number of valid and total candidates could sample the anchor's own positive or hard negative as a negative, which happens for the small last batch of an epoch. Both are fixed; pair-only data and smaller `num_negatives` give identical losses.

### Smaller loss changes

- **`AnglELoss`** (v5.3.0, https://github.com/huggingface/sentence-transformers/pull/3609) accepts (anchor, positive, negative, ...) tuples without labels. Pairs without labels raise `ValueError: AnglELoss requires labels for datasets with pairs.` instead of `TypeError`. A 3+ column dataset that also has a label column now uses the extra columns.
- **`ContrastiveLoss` and `OnlineContrastiveLoss`** (v5.7.0, https://github.com/huggingface/sentence-transformers/pull/3838) emit a one-time `UserWarning` when labels are not 0 or 1. The math is unchanged; this matters under `-W error`.
- **Loss model-card configs** (v6.0.0, https://github.com/huggingface/sentence-transformers/pull/3794): `MultipleNegativesRankingLoss` (and its cached variant), `MarginMSELoss`, `DistillKLDivLoss` and `ContrastiveTensionLoss` render `similarity_fct` through `util.similarity_fct_name`, so `functools.partial` objects get readable names. For MNRL, a partial raised `AttributeError` in `get_config_dict()` on v5.1.2 to v5.3.0 and was rendered as its raw `repr` on v5.4.0 to v5.7.0. `CoSENTLoss`, `AnglELoss` and `GlobalOrthogonalRegularizationLoss` still raise `AttributeError` for a partial in v6.1.0.
- **New losses** (additive): `GlobalOrthogonalRegularizationLoss` and sparse `CachedSpladeLoss` (v5.3.0), `EmbedDistillLoss` and CrossEncoder `ADRMSELoss` (v5.5.0).

## Evaluators

Evaluator class names and returned metric keys are unchanged across the whole range, apart from the `ReciprocalRankFusionEvaluator` primary-metric prefix noted below. Several evaluators changed the *values* they report.

### NanoBEIR evaluators take a new second parameter and load from a single dataset repository (v5.2.0)

**First release:** v5.2.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3583 · **Impact:** hard error for positional calls, offline use and removed module constants

`NanoBEIREvaluator`, `SparseNanoBEIREvaluator` and `CrossEncoderNanoBEIREvaluator` gained `dataset_id="sentence-transformers/NanoBEIR-en"` as the **second** parameter (it allows translated NanoBEIR versions). Metric keys and values are unchanged.

**What fails:**
- A positional second argument (`mrr_at_k` for the dense and sparse evaluators, `rerank_k` for the CrossEncoder one) becomes the dataset id: `ValueError: Could not load subset 'corpus' split 'NanoSciFact' from dataset '[5]'.`
- The data now comes from one repository, `sentence-transformers/NanoBEIR-en`, instead of 13 per-dataset repositories, so an offline environment that only cached the old ones fails (`ValueError: Could not load subset 'corpus' split 'NanoSciFact' from dataset 'sentence-transformers/NanoBEIR-en'.`).
- The module-level `dataset_name_to_id` was removed (`ImportError`) and `dataset_name_to_human_readable` renamed `DATASET_NAME_TO_HUMAN_READABLE`.

Before (v5.1.2):
```python
evaluator = NanoBEIREvaluator(["msmarco", "nq"], [10, 100])
ce_evaluator = CrossEncoderNanoBEIREvaluator(["msmarco"], 50)
```

After (v5.2.0; also valid on v5.1.2):
```python
evaluator = NanoBEIREvaluator(["msmarco", "nq"], mrr_at_k=[10, 100])
ce_evaluator = CrossEncoderNanoBEIREvaluator(["msmarco"], rerank_k=50)
```

### CrossEncoder evaluators: `batch_size` is honored (v5.3.0), and `prompt_name` comes before it (v5.4.0)

**First release:** v5.3.0, v5.4.0 · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3643, https://github.com/huggingface/sentence-transformers/pull/3554 · **Impact:** silent memory change; hard error for positional `batch_size`

`CrossEncoderClassificationEvaluator` and `CrossEncoderCorrelationEvaluator` accepted `batch_size` but always predicted with 32; from v5.3.0 the value is used, so a large value can now run out of memory. In v5.4.0, `CrossEncoderRerankingEvaluator`, `CrossEncoderCorrelationEvaluator` and `CrossEncoderNanoBEIREvaluator` gained `prompt_name` right before `batch_size`.

**What fails:** `CrossEncoderRerankingEvaluator(samples, 10, True, "dev", 32)` raises `ValueError: Prompt name '32' not found in the configured prompts dictionary with keys [].` when the evaluator runs, possibly in the middle of training.

Before (v5.1.2):
```python
ev = CrossEncoderRerankingEvaluator(samples, 10, True, "dev", 32)
```

After (v5.4.0; also valid on v5.1.2):
```python
ev = CrossEncoderRerankingEvaluator(samples, at_k=10, always_rerank_positives=True, name="dev", batch_size=32)
```

### `BinaryClassificationEvaluator` reported inverted Euclidean and Manhattan metrics until v5.7.0

**First release:** v5.7.0 · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3886, https://github.com/huggingface/sentence-transformers/pull/3878, https://github.com/huggingface/sentence-transformers/pull/3853 · **Impact:** silent change in reported metrics

- The `*_euclidean_*` and `*_manhattan_*` accuracy, F1, precision, recall, AP, MCC and thresholds evaluated the opposite classifier. They are now correct: in a test, Euclidean accuracy went from 0.5 to 0.875, and thresholds became positive distances instead of negative numbers. `max_*` aggregates and `primary_metric` change accordingly, which can change `metric_for_best_model` / `load_best_model_at_end` decisions. `SparseBinaryClassificationEvaluator` inherits the fix.
- The results CSV had its values shifted under the wrong headers (even in the default cosine-only setup); the returned dict was correct. Don't append new rows to CSV files written by v5.6.1 or earlier.
- `similarity_fn_names` without `"cosine"` (for example `["dot", "euclidean"]`) used to raise `KeyError: 'cosine'`; it works now.

No code change; re-baseline stored Euclidean or Manhattan metrics and thresholds.

### `ReciprocalRankFusionEvaluator` uses canonical RRF and prefixes `primary_metric` (v5.7.0)

**First release:** v5.7.0 · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3883, https://github.com/huggingface/sentence-transformers/pull/3861 · **Impact:** silent change in fused metrics; key change

Fusion now uses 1-based ranks and gives no contribution from a retriever that didn't return a document (it used 0-based ranks and treated a missing document as ranked last), so fused rankings and metrics change. With `name="hybrid"`, `primary_metric` changes from `"ndcg@10"` to `"hybrid_ndcg@10"`, which makes `results[evaluator.primary_metric]` work (it raised `KeyError` before).

Before (v5.1.2):
```python
res = rrf_eval(); score = res[f"{rrf_eval.name}_{rrf_eval.primary_metric}"]
```

After (v5.7.0):
```python
res = rrf_eval(); score = res[rrf_eval.primary_metric]   # already prefixed
```

### `TripletEvaluator` encodes anchors as queries and validates its arguments (v6.0.0)

**First release:** v6.0.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3794 · **Impact:** silent metric change for models with query/document prompts; hard error for typos and some subclasses

`TripletEvaluator` and `SparseTripletEvaluator` now embed anchors with `encode_query` and positives and negatives with `encode_document`. Models with `query`/`document` prompts or routes report different accuracy (1.0 vs. 0.75 for the test model); others are unaffected. Metric keys are unchanged.

**What fails:**
- Unknown `similarity_fn_names` or `margin` keys raise `` ValueError: `similarity_fn_names` got unexpected names ['cos'], expected names in ['cosine', 'dot', 'manhattan', 'euclidean']. `` (v5 silently produced no metric, or ignored the margin).
- Subclasses that override `embed_inputs(self, model, sentences)` raise `TypeError: ... got an unexpected keyword argument 'encode_fn_name'`; overrides that pass `**kwargs` on to `model.encode` raise `ValueError` from `encode`'s argument check.

Before (v5.1.2):
```python
ev = TripletEvaluator(anchors, positives, negatives, similarity_fn_names=["cos"], margin={"cos": 0.1})

class MyEval(TripletEvaluator):
    def embed_inputs(self, model, sentences):
        return model.encode(sentences, normalize_embeddings=True)
```

After (v6.0.0):
```python
ev = TripletEvaluator(anchors, positives, negatives, similarity_fn_names=["cosine"], margin={"cosine": 0.1})

class MyEval(TripletEvaluator):
    def embed_inputs(self, model, sentences, encode_fn_name=None, **kwargs):
        fn = {"query": model.encode_query, "document": model.encode_document}.get(encode_fn_name, model.encode)
        return fn(sentences, normalize_embeddings=True)
```

### `InformationRetrievalEvaluator` breaks exact score ties by corpus ID (v6.0.0)

**First release:** v6.0.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3794 · **Impact:** silent metric change where scores tie exactly

v5 kept the tied documents with the largest corpus IDs, and which of them survived could also depend on `corpus_chunk_size` boundaries. Ties are now broken by corpus ID (smallest first), so metrics on exact ties (duplicate documents, quantized embeddings) can change even at a fixed chunk size, and no longer depend on it. In a test with exact ties, v5.7.0 gave NDCG@10 between 0.33 and 0.43 depending on the chunk size and v6.0.0 gave 0.33 for every chunk size. Scores that differ only in the last float bit are not ties and can still move. The sparse IR and NanoBEIR evaluators inherit this. Separately, v5.3.0 made `write_predictions=True` create a missing `output_path` instead of raising `FileNotFoundError`.

## Utilities

### `mine_hard_negatives`: `n-tuple-scores` is deprecated and its column is renamed (v5.2.0)

**First release:** v5.2.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3506 · **Impact:** logged deprecation plus a column rename

`output_format="n-tuple-scores"` logs `` `output_format="n-tuple-scores"` is deprecated. Use `output_format="n-tuple"` with `output_scores=True` instead ... `` and is rewritten to that. The scores column is now named **`scores`**; v5.1.2 wrote **`score`**. Values and other columns are identical. Still accepted in v6.1.0.

**What fails:** `ds["score"]` and similar column operations raise `KeyError` or a `datasets` `ValueError`. Training is unaffected, because v5.2.0 also treats `scores` as a label column.

**Notes vs. code:** the notes don't mention the column rename.

Before (v5.1.2):
```python
ds = mine_hard_negatives(dataset, model, num_negatives=5, output_format="n-tuple-scores")
teacher_scores = ds["score"]
```

After (v5.2.0):
```python
ds = mine_hard_negatives(dataset, model, num_negatives=5, output_format="n-tuple", output_scores=True)
teacher_scores = ds["scores"]
```

### `mine_hard_negatives`: different output for multiple positives, duplicates and negative scores (v5.2.0, v5.6.0)

**First release:** v5.2.0, v5.6.0 · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3506, https://github.com/huggingface/sentence-transformers/pull/3821, https://github.com/huggingface/sentence-transformers/pull/3816 · **Impact:** silent change in mined datasets

- **v5.2.0:** duplicate (anchor, positive) rows are deduplicated; `labeled-pair` with several positives per query attached negatives to the wrong query; and margin filtering used the wrong positive score once queries had different numbers of positives. All three are fixed. With one positive per query and no duplicates, the output is identical across v5.1.2, v5.2.0 and v6.1.0.
- **v5.6.0:** `relative_margin` uses `positive - |positive| * margin` instead of `positive * (1 - margin)`, which differs only when the positive pair's similarity is negative (for example with dot-product models). The non-FAISS path now computes similarities in chunks of `faiss_batch_size`, shows a "Computing similarity scores" progress bar even with `verbose=False`, and raises `ValueError: faiss_batch_size must be a positive integer, got 0.` for non-positive values.

Re-mine datasets cached from v5.1.2 if they came from multi-positive data.

### `mine_hard_negatives`: input validation, small corpora and cache keys (v5.7.0)

**First release:** v5.7.0 · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3904, https://github.com/huggingface/sentence-transformers/pull/3872, https://github.com/huggingface/sentence-transformers/pull/3873, https://github.com/huggingface/sentence-transformers/pull/3871 · **Impact:** hard error for invalid settings; silent output changes

- New `ValueError`s before any encoding: `num_negatives must be at least 1, got num_negatives=0.`, `range_min must be non-negative, got range_min=-2.`, `range_min must be smaller than range_max, ...`, and `Cannot mine 3 negatives per query: only 2 candidates remain ...`. In v5.6.1, `num_negatives=0` returned an empty dataset, a negative `range_min` silently mined the *easiest* candidates, and too-small windows crashed with `RuntimeError` or `IndexError`.
- Corpora smaller than `range_max` no longer crash (`RuntimeError: selected index k out of range`). The FAISS path retrieves `range_max + max_positives` candidates (was `range_max + 1`), ignores FAISS's `-1` padding, and caps an automatically derived `range_max` (at `2048 - max_positives`) only on GPU FAISS; v5.6.1 capped it at 2048 on CPU FAISS too.
- `use_faiss=True` without `faiss` installed raises `ImportError` before encoding.
- With `cache_folder`, the cache key now includes the prompts, so caches written by v5.6.1 or earlier are recomputed once (old `.npy` files stay on disk). Older releases could reuse embeddings computed with a different prompt.

Before (v5.1.2):
```python
ds = mine_hard_negatives(pairs, model, range_max=2, num_negatives=3)   # crashed mid-run
```

After (v5.7.0):
```python
ds = mine_hard_negatives(pairs, model, range_max=5, num_negatives=3)   # window must fit num_negatives
```

### int8/uint8 quantization saturates and floors; binary quantization packs per embedding (v5.7.0)

**First release:** v5.7.0 · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3847, https://github.com/huggingface/sentence-transformers/pull/3825 · **Impact:** silent change in quantized values

`quantize_embeddings(..., precision="int8"/"uint8")` cast values straight to integers, so values outside the calibration range wrapped around (the PR example gives `[-42, 42, -112]` where `[-128, 42, 127]` is correct), and int8 truncated toward zero. v5.7.0 clips to the bounds and floors. As a result, **int8 output is not bit-identical with v5.1.2**: about half of all int8 values in a realistic test moved down by one level. uint8 changes only where values used to wrap. `semantic_search_usearch(..., corpus_precision="int8")` and `semantic_search_faiss(..., corpus_precision="uint8")` quantize queries internally and are affected too.

`binary`/`ubinary` packing used to run over the flattened batch: for dimensions that are not a multiple of 8 it raised `ValueError: cannot reshape array ...` or, for some batch shapes, silently mixed bits across embeddings. It now packs each embedding separately; dimensions divisible by 8 are unchanged.

Separately, since v5.4.0 a dimension that is constant in the calibration data quantizes to the lowest bucket (-128 for int8) instead of v5.1.2's undefined NaN cast (0 in a test, with a `RuntimeWarning`).

Re-quantize stored int8 corpora (don't mix old and new int8 arrays) and rebuild binary indexes made with non-multiple-of-8 dimensions:
```python
corpus_int8 = quantize_embeddings(corpus_emb, precision="int8", calibration_embeddings=calib)
```

### `semantic_search_faiss` validates `corpus_precision` and drops padding results (v5.7.0)

**First release:** v5.7.0 · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3887, https://github.com/huggingface/sentence-transformers/pull/3888 · **Impact:** hard error for unsupported precisions; shorter result lists on small corpora

Only `"float32"`, `"uint8"` and `"ubinary"` are accepted (`ValueError: corpus_precision must be "float32", "uint8", or "ubinary" for faiss`). In v5.1.2, `"int8"` and `"binary"` crashed with `AttributeError` when `corpus_embeddings` was given, but silently ran against a pre-built `corpus_index`. When `top_k` exceeds the corpus size, results no longer contain `{"corpus_id": -1, ...}` placeholders; with rescoring those placeholders could push a real document out of the results. Lists can now be shorter than `top_k`.

### `quantize_embeddings` returns a list for a list of 2-D arrays (v6.0.0)

**First release:** v6.0.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3794 · **Impact:** silent return-type change

A list whose first element is a 2-D array (for example per-token embeddings) is quantized matrix by matrix and returned as a `list` of arrays, with int8 ranges shared across all tokens. v5 stacked same-shaped matrices into one 3-D array (and failed on ragged lists). `quantize_embeddings([])` returns `[]` instead of raising `IndexError`, and a `(0, dim)` matrix returns an empty array. Lists of 1-D vectors and plain 2-D arrays, the usual sentence-embedding case, are unchanged.

Before (v5.1.2):
```python
q = quantize_embeddings([m1, m2], precision="int8")   # one (2, T, D) array
```

After (v6.0.0):
```python
q = np.stack(quantize_embeddings([m1, m2], precision="int8"))   # list -> stack it yourself
```

### `util.http_get` uses `httpx` (v5.2.2)

**First release:** v5.2.2 · **PR:** https://github.com/huggingface/sentence-transformers/pull/3618 · **Impact:** exception type change

`sentence_transformers.util.http_get` (also used by `WordEmbeddings.from_text_file` to download a missing embeddings file by name) switched from `requests` to a lazily imported `httpx`, which is not declared as a dependency (it comes with huggingface-hub 1.x and `datasets`). HTTP errors raise `httpx.HTTPStatusError` instead of `requests.exceptions.HTTPError` (which subclassed `OSError`, so `except OSError` no longer catches them), and without httpx it raises `` ImportError: httpx is required to use this function. Please install it via `pip install httpx`. `` Partial downloads are cleaned up.

Before (v5.1.2):
```python
try:
    http_get(url, "data.tsv.gz")
except requests.exceptions.HTTPError:
    ...
```

After (v5.2.2):
```python
import httpx

try:
    http_get(url, "data.tsv.gz")
except httpx.HTTPStatusError:
    ...
```

### Other utility changes

- **`community_detection`** (v5.7.0, https://github.com/huggingface/sentence-transformers/pull/3900): similarities exactly equal to the threshold now expand the search window, so communities are no longer capped at about 50 members when many scores tie (120 identical vectors with `threshold=1.0`: one community of 50 before, 120 after).
- **`pairwise_angle_sim`** (v5.3.0, https://github.com/huggingface/sentence-transformers/pull/3623) supports odd embedding dimensions (it raised `RuntimeError`).
- **Generated model cards**: v5.3.0 (https://github.com/huggingface/sentence-transformers/pull/3680) drops a `dataset_name` column from the training-data section and no longer raises `KeyError` with `set_transform`; v6.0.0 changes the default wording from "maps sentences & paragraphs to" to "maps inputs to". Tests that compare generated README text need updating.

## Multimodal inputs

Multimodal input arrived in v5.4.0: `encode`, `predict` and `rank` accept images, audio, video, dicts that combine them, and chat messages, on models that support those modalities. v5.1.2 had only text (plus the `CLIPModel` module), so none of this changes v5.1.2 code, except where a text-only feature now takes a multimodal path (dict inputs, chat templates and media-looking strings; see [SentenceTransformer inference](#sentencetransformer-inference) and [CrossEncoder](#crossencoder)). The entries below matter if you adopted multimodal inputs on v5.4 to v6.0.

### Stricter audio and video batches (v5.7.0)

**First release:** v5.7.0 · **PRs:** https://github.com/huggingface/sentence-transformers/pull/3876, https://github.com/huggingface/sentence-transformers/pull/3866, https://github.com/huggingface/sentence-transformers/pull/3841, https://github.com/huggingface/sentence-transformers/pull/3897 · **Impact:** hard error for mixed batches

- A batch mixing audio sampling rates used to be processed silently at the last rate; it now raises `ValueError: This batch mixes audio with different sampling rates (16000 and 8000), but the processor accepts a single sampling rate per batch. ...`. Resample, or encode each rate separately.
- Per-sample `video_metadata` stays aligned with its video; a batch that mixes path or URL videos with and without metadata raises `ValueError`.
- `{"video": ["f1.jpg", "f2.jpg"]}` is no longer mistaken for a text pair. A torchcodec install that raises `RuntimeError` on import no longer breaks `import sentence_transformers`.

### Content order, lists and video frames in chat-template inputs (v6.1.0)

**First release:** v6.1.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/4037 · **Impact:** silent embedding and score change

Items 1 and 2 only apply when inputs are rendered through a chat template (processors without one, such as CLIP's, are unaffected); item 3 is decided while parsing inputs:

1. **Content follows each input's own key order.** CrossEncoder pair elements like `{"text": caption, "image": img}` were rendered in alphabetical key order (image first); they now follow the dict's order (text first). In a batch whose dicts have different key orders, each input now keeps its own order instead of the first input's. To keep the pre-6.1 rendering, order the keys of CrossEncoder pair elements alphabetically, and give every input in a batch the first input's key order.
2. **Lists under a modality key become separate items** in one input: `{"text": ["19 inch monitor", "Black finish"], "image": [img_a, img_b]}` gives four content items. Before, the list was placed in a single item. A text-only input with exactly two strings, such as `{"text": ["query", "document"]}`, is still a query/document pair.
3. **A list under `"video"` is one video only if every item is recognized as an image** (PIL images, 3-D arrays, image URLs with an image extension, existing image files, `data:image/` URIs). Otherwise each item becomes a separate video, so frame URLs without an image extension now produce several one-frame videos. The wrapper `{"array": frames, "video_metadata": None}` always means one video.

Before (v5.7.0 to v6.0.1; on v5.4.0 to v5.6.1 exactly two string frames were mistaken for a query/document text pair):
```python
emb = model.encode({"video": ["https://host/frame/1", "https://host/frame/2"]})   # one 2-frame video
```

After (v6.1.0):
```python
emb = model.encode({"video": {"array": ["https://host/frame/1", "https://host/frame/2"], "video_metadata": None}},
                   processing_kwargs={"video": {"do_sample_frames": False}})
```

`MultimodalInput` became a `TypedDict` in the same release, which only matters for static type checkers.

### `load_audio_from_video` can be set under `chat_template` (v6.1.0)

**First release:** v6.1.0 · **PR:** https://github.com/huggingface/sentence-transformers/pull/4048 · **Impact:** bug fix

With transformers 5.4 or newer, setting `processing_kwargs={"chat_template": {"load_audio_from_video": True}}` raised `TypeError: ... got multiple values for keyword argument 'load_audio_from_video'`, and with transformers 5.0 to 5.3 the `video` setting did not reach the top level. Both spellings work from v6.1.0, and the `chat_template` value wins if both are given.

## Hidden breakage for custom modules, custom losses and remote code

If your project extends Sentence Transformers rather than just calling it, check these entries first. They are easy to miss because the release notes are silent or describe them as compatible:

- **Custom module classes** need `trust_remote_code=True` from v6.0.0, including models in local directories and classes from your own installed packages ([details](#custom-module-classes-require-trust_remote_codetrue-including-local-directories-v560-warning-v600-error)). Trainer checkpoint reloads exempt only subclasses of `sentence_transformers.base.modules.Module`, and `load_best_model_at_end` fails silently ([details](#trainer-checkpoint-reload-of-custom-modules-v600)).
- **Custom input modules**: override `preprocess`, not `tokenize`; a `Transformer` subclass that only overrides `tokenize` is silently bypassed from v5.4.0 ([details](#custom-input-modules-override-preprocess-not-tokenize-v540)). Behind a `Router`, `tokenize`/`preprocess` must accept `**kwargs` from v6.0.1 ([details](#router-passes-task-to-the-routed-module-v601)).
- **Modules after `Transformer`** that read `attention_mask` break with flash-attention input flattening ([details](#flash-attention-2-inputs-are-flattened-and-roberta-family-position-ids-were-wrong-until-v561-v540)), and no longer see a prompt-masked `attention_mask` from v6.0.0 ([details](#poolinginclude_promptfalse-leaves-attention_mask-intact-v600)).
- **Custom `Dense` activations** outside `torch.*` fall back to `Tanh`, with only a logged warning, from v5.4.1 unless the model is trusted ([details](#dense-only-loads-torch-activation-functions-unless-the-model-is-trusted-v541)).
- **Subclasses of `SentenceTransformer`, `CrossEncoder` or `SparseEncoder`** save and load with the base `model_type` from v5.5.0; checkpoints saved through a subclass on v5.4.1 or earlier lose modules on reload ([details](#subclasses-of-the-model-classes-inherit-the-base-model_type-v550)). Private hooks such as `_load_auto_model`, `_load_sbert_model` and `_encode_multi_process` were renamed in v5.4.0 ([details](#smaller-inference-changes-v540)).
- **Custom losses**: `MultipleNegativesRankingLoss.cross_entropy_loss` is gone (v5.3.0); the cached losses lost `.cache` and `.random_states`, `calculate_loss_and_cache_gradients` was removed or now returns a tuple, and `calculate_loss` changed (v5.7.0); per-column feature dicts are no longer filled in place by merged forward passes (v6.0.0); CrossEncoder losses must call `model(model.preprocess(pairs))["scores"]` instead of `model(**tokens)` (v5.4.0). See [Losses](#losses).
- **Custom evaluators**: `TripletEvaluator.embed_inputs` receives `encode_fn_name` from v6.0.0 ([details](#tripletevaluator-encodes-anchors-as-queries-and-validates-its-arguments-v600)).
- **Custom trainers**: `get_eval_dataloader` receives dataset names from v5.7.0 ([details](#trainer-subclasses-receive-dataset-names-in-get_eval_dataloader-v570)), and the CrossEncoder and SparseEncoder trainers no longer subclass the SentenceTransformer trainer ([details](#crossencoder-and-sparseencoder-trainers-no-longer-subclass-the-sentencetransformer-ones-v540)).
- **Code that touches internals**: `model.tokenizer = ...` raises and `model[0].auto_model = ...` is ignored ([details](#modeltokenizer-can-no-longer-be-assigned-and-assigning-auto_model-is-silently-ignored-v540)); `model.tokenize()` output contains a string ([details](#tokenize-is-deprecated-and-its-output-gains-a-modality-string-v540)); `CrossEncoder.config.sentence_transformers` is no longer populated ([details](#crossencoder-is-now-a-module-chain-what-happened-to-its-attributes-v540)).
- **Remote-code modules**: not a code change, but the v6.1.0 documentation now warns that transformers' static import scan (which applies in every release since v5.1.2) turns unguarded optional imports in custom modules into hard requirements; guard them with `try`/`except`.

## Per-version summary

One entry per stable release after v5.1.2. "Notes vs. code" lists places where the release notes (or, where named, the project's migration guide or a PR description) disagree with what the released code does; the guide always follows the code.

### v5.2.0

Released 2025-12-11 · [release notes](release-notes/v5.2.0.md)

- Python 3.10 is required ([details](#python-310-or-newer-is-required-v520)); the transformers cap is raised to `<6`, so fresh installs get transformers v5 ([details](#fresh-installs-of-v52x-get-transformers-v5-v520)); Pillow moves to the `image` extra ([details](#pillow-is-no-longer-installed-by-default-v520)); the ONNX extras point to `optimum-onnx` and the OpenVINO extra drops its `>=1.20.0` floor.
- NanoBEIR evaluators gain a second positional `dataset_id` and load from one repository ([details](#nanobeir-evaluators-take-a-new-second-parameter-and-load-from-a-single-dataset-repository-v520)).
- `mine_hard_negatives`: `n-tuple-scores` deprecated and its column renamed `score` to `scores`; fixes for multiple positives and duplicates.
- Training collator treats `labels`/`scores` columns as labels; `AdaptiveLayerLoss`/`Matryoshka2dLoss` include the embedding layer.
- Additive: CrossEncoder multi-processing, `output_scores`.
- **Notes vs. code:** the notes call Python 3.9 "deprecated"; the metadata drops it. The `score` to `scores` column rename is not mentioned, and the `AdaptiveLayerLoss` change appears only as a fix (#3560) to when layer embeddings are restored, without saying the embedding layer is now trained.

### v5.2.1

Released 2026-01-26 · [release notes](release-notes/v5.2.1.md)

No breaking changes. Patch release fixing training with transformers 5.0, which crashed at the first checkpoint save (`'SentenceTransformerTrainingArguments' object has no attribute 'save_safetensors'`), and declaring `numpy` as a dependency. Importing still fails when `requests` is missing (fixed in v5.2.2), and logging still fails with transformers 5.2 or newer (fixed in v5.2.3).

### v5.2.2

Released 2026-01-27 · [release notes](release-notes/v5.2.2.md)

Patch release removing the undeclared module-level `import requests`, so `import sentence_transformers` no longer fails on installs without `requests`. Small breaking change for direct users of `util.http_get`: it now uses `httpx`, raising `httpx.HTTPStatusError` instead of `requests.exceptions.HTTPError`, and `ImportError` if httpx is missing ([details](#utilhttp_get-uses-httpx-v522)).

### v5.2.3

Released 2026-02-17 · [release notes](release-notes/v5.2.3.md)

No breaking changes. Patch release fixing training with transformers 5.2 or newer, which crashed at the first logging step (`'SentenceTransformerTrainer' object has no attribute '_nested_gather'`).

### v5.3.0

Released 2026-03-12 · [release notes](release-notes/v5.3.0.md)

- `MultipleNegativesRankingLoss` rewritten with `directions`, `partition_mode` and hardness weighting; `scale <= 0` rejected; `cross_entropy_loss` removed ([details](#multiplenegativesrankingloss-internals-and-validation-v530-v600)). The symmetric MNRL losses are deprecated ([details](#multiplenegativessymmetricrankingloss-and-its-cached-variant-are-deprecated-v530)), and `CachedMultipleNegativesRankingLoss` positional arguments shift ([details](#cachedmultiplenegativesrankingloss-and-other-cached-losses-positional-arguments-shifted-v530-v570)).
- `GroupByLabelBatchSampler` rewritten ([details](#groupbylabelbatchsampler-was-rewritten-v530)); new `NO_DUPLICATES_HASHED` sampler.
- `warmup_ratio` kept as a Sentence Transformers argument ([details](#warmup_ratio-is-kept-as-a-sentence-transformers-argument-v530)).
- `Pooling(include_prompt=False)` handles left padding; `ContrastiveTensionLossInBatchNegatives` reads one column; TSDAE raises a clear `RuntimeError` on transformers 5; `_load_t5_model` removed.
- Fixes: CrossEncoder MNRL negatives, CrossEncoder evaluator `batch_size`, IR evaluator `write_predictions`, odd-dimension `pairwise_angle_sim`.
- Additive: `GlobalOrthogonalRegularizationLoss`, `CachedSpladeLoss`, AnglE n-tuples, T5Gemma support.
- **Notes vs. code:** the notes list changes that already shipped in v5.2.1 to v5.2.3 (#3596, #3597, #3608, #3615, #3618, #3664) as if they were new. The deprecation of the symmetric MNRL losses and the `CachedMultipleNegativesRankingLoss` positional break are not mentioned. (PR #3667's description also mentions the GIST losses, but as the notes say, only MNRL and its cached variant gained hardness weighting.)

### v5.4.0

Released 2026-04-09 · [release notes](release-notes/v5.4.0.md)

The large restructure (#3554): a shared `base` package, multimodal inputs, and a modular `CrossEncoder`.

- Import paths moved, with deprecation aliases for most ([details](#import-paths-moved-old-paths-still-work-with-a-deprecationwarning-v540)); a few have none ([details](#some-old-import-paths-were-not-kept-v540)).
- Hard errors: keyword-only constructors for `SentenceTransformer`, `SparseEncoder` and `Transformer`; `model.tokenizer` assignment; pandas `Series` and non-string inputs; `{"query": ...}` dict inputs on models without a `Router` (and `{"text": ...}` until v5.5.1); `encode(output_value=None)` on five or more inputs; removed `Pooling`/`LSTM`/`CNN` attributes; `CrossEncoder.set_activation_fn`/`set_config_value`, `model(**tokens)`, `save(**hf_kwargs)`, `push_to_hub(tags=)`; positional `batch_size` in CrossEncoder `predict`/`rank`/evaluators; collator `tokenize_fn`; the default mini-batched `MegaBatchMarginLoss` (`KeyError: 'te'`, until v5.7.0); `SparseEncoder.encode_multi_process`/`fit`/`load` and `max_active_dims=0`; `CLIPModel(model_name=)`; positional `MLMTransformer` arguments; `SparseAutoEncoder.preprocess`.
- Silent changes: whitespace no longer stripped; `max_seq_length` follows `tokenizer.model_max_length`; `auto_model` assignment ignored; raw causal-LM checkpoints get last-token pooling and a chat template; CrossEncoder rerankers with a chat template get chat-formatted input; raw causal-LM CrossEncoders become generative rerankers; `predict(activation_fn=)` no longer sticks; CrossEncoder positional arguments past the fifth dropped; CrossEncoder training prompts applied; the `TripletLoss` default distance fixed ([details](#tripletloss-with-the-default-euclidean-or-manhattan-distance-was-inverted-until-v540)).
- Saved models cannot be loaded by v5.3.x or earlier ([details](#models-saved-by-v540-or-later-cannot-be-loaded-by-v53x-or-earlier-v540)).
- Deprecations still working in v6.1.0: `encode(sentences=)`, `tokenizer_kwargs`, `get_sentence_embedding_dimension`, `truncate_sentence_embeddings`, `tokenize()`, `Transformer`/`Pooling`/`SpladePooling` argument renames, `MLMTransformer`, `CLIPModel`, CrossEncoder `max_length` property, trainer `tokenizer=`.
- **Notes vs. code:** "there are no hard deprecations, all existing code should continue to work with warnings at worst" is not accurate; see the hard errors above. "`default_activation_function` -> `activation_fn` (with deprecation warning)" is not new: v5.1.2 already accepted `activation_fn` and warned on `default_activation_function`. The `max_length` deprecation applies to the property only, not to the constructor argument.

### v5.4.1

Released 2026-04-14 · [release notes](release-notes/v5.4.1.md)

Patch release. Restores numpy string arrays as batches in `encode` (broken in v5.4.0) and lets `CrossEncoder.predict` take numpy pair arrays. One silent change: `Dense` modules only load `torch.*` activation functions unless the model is loaded with `trust_remote_code=True`, falling back to `Tanh` otherwise ([details](#dense-only-loads-torch-activation-functions-unless-the-model-is-trusted-v541)).

- **Notes vs. code:** the notes say numpy string arrays were "silently treated as a single input" in v5.4.0; on text-only models `SentenceTransformer.encode`, `SparseEncoder.encode` and `CrossEncoder.predict` raised `ValueError` instead.

### v5.5.0

Released 2026-05-12 · [release notes](release-notes/v5.5.0.md)

- Subclasses of the model classes inherit the base `model_type` ([details](#subclasses-of-the-model-classes-inherit-the-base-model_type-v550)); `model.config` is a read-only property until v5.7.0 ([details](#modelconfig-exists-and-is-read-only-until-v570-v550)); Hub errors propagate as `huggingface_hub` exceptions ([details](#hub-errors-propagate-as-huggingface_hub-exceptions-v550)).
- CLS pooling uses the first unmasked token ([details](#cls-pooling-uses-the-first-unmasked-token-v550)); `MSELoss` becomes a subclass of `EmbedDistillLoss`; listwise CrossEncoder losses raise on batches without documents.
- Source builds need `setuptools>=64`.
- Raw causal-LM CrossEncoders (generative rerankers, new in v5.4.0) loaded without `num_labels` default to a `Sigmoid` activation instead of `Identity`.
- Additive: `EmbedDistillLoss`, `ADRMSELoss`, per-call `processing_kwargs`, CrossEncoder `Dense` score heads.
- **Notes vs. code:** the CLS change for `include_prompt=False` with prompts, and the listwise-loss change, are not mentioned.

### v5.5.1

Released 2026-05-20 · [release notes](release-notes/v5.5.1.md)

No breaking changes. Patch release fixing single-key multimodal dicts (`{"image": x}`, `{"text": s}`) being treated as a combined modality, which also makes `encode([{"text": ...}])` work again as in v5.1.2 (broken in v5.4.0 to v5.5.0).

### v5.6.0

Released 2026-06-16 · [release notes](release-notes/v5.6.0.md)

- `FutureWarning` when loading custom code from a local model directory without `trust_remote_code=True` ([details](#custom-module-classes-require-trust_remote_codetrue-including-local-directories-v560-warning-v600-error)).
- Truncated chat-template inputs keep the template suffix ([details](#truncated-chat-template-inputs-keep-the-template-suffix-v560)); `device_map` takes precedence over `device` ([details](#device_map-controls-placement-and-device-is-ignored-when-both-are-given-v560)); `old_fit(use_amp=True)` autocasts on CPU and MPS.
- TSDAE with the default `tie_encoder_decoder=True` works on transformers 5 again; GIST and hard-negative relative margins work for negative similarities; the GIST multi-GPU positive mask is fixed; `faiss_batch_size <= 0` raises.
- PEP 639 license metadata; source builds need `setuptools>=77.0.3`. MPS crash fixes.
- **Notes vs. code:** the notes say the TSDAE `RuntimeError` was raised by v5.5; it is in the code from v5.3.0. The `old_fit` change is described as an MPS fix to `model.fit`; the changed code is `old_fit`, and CPU training is affected too.

### v5.6.1

Released 2026-07-23 · [release notes](release-notes/v5.6.1.md)

No breaking API changes. Patch release fixing position IDs of flattened flash-attention inputs for RoBERTa-family models, which corrects (and therefore changes) their embeddings under transformers 5 with flash attention ([details](#flash-attention-2-inputs-are-flattened-and-roberta-family-position-ids-were-wrong-until-v561-v540)).

- **Notes vs. code:** the notes say the bug existed "since v5.5.0"; the same code is in v5.4.0 and v5.4.1.

### v5.7.0

Released 2026-08-06 · [release notes](release-notes/v5.7.0.md)

- int8/uint8 quantization clips and floors, so int8 output is not bit-identical with earlier releases; binary packing fixed ([details](#int8uint8-quantization-saturates-and-floors-binary-quantization-packs-per-embedding-v570)).
- `AdaptiveLayerLoss`/`Matryoshka2dLoss` default `layer_weighting="uniform"` and a detached KL teacher ([details](#adaptivelayerloss-and-matryoshka2dloss-train-different-layers-with-different-weights-v520-v570)).
- Cached losses rebuilt on a shared GradCache engine, with gradient fixes and `mini_batch_num_tokens` ([details](#cached-losses-share-a-new-gradcache-engine-v570)); `MegaBatchMarginLoss` trains on the whole batch.
- `model.compile()` takes effect and forward hooks fire ([details](#modelcompile-takes-effect-and-forward-hooks-fire-during-inference-v570)).
- `FutureWarning` for third-party module classes from Hub models; `NotADirectoryError` for file paths; `model.config` can be assigned again.
- Evaluators: `BinaryClassificationEvaluator` Euclidean/Manhattan metrics and CSV fixed; canonical RRF.
- `mine_hard_negatives` validation, small-corpus and cache-key fixes; `semantic_search_faiss` validates precision.
- `ListMLELoss`/`PListMLELoss` padding, `SparseCoSENTLoss` default similarity, trainer `get_eval_dataloader` names, `tokenizers>=0.19`.
- New `ValueError`s: `max_active_dims <= 0` in `SparseEncoder.encode`, chat templates that cannot carry a query/document pair, mixed audio sampling rates or video metadata in one batch.
- Regression on v5.7.0 only: `Pooling(include_prompt=False)` writes its prompt-masked `attention_mask` back into the features, so `AdaptiveLayerLoss`/`Matryoshka2dLoss` training with prompts computes a different loss (0.920 vs. 0.752 on v5.1.2 and v6.0.0 in a seeded test); fixed in v6.0.0.
- **Notes vs. code:** #3879 is listed but first shipped in v5.6.1. The notes say `MegaBatchMarginLoss` used to ignore a third column silently; it already raised. `layer_weighting="linear"` restores only the v5.2.0 to v5.6.1 layer weights (the KL teacher is now detached), and nothing reproduces v5.1.2.

### v6.0.0

Released 2026-08-18 · [release notes](release-notes/v6.0.0.md)

Major release.

- Requires transformers v5, huggingface-hub v1 and torch 2.2 ([details](#v600-requires-transformers-v5-huggingface-hub-v1-and-torch-22-v600)); the `onnx` extra cannot currently be installed ([details](#onnx--openvino-extras-now-point-to-optimum-onnx-and-optimum-intel-v520-the-onnx-extra-cannot-be-installed-with-v6)).
- Custom module classes require `trust_remote_code=True`, including local directories ([details](#custom-module-classes-require-trust_remote_codetrue-including-local-directories-v560-warning-v600-error)); trainer reloads exempt only `Module` subclasses ([details](#trainer-checkpoint-reload-of-custom-modules-v600)).
- Scores computed in float32 for half-precision inputs ([details](#scores-are-computed-in-float32-for-half-precision-embeddings-v600)); CrossEncoder activation in float32; `rank()` returns Python floats ([details](#rank-returns-python-floats-v600)).
- `similarity` becomes a method; a bare list of chat messages is one conversation; multi-process int8 quantizes once; `quantize_embeddings` returns lists for lists of 2-D arrays; `Pooling(include_prompt=False)` leaves `attention_mask` intact.
- `Normalize` moves to `sentence_transformers.base.modules`; models can declare required package versions.
- `TripletEvaluator` uses `encode_query`/`encode_document` and validates arguments; IR evaluator tie-breaking; `DistillKLDivLoss` validation; MNRL rejects `scale=nan`; merged forward pass for candidate columns; `NO_DUPLICATES_HASHED` works with xxhash 4.
- New `MultiVectorEncoder`.
- v6.0.0 removed no API that was deprecated during 5.x: every deprecated form in this guide that still worked on v5.7.0 still works in v6.1.0. (`SparseEncoder.encode_multi_process`, deprecated in v5.0, was already removed in v5.4.0.)
- **Notes vs. code:** "cross-family conversion no longer inherits inference settings" is not a change from v5.7.0 (only the log level changed). Checkpoint reloading keeps working without the flag only for `Module` subclasses. The v5.6/v5.7 warnings did not cover classes from installed packages referenced by local models, which now raise. The migration guide omits the `typing_extensions>=4.10.0` and `optimum-intel[openvino]>=2.0.0` floors.

### v6.0.1

Released 2026-08-31 · [release notes](release-notes/v6.0.1.md)

Patch release with two behavior changes. `MultiVectorEncoder` combines PyLate `[Q]`/`[D]` prefixes with saved prompts, which changes embeddings for `lightonai/ColBERT-Zero`, `ColBERT-Zero-supervised` and `ColBERT-Zero-unsupervised` ([details](#pylate-query--document-prefixes-are-combined-with-saved-prompts-v601)). `Router` passes `task` to the routed module, which makes per-task settings work and breaks custom routed modules without `**kwargs` ([details](#router-passes-task-to-the-routed-module-v601)). Also documentation fixes (README minimum versions, model tables).

- **Notes vs. code:** the notes don't mention that an explicit empty prompt now disables the legacy ColBERT markers.

### v6.1.0

Released 2026-09-18 · [release notes](release-notes/v6.1.0.md)

Minor release with one silent behavior change for chat-template multimodal inputs: content follows each input's own key order, lists under a modality key become separate items, and a list under `"video"` is one video only if every item is recognized as an image ([details](#content-order-lists-and-video-frames-in-chat-template-inputs-v610)). Also fixes `load_audio_from_video` routing. Everything else is docstrings, type annotations and documentation. No dependency, training-argument, loss or evaluator changes.

## Quick Reference: most likely breakage points when upgrading from v5.1.2

Roughly in order of how many v5.1.2 programs they affect.

| # | What you'll see | Cause | Version | Fix |
|---|---|---|---|---|
| 1 | pip upgrades transformers to 5.x and huggingface-hub to 1.x, or `ResolutionImpossible` against an old pin | New dependency floors | v6.0.0 | [Upgrade the stack](#v600-requires-transformers-v5-huggingface-hub-v1-and-torch-22-v600); training on transformers 5 needs v5.2.3+ |
| 2 | `sentence-transformers[onnx]` installs 5.7.0, or v6 + `[onnx]` is unsatisfiable | `optimum-onnx` caps transformers <4.58 | v6.0.0 | [Stay on 5.7.0 for ONNX](#onnx--openvino-extras-now-point-to-optimum-onnx-and-optimum-intel-v520-the-onnx-extra-cannot-be-installed-with-v6) |
| 3 | `ValueError: The model ... references the module class ..., which is not part of Sentence Transformers.` | Custom module classes need trust | v6.0.0 | [`trust_remote_code=True`](#custom-module-classes-require-trust_remote_codetrue-including-local-directories-v560-warning-v600-error) |
| 4 | `TypeError: SentenceTransformer.__init__() takes from 1 to 2 positional arguments ...` (also `SparseEncoder`, `Transformer`) | Keyword-only constructors | v5.4.0 | [Use keywords](#constructor-arguments-after-the-model-name-are-keyword-only-v540) |
| 5 | `ValueError: Prompt name '64' not found ...` from `predict`, `rank` or a CrossEncoder evaluator | `prompt_name` inserted before `batch_size` | v5.4.0 | [Use keywords](#predict-and-rank-take-prompt_name-and-prompt-before-batch_size-v540) |
| 6 | `ValueError: Unsupported input type: Series ...` | pandas and non-string inputs rejected | v5.4.0 | [Pass a list of `str`](#encode-rejects-pandas-series-and-non-string-items-v540) |
| 7 | Different embeddings for the same text, no error | Whitespace no longer stripped (BPE tokenizers); `model_max_length` now truncates; CLS pooling; raw LLM checkpoints get last-token pooling and a chat template | v5.4.0, v5.5.0 | [Whitespace](#inputs-are-no-longer-stripped-of-surrounding-whitespace-v540), [truncation](#max_seq_length-now-follows-tokenizermodel_max_length-v540), [CLS](#cls-pooling-uses-the-first-unmasked-token-v550), [raw LLMs](#raw-causal-lm-checkpoints-get-last-token-pooling-and-a-chat-template-v540) |
| 8 | Different embeddings from a model with a `Dense` layer, plus a "not trusted, falling back ... (Tanh)" warning | Non-`torch` activations refused | v5.4.1 | [`trust_remote_code=True`](#dense-only-loads-torch-activation-functions-unless-the-model-is-trusted-v541) |
| 9 | `AttributeError` when setting `model.tokenizer`; swapped `auto_model` has no effect | Read-only properties | v5.4.0 | [Set `model[0].processor` / `model[0].model`](#modeltokenizer-can-no-longer-be-assigned-and-assigning-auto_model-is-silently-ignored-v540) |
| 10 | `ModuleNotFoundError: No module named 'sentence_transformers.base'` (server on v5.3 or older) or `... 'sentence_transformers.base.modules.normalize'` (v5.4 to v5.7 loading a v6 save) | New save format | v5.4.0, v6.0.0 | [Pin the same version everywhere](#models-saved-by-v540-or-later-cannot-be-loaded-by-v53x-or-earlier-v540) |
| 11 | CrossEncoder `set_activation_fn`, `model(**tokens)`, `save(max_shard_size=...)` or `push_to_hub(tags=...)` fail | Modular CrossEncoder | v5.4.0 | [CrossEncoder](#crossencoder) |
| 12 | `rank()` scores are floats; `.item()` fails | Python floats | v6.0.0 | [Use the value directly](#rank-returns-python-floats-v600) |
| 13 | Training results differ with the same script | `TripletLoss` default distance fixed, `AdaptiveLayerLoss` layers and weights, `GroupByLabelBatchSampler`, cached-loss gradient fixes, merged forward pass | v5.2.0 – v6.0.0 | [Losses](#losses), [Training](#training) |
| 14 | Stored int8 indexes no longer match new queries | int8 clip-and-floor; multi-process quantization | v5.7.0, v6.0.0 | [Re-quantize](#int8uint8-quantization-saturates-and-floors-binary-quantization-packs-per-embedding-v570) |
| 15 | Similarity scores are float32 for fp16/bf16 embeddings; top-k ties resolve differently | Higher-precision scoring | v6.0.0 | [Cast back if needed](#scores-are-computed-in-float32-for-half-precision-embeddings-v600) |
| 16 | `ModuleNotFoundError` for `sentence_transformers.fit_mixin` or `model_card_templates`; `DeprecationWarning` for `sentence_transformers.models` | Package restructure | v5.4.0 | [New import paths](#import-paths-moved-old-paths-still-work-with-a-deprecationwarning-v540) |
| 17 | NanoBEIR evaluator fails to load data | Positional `dataset_id`; new data repository | v5.2.0 | [Use keywords; re-download](#nanobeir-evaluators-take-a-new-second-parameter-and-load-from-a-single-dataset-repository-v520) |
| 18 | `ValueError: Column 'score' doesn't exist.` (or `KeyError: 'score'` on row access) after `mine_hard_negatives(..., output_format="n-tuple-scores")` | Column renamed `scores` | v5.2.0 | [Use `scores`](#mine_hard_negatives-n-tuple-scores-is-deprecated-and-its-column-is-renamed-v520) |
| 19 | `load_best_model_at_end` silently keeps the last checkpoint | Trainer reload refuses plain `nn.Module` custom modules | v6.0.0 | [Trust or subclass `Module`](#trainer-checkpoint-reload-of-custom-modules-v600) |
| 20 | `model.compile()` now fails or is slow on the first `encode()` | Compilation takes effect | v5.7.0 | [Drop or configure it](#modelcompile-takes-effect-and-forward-hooks-fire-during-inference-v570) |
