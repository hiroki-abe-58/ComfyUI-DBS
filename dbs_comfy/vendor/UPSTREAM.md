# Vendored code: Dynamic Band Splitting (DBS)

| | |
| --- | --- |
| Upstream | https://github.com/JusperLee/Dynamic-Band-Splitting |
| Revision | `ff526463a58ee0278e7bd05c4b8bce112b743e34` (2026-10-15) |
| License | Apache-2.0 (`LICENSE`, SHA-256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`); parts adapted from MIT-licensed projects, notices in `THIRD_PARTY.md` (SHA-256 `0e9fcb686f497d89f01878ba3aa335595c03b387ff409533b6124a1a11de1f79`) |
| Paper | Bingzhe Zhang, Kai Li, Xiaolin Hu, "Learning Where to Split: Dynamic Band Splitting for Audio Source Separation", arXiv:2610.14993 |

Only the files that inference with the two released DnR v2 models needs are vendored. `MANIFEST.sha256` lists every vendored
file with its SHA-256; `scripts/vendor_sync.py` rebuilds the tree from the revision above and `tests/test_vendor.py` checks it in CI.

## Copied unchanged (byte for byte)

`dbs/models/bsrnn.py`, `dbs/models/tiger.py`, `dbs/modules/__init__.py`, `dbs/modules/band_split.py`, `dbs/modules/dbs.py`,
`dbs/modules/stft.py`, `dbs/engine/inference.py`, `LICENSE`, `THIRD_PARTY.md`.

## Replaced by this pack's own files (shims)

| File | Upstream | Here |
| --- | --- | --- |
| `dbs/__init__.py` | docstring and `__version__` | the same plus a note that this is the vendored subset |
| `dbs/engine/__init__.py` | lazy exports of the training engine (PyTorch Lightning, OmegaConf) | exports `separate` only |
| `dbs/models/__init__.py` | the four models, including the RoFormer variants (need rotary-embedding-torch) | DBS-BSRNN and DBS-TIGER only |
| `dbs/hub.py` | `huggingface_hub.PyTorchModelHubMixin` and model-card metadata | an empty mixin: nothing in the vendored code can download |

Upstream loads a released model with `from_pretrained(<folder>)`: it builds the class from `config.json` and loads
`model.safetensors` with `safetensors.torch.load_model(..., strict=False)`. `dbs_comfy/runtime.py` does the same with
`strict=True` (the released files load strictly; `tests/test_parity.py` compares the weights with the official loader's).

No upstream code is modified.

## Model files (not in this repository)

`tools/download_model.py` fetches them at pinned revisions and checks the SHA-256 in `tools/model_files.sha256`.

| Model | Repository and revision | `model.safetensors` SHA-256 | License |
| --- | --- | --- | --- |
| DBS-TIGER-DnR (1.0M parameters) | `JusperLee/DBS-TIGER-DnR` `22da7d4cd3ea63f9365181008bff1c0d2ab6fb2b` | `3d8205743f2a6f5a6704f5d4104d7cf5bf7bf02dab2fd728da31f4c69225cadb` | Apache-2.0 (model card) |
| DBS-BSRNN-DnR (4.7M parameters) | `JusperLee/DBS-BSRNN-DnR` `5af343f57f106fbb95efe8a60f1a3999eeb0aef8` | `26290f93e7b501113325e922e7b0c18edca944634e6b9be39856a55204628405` | Apache-2.0 (model card) |

Both were trained from scratch on the DnR v2 training set (Divide and Remaster v2, Petermann et al., zenodo.org/records/6949108,
CC BY 4.0; mixtures built from LibriSpeech, the Free Music Archive and FSD50K).
