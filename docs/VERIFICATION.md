# Verification

What was checked, how, and the raw numbers. Machine: Windows 11, NVIDIA GeForce RTX 5090 (32 GB), Intel Core Ultra 9 285K (24
threads). Pack environment: ComfyUI v0.38.0 (`6b747c0428c343e1417219641db93a4fb7cb69ae`), Python 3.12.13, torch 2.14.1+cu130.

## Reference: the official package, in a separate environment

- Python 3.11.9 with the exact versions of the upstream `requirements.txt` (torch 2.14.1+cu130, numpy 2.4.6, einops 0.8.2,
  huggingface_hub 2.2.0, safetensors 0.8.0, rotary-embedding-torch 0.9.1, soundfile 0.14.0) and the official package installed
  from the pinned checkout `ff526463` (`pip install --no-deps -e .`). The upstream test suite passes there (17 passed).
- The official usage of the README and the model cards, unchanged:
  `<cls>.from_pretrained(<model folder>).eval()`, `mixture = torch.from_numpy(audio.mean(1)).view(1, 1, -1)`,
  `separate(model, mixture, {"mode": "chunked", "chunk_seconds": 6.0, "hop_seconds": 3.0}, sample_rate=44100)[0]` (DBS-TIGER;
  3.0 / 1.5 s for DBS-BSRNN), run under `torch.inference_mode()`. For the pack's per-channel mode the same call ran on every
  channel.
- Inputs: the three 30 s mixtures of the project page (Tears of Steel 03:10 and 05:35, Sintel 04:55; Blender Foundation, CC BY
  3.0), 44.1 kHz stereo AAC in MP4, decoded once with PyAV into 32-bit float WAV files that both sides read. Short cases cut from
  the first: 10 s (channel mean, mono file), 4.5 s and 2 s (stereo).
- The released weights load with `strict=True` in the pack; their tensors equal those of the official `from_pretrained`
  (`tests/test_parity.py`).

## Pack vs official: CPU, bit for bit

The pack's runtime (`dbs_comfy/runtime.py`, ComfyUI environment) against the official outputs (reference environment): every stem
compared as raw float32 bytes.

| Case | Model | Path | Windows | Result |
| --- | --- | --- | --- | --- |
| 10 s, channel mean | DBS-TIGER | several windows | 3 | identical |
| 10 s, channel mean | DBS-BSRNN | several windows | 6 | identical |
| 2 s, channel mean | DBS-TIGER | one zero-padded window | 1 | identical |
| 4.5 s, channel mean | DBS-TIGER | one unpadded window | 1 | identical |
| 4.5 s, channel mean | DBS-BSRNN | two windows | 2 | identical |
| 2 s, each channel | DBS-BSRNN | per channel | 2 | identical |

6 of 6. The same comparison runs in CI in one process, on synthetic audio, for both models and every path (`tests/test_parity.py`).

## Pack vs official: CUDA

The official code is not deterministic on CUDA: two runs of the same official command on the same file give different bytes
(Tears of Steel 03:10, 30 s: DBS-TIGER max difference 3.5e-4, SNR 77.5 dB between the runs; DBS-BSRNN 3.7e-5, 90.7 dB). So both
sides were run again with PyTorch's deterministic algorithms (`torch.use_deterministic_algorithms(True)`,
`torch.backends.cudnn.deterministic = True`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`); with that, the official output is the same on
every run, and the pack gives it bit for bit:

| Case (30 s) | Model | Mode | Windows | Result |
| --- | --- | --- | --- | --- |
| Tears of Steel 03:10 | DBS-TIGER | channel mean | 9 | identical |
| Tears of Steel 03:10 | DBS-BSRNN | channel mean | 19 | identical |
| Sintel 04:55 | DBS-TIGER | each channel | 18 | identical |
| Tears of Steel 05:35 | DBS-BSRNN | each channel | 38 | identical |

4 of 4. Without the deterministic switch the pack differs from one official run by at most 3.3e-4 (DBS-TIGER) and 4.2e-5
(DBS-BSRNN), the same size as the official run-to-run differences.

The stems add up to the input: the largest difference between music + sfx + speech and the (mono or per-channel) input was 5e-7
in every case above.

## Real ComfyUI queue

**CPU, `scripts/ci_queue_check.py`** (also in CI): a ComfyUI server with the pack copied into `custom_nodes`, synthetic 7 s stereo
soundtrack and a 3 s MPEG-4 video with a stereo AAC track in its input folder, the workflows of `workflows/api` through
`POST /prompt`:

- `separate_soundtrack` (DBS-TIGER, channel mean): the three saved 16-bit FLAC files equal the official package's stems for the
  same file to within half a 16-bit step (the FLAC rounding).
- `remove_dialogue` (each channel, Merge Audio of music + sfx): the saved stereo FLAC equals the official music + sfx per channel,
  and the dialogue FLAC the official speech per channel, both to within half a step.
- `video_remove_dialogue`: Load Video -> Get Video Components -> DBS Separate -> Merge Audio -> Create Video -> Save Video writes a
  3.0 s MP4 with video and audio streams, and the dialogue as FLAC.
- The same job again gives the same audio; DBS-BSRNN gives other audio; `channels` = `stereo` and a model name with a path are
  rejected by ComfyUI's validation.

**GPU (RTX 5090), the 30 s clips through the queue:** with ComfyUI started with `--deterministic`, the saved FLAC equals the
official deterministic CUDA output to within half a step for DBS-TIGER (2 of 2 jobs: channel mean and each channel); for
DBS-BSRNN it is within 1.5 to 2.5 steps (about 80 dB SNR), because ComfyUI's `--deterministic` does not set
`torch.backends.cudnn.deterministic`, which the BSRNN's LSTM layers need for identical results. Without `--deterministic`: 69 to
80 dB SNR against the official default-mode outputs, as expected from the run-to-run differences above.

## Clean install

`git clone` of this repository from GitHub into a fresh ComfyUI v0.38.0 with a fresh Python 3.12.13 environment (torch
2.14.1+cu130 and ComfyUI's requirements); `pip install -r requirements.txt` added no package (`einops` 0.8.2 and `safetensors`
0.8.0 come with ComfyUI's requirements); `tools/download_model.py` downloaded both models from Hugging Face and verified all six
files; `scripts/ci_queue_check.py --keep-pack` against the installed pack passed with the same results as above (the check itself
needs `soundfile`, and `rotary-embedding-torch` for importing the official package; the pack needs neither).

## CI

GitHub Actions, CPU, the real models (fetched and SHA-256 checked by `tools/download_model.py`) and the upstream checkout at the
pinned commit, four jobs: ComfyUI v0.38.0 on Ubuntu 24.04 and Windows (Python 3.12, torch 2.14.1), on Ubuntu with Python 3.10 /
torch 2.8.0, and ComfyUI v0.42.1 on Ubuntu (Python 3.12, torch 2.14.1). Each job checks the vendored code against the upstream
commit (`scripts/vendor_sync.py --check`), runs the 47 tests with every reference required (none may be skipped; the pack is
compared with the official package in the same process) and the real-queue check. All four passed on the first commit.

## Speed and memory

| Model | Device | Input | Time | Speed | Peak GPU memory |
| --- | --- | --- | --- | --- | --- |
| DBS-TIGER | RTX 5090 | 10 min, channel mean | 37.9 s | 15.9x real time | 1.64 GiB |
| DBS-TIGER | RTX 5090 | 10 min, each channel (stereo) | 74.5 s | 16.1x real time per channel | 1.61 GiB |
| DBS-BSRNN | RTX 5090 | 10 min, channel mean | 26.9 s | 22.4x real time | 1.42 GiB |
| DBS-BSRNN | RTX 5090 | 10 min, each channel (stereo) | 53.7 s | 22.4x real time per channel | 1.42 GiB |
| DBS-TIGER | CPU, 24 threads | 60 s, channel mean | 122.5 s | 0.49x real time | - |
| DBS-BSRNN | CPU, 24 threads | 60 s, channel mean | 32.8 s | 1.83x real time | - |

Model loaded, pack runtime in-process. The stems of the whole input are collected on the device until the end (12 bytes per input
sample at 44.1 kHz, about 0.3 GB of the peak for 10 minutes), so the peak grows slowly with the input length.

## Things found on the way

- The official demo backend runs DBS-TIGER on the CPU with one thread and NNPACK off and says the outputs are identical. Here
  that mode gave different bytes and was 3.3 times slower (71.5 s against 21.4 s for 10 s of audio), so the pack uses the README
  usage (PyTorch's default threads).
- The separated clips on the project page are not a reference for bit comparisons: they are AAC re-encodes, and their windows
  (the demo defaults to 3 s windows without overlap) and source audio differ from the README usage. Against our official runs
  they agree to about 7 to 8 dB SNR, without a time offset.

## Not checked

- Separation quality itself: the pack reproduces the official outputs; it does not re-evaluate the models (the model cards give
  SI-SDR on the DnR v2 test set: DBS-TIGER 7.78 / 6.70 / 15.90 dB for music / sfx / speech, DBS-BSRNN 6.48 / 5.73 / 15.01 dB).
- Audio other than film-like mixtures (DnR v2 is speech + music + sound effects; singing is likely to go to "music").
- Linux and macOS outside CI, AMD or Apple GPUs (the pack uses the CPU when ComfyUI's device is not CUDA).
