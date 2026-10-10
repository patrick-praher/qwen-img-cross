# qwenimg

Local text-to-image generation with **Qwen-Image-2.1**, driven by
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).

One Python file, standard library only — no `pip install`, no PyTorch, no CUDA
toolkit. The backend binary and the model weights are downloaded on first use.

* **Windows, Linux and macOS**, x86-64 and Apple silicon
* **GPU accelerated**: Metal (macOS), CUDA / Vulkan / ROCm (Windows),
  Vulkan / ROCm (Linux), with a CPU fallback
* Reads the prompt **interactively**, from the **command line**, from a
  **file** (one prompt per line) or from **stdin**

> [!NOTE]
> **Tested on macOS (Apple silicon, Metal) and Windows (NVIDIA, CUDA).** The
> Linux paths and the other Windows backends are implemented but have not yet
> been run on real hardware — see
> [Platform status](#platform-status). Reports from either are very welcome.

## Quick start

```sh
git clone https://github.com/patrick-praher/qwen-img-cross
cd qwen-img-cross

python3 qwenimg.py setup                       # download backend + weights (~11 GB)
python3 qwenimg.py --turbo "a red fox curled up in deep snow, soft winter light"
```

Use `--turbo` unless you have a reason not to — see [Speed](#speed). It is the
difference between a few minutes and a quarter of an hour per image.

`setup` is optional — the first generation downloads whatever is missing.
There are thin launchers too: `./qwenimg "..."` on macOS/Linux and
`qwenimg.cmd "..."` on Windows.

## Usage

```sh
qwenimg "a neon city street in the rain"          # single prompt
qwenimg -f prompts.txt                            # one image per line
qwenimg --whole-file -f long-prompt.txt           # the whole file as one prompt
echo "an overgrown greenhouse" | qwenimg          # from stdin
qwenimg                                           # interactive: waits for prompts

qwenimg --turbo "a lighthouse in a storm"         # ~5x faster, 8 steps
qwenimg --size 16:9 --steps 24 --seed 1234 "a lighthouse in a storm"
qwenimg -b 4 "studio portrait of a tabby cat"     # 4 images, one model load
qwenimg --quant q8_0 "maximum quality, if it fits"
qwenimg --variant base "the official Qwen weights instead"
```

Interactive mode keeps settings between prompts:

```
prompt> a red fox in deep snow
prompt> /size 16:9
prompt> /steps 30
prompt> /neg blurry, low quality
prompt> a red fox in deep snow
prompt> /open
prompt> /quit
```

`/help` lists the commands. Each generation reloads the weights, so for long
sessions `qwenimg serve` is faster — it runs the stable-diffusion.cpp web UI
with the model held in memory.

### Commands

| Command | Purpose |
| --- | --- |
| `gen` (default) | generate images |
| `setup` | download the backend and weights, then exit |
| `doctor` | show platform, detected GPU, paths and ggml devices |
| `models` | show which model files are selected and whether they exist |
| `serve` | run the sd.cpp web UI with these models (model stays loaded) |
| `clean` | delete cached downloads (`--all` also drops the weights) |

### Useful options

| Option | Meaning |
| --- | --- |
| `-S, --size` | `WxH`, or a preset: `1:1 16:9 9:16 4:3 3:4 3:2 2:3` |
| `--steps`, `--cfg`, `--sampler`, `-s/--seed`, `-b/--batch` | sampling controls |
| `-n/--negative`, `--negative-file` | negative prompt |
| `--whole-file` | treat the whole prompt file (or stdin) as one multi-line prompt |
| `-o, --outdir` | output directory (default `outputs/`) |
| `--variant` | `uncensored` (default) or `base` |
| `--turbo` | apply the official turbo LoRA: 8 steps at cfg 1.0, ~5x faster |
| `--quant`, `--llm-quant` | quantisation; `auto` sizes it to your hardware |
| `--backend` | sd.cpp device map, e.g. `cpu` or `diffusion=cuda0,vae=cpu` |
| `--offload` | `auto\|on\|off` — keep weights in RAM, stream to VRAM |
| `--mmap` | memory-map weights (faster reloads via the page cache) |
| `--verify` | check every download against its SHA-256 |
| `--dry-run` | print the `sd-cli` command instead of running it |
| `--extra ...` | everything after this is passed to `sd-cli` verbatim |

Defaults can live in `qwenimg.json` next to the script — copy
`qwenimg.json.example`. Precedence: CLI flag → config file → built-in default.

## Speed

Generation cost is driven by *forward passes*, not steps: at `--cfg` above 1.0
the sampler runs a conditional **and** an unconditional pass per step, so 20
steps at cfg 6 is 40 passes. The `--turbo` LoRA is distilled to work at cfg 1.0
in 8 steps, which is 8 passes.

Measured on an Apple M5 (32 GB, Metal, `q6_k`) at 1024x1024:

| Settings | Passes | Time |
| --- | ---: | ---: |
| `--steps 12` (cfg 6.0) | 24 | 543 s |
| `--turbo` (8 steps, cfg 1.0) | 8 | 214 s |

The turbo run was both faster *and* visibly better here, because 12 steps at
cfg 6 is well short of convergence for this model — the full model wants ~20-30
steps, which on this hardware is 15-25 minutes per image. Reach for `--turbo`
for anything interactive, and drop it when you want the undistilled model.

On Windows with an RTX 2080 Ti (11 GB, CUDA, `q4_k_m` picked by `--quant auto`),
`--turbo` at the default 1344x1344 took 76 s end to end: 3 s text encoding,
42 s sampling, 28 s VAE decode. The decode does not fit in 11 GB at full size;
stable-diffusion.cpp logs `vae decode compute failed`, then retries with
spatial tiling and succeeds on its own — that error line is expected.

`--turbo` logs `Only (569 / 665) LoRA tensors have been applied` plus a run of
`unused lora tensor` warnings. That is stable-diffusion.cpp declining to map the
LoRA's fused `img_mlp.gate_up` tensors; the LoRA still works, as the timings and
output above show. The warnings come from the backend, so they are passed
through rather than hidden.

Other levers: a smaller `--quant`, fewer `--steps`, a smaller `--size`, and
`--mmap` (which lets the OS page cache survive between runs, cutting load time
on repeated generations). For many prompts in a row, `qwenimg serve` loads the
model once instead of per image.

## Models

| Part | `uncensored` (default) | `base` |
| --- | --- | --- |
| Diffusion | [`abenzerps/Qwen-Image-2.1-Uncensored-GGUF`](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF) | [`leejet/Qwen-Image-2.1-GGUF`](https://huggingface.co/leejet/Qwen-Image-2.1-GGUF) |
| Text encoder | [`pottokao/…-Text-Encoder-Heretic-GGUF`](https://huggingface.co/pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF) (abliterated Qwen3-VL-8B) | [`Qwen/Qwen3-VL-8B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` | same file, from `Comfy-Org/Qwen-Image-2.1` |

`--turbo` additionally fetches `loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors`
(0.9 GB) from `Comfy-Org/Qwen-Image-2.1`; it applies to either variant.

Two notes on "uncensored", so the naming is not misleading:

* The `UC` GGUFs are a community repack. Its own model card says the files are
  quantisations of the upstream `Qwen/Qwen-Image-2.1` weights, so treat the
  diffusion model as the base model unless you measure a difference. The repo
  also ships a separate `qwen-image-2.1-uncensored-lora.safetensors`.
* The *text encoder* is where prompt refusals actually live, which is why the
  default pairs it with the abliterated ("heretic") Qwen3-VL build. Use
  `--variant base` for the fully official stack.

`--quant auto` picks from accelerator memory: ≥40 GB → `q8_0`, ≥28 → `q6_k`,
≥18 → `q5_k_m`, ≥11 → `q4_k_m`, else `q4_0`. A repo that lacks the exact quant
gets the nearest one on the ladder. Roughly 11 GB total at the default `q6_k`
on a 32 GB machine, or about 9.5 GB at `q4_k_m`.

Image editing works too, since Qwen-Image-2.1 is also an edit model: pass
`-r reference.png` with a prompt describing the change. That additionally
fetches the ~1.2 GB vision projector.

Transparent output is prompt-driven — Qwen recommends wrapping the request:
`This is an RGBA image with transparency. <description>. The image has alpha
channel and the background is transparent.` Keep the output as `.png`.

## Where things go

Generated images land in the working directory; everything downloaded lands in
a cache outside the repo, so multi-gigabyte weights can never be committed by
accident.

| | |
| --- | --- |
| Images | `./outputs/<timestamp>_s<seed>_<slug>.png` |
| Generation log | `./outputs/index.jsonl` (prompt, seed and settings per run) |
| Backend + weights | the model cache, below |

Generation parameters are also embedded in each PNG; read them back with
`sd-cli -M metadata --image out.png`.

### Model cache

| OS | Location |
| --- | --- |
| macOS | `~/Library/Caches/qwenimg` |
| Linux | `$XDG_CACHE_HOME/qwenimg`, else `~/.cache/qwenimg` |
| Windows | `%LOCALAPPDATA%\qwenimg` |

```
qwenimg/
├── models/
│   ├── uncensored/   the diffusion model, VAE and text encoder (~11 GB)
│   └── loras/        the turbo LoRA (0.9 GB), shared by both variants
├── backend/<release>_<flavour>/   sd-cli, sd-server and their library
└── downloads/        the backend archive it was extracted from (deletable)
```

Weights are keyed by variant and filename, so nothing is overwritten in place:
`--variant base` downloads alongside `uncensored/` rather than replacing it,
and switching quant fetches a second file while keeping the first. That is
convenient for comparing, and it adds up — `qwenimg models` prints the resolved
paths and which files are present, and `qwenimg clean` removes the backend and
archives while keeping the weights (`--all` drops those too, after asking).

To put the cache elsewhere — an external drive, say — set `QWENIMG_HOME`:

```sh
export QWENIMG_HOME=/Volumes/Big/qwenimg
```

Move the existing directory there first and nothing re-downloads: each file is
skipped when it is already present at the expected size (add `--verify` to
check SHA-256 as well).

## Environment variables

| Variable | Effect |
| --- | --- |
| `QWENIMG_HOME` | cache root for binaries and weights |
| `QWENIMG_BACKEND` | force a flavour: `metal cuda rocm vulkan cpu` |
| `QWENIMG_SD_CLI` | use an existing `sd-cli` instead of downloading one |
| `QWENIMG_CONNECTIONS` | parallel download connections (default 4) |
| `HF_TOKEN` | Hugging Face token, for gated or rate-limited repos |

## Platform status

| Platform | Backend | Status |
| --- | --- | --- |
| macOS, Apple silicon | Metal | Verified end to end on an M5 (32 GB) |
| macOS, Intel | Metal / CPU | Not run. The macOS archive is a universal binary (`x86_64 arm64` confirmed with `lipo`), so it should execute; Metal coverage on Intel GPUs is weaker, and `--backend cpu` is the fallback |
| Windows x64, NVIDIA | CUDA | Verified end to end on an RTX 2080 Ti (11 GB), Windows Server 2025, Python 3.12 |
| Windows x64, other | Vulkan / ROCm / CPU | **Not run.** Archive selection and download URLs verified |
| Linux x86-64 | Vulkan / ROCm / CPU | **Not run.** Archive selection and download URLs verified |
| Linux aarch64 | — | No prebuilt archive upstream; build from source (below) |

On Windows/NVIDIA the full path has been exercised: `nvidia-smi` detection,
VRAM-based quant choice, the CUDA archive plus its companion `cudart` DLL
archive (the binary finds the DLLs; `qwenimg doctor` lists `CUDA0`), the
`qwenimg.cmd` launcher, and a turbo generation. Prompt files and stdin are read
whatever encoding Windows tools produce — UTF-8 with or without a BOM (Notepad,
`Set-Content`) and UTF-16 (Windows PowerShell's `>`/`Out-File`).

What has been checked for the remaining platforms: backend archive selection
for all nine platform/GPU combinations resolves to the right release asset,
both against the live GitHub API and against the pinned fallback names, and
every one of those asset URLs returns a valid size. What has **not** been
checked is anything requiring that hardware — that the binaries launch, that
GPU detection reads the right thing from `rocminfo`/`Win32_VideoController`, or
that Vulkan on an NVIDIA card performs reasonably.

The most likely place for a first bug on those platforms is GPU detection
(`qwenimg doctor` prints what it found). It has escape hatches:
`QWENIMG_BACKEND=vulkan` to force a flavour and
`QWENIMG_SD_CLI=/path/to/sd-cli` to bypass the download entirely.

## Requirements

Python 3.8+ and about 15 GB of free disk. Everything else is fetched.

Where no prebuilt backend exists (see [Platform status](#platform-status)),
build stable-diffusion.cpp yourself and point the tool at it:

```sh
git clone --recursive https://github.com/leejet/stable-diffusion.cpp
cmake -B build stable-diffusion.cpp -DCMAKE_BUILD_TYPE=Release   # add -DSD_CUDA=ON / -DSD_VULKAN=ON / -DSD_METAL=ON
cmake --build build --config Release -j
export QWENIMG_SD_CLI=$PWD/build/bin/sd-cli
```

Linux has no prebuilt CUDA archive upstream, so NVIDIA cards use the Vulkan
build by default. That is usually fine; compile with `-DSD_CUDA=ON` as above if
you want the CUDA path.

## Troubleshooting

* **Out of memory** — drop to `--quant q4_k_m` or `q4_0`, add `--offload on`,
  lower `--size`, or push parts to the CPU with `--backend vae=cpu`.
* **Slow or no GPU** — `qwenimg doctor` prints the detected devices. On Linux a
  Vulkan build needs the Vulkan loader and drivers (`vulkaninfo` should work).
* **Interrupted download** — re-run; Hugging Face transfers resume per 32 MB
  chunk. Add `--verify` to re-check checksums.
* **Wrong backend chosen** — override with `QWENIMG_BACKEND=vulkan` and
  `--force-backend`.

## About the code

Fair warning: this is vibe-coded. Claude (Opus 5) wrote essentially all of it
in one session while I steered; I did not review it line by line. It does the
job — I use it — but do not mistake it for carefully crafted software.

Concretely, what that means if you plan to touch it:

* It is one ~1,650-line module. No package, no separation between the
  downloader, the backend resolver and the CLI beyond section comments.
* **No test suite in the repo.** The downloader, resume logic, platform
  matrix and REPL were genuinely exercised during the session — including
  interrupt/resume against a local range server and a server that ignores
  Range headers — but none of that was committed, so nothing stops a
  regression.
* No CI, no linting config, no type checking, despite the annotations.
* Error handling is uniformly "print a message and exit". Fine for a CLI,
  useless if you want to import any of it.
* Only macOS (Metal) and Windows (CUDA) have been run on real hardware — see
  [Platform status](#platform-status).

The parts I would trust most are the ones that were tested hardest: the
downloader's integrity checks and the release-asset matrix. The parts I would
trust least are the ones no machine here could exercise — GPU detection and
the backends on Linux and on non-NVIDIA Windows machines.

PRs welcome, and so is ripping chunks of it out.

## Licensing

This script is yours to use as you like. The model weights are not: the
Qwen-Image-2.1 weights carry their own licence (`qwen-research` on the
quantised repacks), and the community repacks add their own terms. Check each
Hugging Face repo before any commercial or redistributive use.
