# Runpod Pod handoff

## Contract

Target: an interactive single-GPU Pod used for preprocessing, training and
batch evaluation. The image preserves the Runpod PyTorch base `/start.sh`; SSH
and web-terminal behavior are inherited rather than reimplemented.

The image is a recipe, not a published artifact. Image build, registry push,
target-GPU execution and Runpod end-to-end behavior remain unverified until the
human data-use prerequisite is complete and cloud spending begins.

## Storage boundary

- Image-local `/opt/headpose`: immutable code and installed package.
- Network-volume `/workspace/headpose-data`: processed shards and manifests.
- Network-volume `/workspace/headpose-artifacts`: checkpoints and reports.
- Container-local `/tmp/headpose-raw`: temporary governed downloads; delete
  only after conversion validation and never persist them to the volume.

Create the network volume before the Pod and place both in the same data
center. Standard storage is sufficient for the pilot; upgrade to a separate
high-performance volume only after measuring an I/O bottleneck.

## Suggested pilot settings

- One GPU, provisionally 16–24 GB VRAM; final type is chosen from a live price
  and stock read at deployment time.
- CUDA host floor: 12.8.
- Network volume: 100 GB standard tier.
- Container disk: 50 GB for source archives and conversion scratch.
- No public application port is required. Expose SSH or the Runpod web
  terminal only through the supported template settings.

## Build and verification

Build for the Runpod host architecture:

```bash
docker build --platform=linux/amd64 -t <registry>/xr-headpose:<revision> .
```

After the Pod starts, initialize the persistent layout and run a useful smoke
test:

```bash
cd /opt/headpose
python cloud/prepare_workspace.py
bash cloud/smoke_check.sh
```

Expected success signal: `headpose cloud smoke check passed`, plus a
`/workspace/headpose-data/manifests/workspace.json` marker showing that no
governed data has yet been accessed.

## Human prerequisite

The researcher must personally accept the applicable BOXRR-23/Berkeley data-use
terms and configure cloud-side access. No environment variable or script may
stand in for that acceptance. Dataset retrieval is deliberately absent from
the image until this prerequisite is confirmed.
