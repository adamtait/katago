# 005 — AlphaGo Reproduction: GCP Build Plan (Design B)

**Audience:** Claude Code, executing with an authenticated `gcloud` and `docker`.
**Source plan:** `alphago-reproduction` wiki (Phases 0–6). This document is the *infrastructure* layer only. The AlphaGo-specific research code (48-plane encoder, rollout features, REINFORCE loop, MCTS λ-blend) lives in your KataGo fork (`alphago-repro` branch) and is referenced here at clearly marked `# RESEARCH HOOK` points — do **not** synthesize it.

**Design:** Decoupled state (GCS) + ephemeral Spot GPU compute, auto-resume on preemption via size-1 MIGs. No GKE. Self-play uses a Spot CPU worker fleet (regional MIG) feeding one Spot GPU inference VM — the one piece borrowed from a distributed design, without the Kubernetes tax.

## How to execute

Run the numbered steps in order. Each step is idempotent (safe to re-run). **Stop at every `GATE:`** and report the metric before proceeding — these map to the plan's validation targets. Never leave a GPU MIG at size > 0 across a `GATE`.

---

## 0. Config

Set once per shell. Fill `PROJECT_ID` and `WANDB_KEY`.

```bash
export PROJECT_ID=alphago-repro              # set me
export REGION=us-central1
export ZONE=us-central1-b                    # overridden by zone probe in Step 1
export BUCKET=gs://${PROJECT_ID}-alphago
export AR_REPO=alphago
export IMAGE=${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/katago-alphago:latest
export SA=alphago-runner@${PROJECT_ID}.iam.gserviceaccount.com
gcloud config set project "$PROJECT_ID"
gcloud config set compute/region "$REGION"
```

**Machine types (GPU is implied by the machine type — do NOT pass `--accelerator` for a2/a3):**

| Role | Machine type | GPU | Provisioning | Notes |
|---|---|---|---|---|
| SL training (Ph1) | `a3-highgpu-1g` | 1×H100 80GB | **Spot only** | 26 vCPU/234GB; sub-8-GPU A3 is Spot/Flex-only |
| Self-play inference (Ph3–4) | `a2-highgpu-1g` | 1×A100 40GB | Spot | tiny net; L4 `g2-standard-8` is a cheaper alt |
| Self-play workers (Ph3–4) | `c3-standard-22` | — | Spot, regional MIG | CPU MCTS rollouts |
| Rollout benchmark (Ph2) | `c3-standard-8` | — | on-demand, short | same CPU class as workers |
| Eval (Ph5–6) | `a3-highgpu-1g` | 1×H100 80GB | Spot, ephemeral | reuse SL image |
| Controller | `e2-small` | — | on-demand, always-on | ~$12/mo; coordination + gating server |
| Data prep (Ph0) | `c3-standard-22` | — | Spot, ephemeral | SGF→tensor, no GPU |

---

## 1. Preflight: APIs, quota, zone, IAM, secrets

```bash
gcloud services enable compute.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com secretmanager.googleapis.com storage.googleapis.com \
  logging.googleapis.com monitoring.googleapis.com

# Service account + roles
gcloud iam service-accounts create alphago-runner --display-name="AlphaGo runner"
for ROLE in roles/storage.objectAdmin roles/artifactregistry.reader \
  roles/secretmanager.secretAccessor roles/logging.logWriter \
  roles/monitoring.metricWriter roles/compute.instanceAdmin.v1; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA}" --role="$ROLE" --condition=None
done

# W&B key into Secret Manager
printf '%s' "$WANDB_KEY" | gcloud secrets create alphago-wandb-key --data-file=- \
  || printf '%s' "$WANDB_KEY" | gcloud secrets versions add alphago-wandb-key --data-file=-
```

**Quota — likely blocks you.** H100 Spot quota is 0 by default. Check and request before anything else:

```bash
gcloud compute regions describe "$REGION" \
  --format="table(quotas.filter('metric:PREEMPTIBLE_NVIDIA_H100_GPUS'))"
# If 0: request "Preemptible NVIDIA H100 GPUs" >= 1 in $REGION via the console quotas page,
# and "Preemptible NVIDIA A100 GPUs" >= 1, and C3 CPUs >= ~200 for the worker fleet.
```

**Zone probe — pick a zone that actually offers H100:**

```bash
gcloud compute accelerator-types list --filter="name=nvidia-h100-80gb" \
  --format="value(zone)" | grep "^${REGION}" | head
# export ZONE=<one of the above>.  Spot capacity still varies; Step 5 retries across zones.
```

`GATE: H100 Spot quota >= 1 and a valid ZONE chosen.`

---

## 2. State layer: GCS + Artifact Registry

```bash
gcloud storage buckets create "$BUCKET" --location="$REGION" --uniform-bucket-level-access
# Layout (created lazily by jobs, shown for reference):
#   kgs-raw/            raw SGF archives
#   tensors/            48-plane encoded train/val shards
#   checkpoints/sl/     SL policy net  (Phase 1)
#   checkpoints/rl/     RL policy pool (Phase 3)
#   checkpoints/value/  value net      (Phase 4)
#   selfplay/           generated positions + game records
#   rollout/            linear rollout policy weights (Phase 2)
#   eval/               gauntlet results, SGFs
#   _control/           gating server state, "latest" pointers, run locks

gcloud artifacts repositories create "$AR_REPO" --repository-format=docker --location="$REGION"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
```

---

## 3. Container image

One image runs every phase; behavior is selected by the `PHASE` env var. Drivers come from the host (Deep Learning VM image), so the container only needs CUDA userspace + Python + the KataGo fork.

**`Dockerfile`** (build context = your `alphago-repro` fork root):

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git cmake build-essential libzip-dev zlib1g-dev \
    libeigen3-dev curl && rm -rf /var/lib/apt/lists/*
# KataGo fork (this repo) — builds the engine + provides train.py to adapt
WORKDIR /opt/alphago
COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt \
    && mkdir -p cpp/build && cd cpp/build \
    && cmake .. -DUSE_BACKEND=CUDA && make -j
COPY infra/run_phase.sh /usr/local/bin/run_phase.sh
RUN chmod +x /usr/local/bin/run_phase.sh
ENTRYPOINT ["/usr/local/bin/run_phase.sh"]
```

**`infra/run_phase.sh`** — entrypoint. Resume-from-GCS + periodic checkpoint + SIGTERM flush are the spot-safety contract; honor them in every training mode.

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${PHASE:?}" "${BUCKET:?}"
export WANDB_API_KEY="$(gcloud secrets versions access latest --secret=alphago-wandb-key)"
WORK=/work; mkdir -p "$WORK"; cd /opt/alphago

flush() {  # called on Spot preemption (~30s window) and on normal checkpoint ticks
  [ -d "$WORK/ckpt" ] && gcloud storage rsync -r "$WORK/ckpt" "${BUCKET}/checkpoints/${PHASE}/" || true
}
trap flush TERM INT

resume() { gcloud storage rsync -r "${BUCKET}/checkpoints/${PHASE}/" "$WORK/ckpt/" 2>/dev/null || true; }

case "$PHASE" in
  sl)        # Phase 1 — H100. RESEARCH HOOK: train.py is your adapted KataGo trainer.
    resume
    python3 python/train.py --config configs/alphago_sl.cfg \
      --data "${BUCKET}/tensors/" --out "$WORK/ckpt" \
      --ckpt-every-steps 2000 --wandb-project alphago-sl &
    TPID=$!; while kill -0 $TPID 2>/dev/null; do sleep 120; flush; done; wait $TPID; flush ;;
  selfplay-worker)   # Phase 3–4 — CPU MCTS. Pulls latest RL ckpt, hits inference VM.
    INFER=${INFER_HOST:?}; resume
    ./cpp/build/katago selfplay -config configs/alphago_selfplay.cfg \
      -nn-host "$INFER" -output-dir "$WORK/games"
    gcloud storage rsync -r "$WORK/games" "${BUCKET}/selfplay/" ;;
  inference)         # Phase 3–4 — A100 serves the policy net to the worker fleet.
    resume
    ./cpp/build/katago gtp -config configs/alphago_infer.cfg -model "$WORK/ckpt/latest.bin.gz" ;;
  eval)              # Phase 5–6 — gauntlet vs GNU Go / Pachi. RESEARCH HOOK: gauntlet.py.
    resume
    python3 python/gauntlet.py --model "$WORK/ckpt/latest.bin.gz" \
      --opponents gnugo,pachi10k --out "${BUCKET}/eval/$(date +%s)/" ;;
  *) echo "unknown PHASE=$PHASE" >&2; exit 1 ;;
esac
```

Build and push (Cloud Build keeps the image build off your machine):

```bash
gcloud builds submit --tag "$IMAGE" .
```

---

## 4. Controller VM (always-on)

Hosts the self-play **gating server** (checkpoint-pool arbiter for Phase 3 — sampling opponents from previous checkpoints is the stability trick the plan flags), drives MIG lifecycle, relays logs. Cheap and persistent.

```bash
gcloud compute instances create alphago-controller \
  --zone="$ZONE" --machine-type=e2-small \
  --image-family=common-cpu --image-project=deeplearning-platform-release \
  --service-account="$SA" --scopes=cloud-platform \
  --metadata=BUCKET="$BUCKET",IMAGE="$IMAGE"
# Then: ssh in, `docker pull $IMAGE`, run the gating server bound to ${BUCKET}/_control/.
# RESEARCH HOOK: gating server policy = your checkpoint-pool sampler.
```

---

## 5. Phase 0 — Data prep (ephemeral Spot CPU)

Downloads KGS, filters to 6+ dan (~160k games), runs the SGF→48-plane pipeline, writes sharded tensors to GCS. No GPU. Self-deletes on completion.

```bash
gcloud compute instances create alphago-dataprep \
  --zone="$ZONE" --machine-type=c3-standard-22 \
  --provisioning-model=SPOT --instance-termination-action=DELETE \
  --image-family=common-cpu --image-project=deeplearning-platform-release \
  --boot-disk-size=400GB --service-account="$SA" --scopes=cloud-platform \
  --metadata=BUCKET="$BUCKET",IMAGE="$IMAGE",startup-script='#!/bin/bash
docker run --rm -e PHASE=dataprep -e BUCKET='"$BUCKET"' '"$IMAGE"' \
  || true; gcloud compute instances delete $(hostname) --zone='"$ZONE"' --quiet'
# RESEARCH HOOK: PHASE=dataprep branch = your SGF parser + 48-plane encoder (Nature ED Table 2).
```

`GATE: tensor shard count and a spot-check decode match the encoder unit tests. ~29M positions in ${BUCKET}/tensors/.`

---

## 6. Phase 1 — SL policy on H100 (Spot, size-1 MIG, auto-resume)

A managed instance group of size 1 is the resume primitive: on preemption the MIG recreates the VM, the startup script re-pulls the image, and `run_phase.sh` resumes from the last GCS checkpoint. You manage training by scaling the MIG to 1 (start) or 0 (stop).

```bash
gcloud compute instance-templates create alphago-sl-tmpl \
  --machine-type=a3-highgpu-1g \
  --provisioning-model=SPOT --instance-termination-action=DELETE \
  --maintenance-policy=TERMINATE \
  --image-family=common-cu123 --image-project=deeplearning-platform-release \
  --boot-disk-size=300GB --boot-disk-type=pd-ssd \
  --service-account="$SA" --scopes=cloud-platform \
  --metadata=BUCKET="$BUCKET",IMAGE="$IMAGE",startup-script='#!/bin/bash
docker pull '"$IMAGE"'
docker run --rm --gpus all --name sl \
  -e PHASE=sl -e BUCKET='"$BUCKET"' '"$IMAGE"'' \
  --shutdown-script='#!/bin/bash
docker kill -s TERM sl; sleep 25'   # propagate Spot preemption to the flush() trap

gcloud compute instance-groups managed create alphago-sl-mig \
  --template=alphago-sl-tmpl --size=1 --zone="$ZONE"
# If creation fails with capacity exhaustion, retry --zone across the Step 1 zone list.

# Stop training:  gcloud compute instance-groups managed resize alphago-sl-mig --size=0 --zone="$ZONE"
```

`GATE: SL top-1 move accuracy >= 50% on held-out test (paper 57.0%). Then resize MIG to 0.`

---

## 7. Phase 2 — Rollout policy + speed benchmark (CPU)

```bash
gcloud compute instances create alphago-rollout \
  --zone="$ZONE" --machine-type=c3-standard-8 \
  --image-family=common-cpu --image-project=deeplearning-platform-release \
  --service-account="$SA" --scopes=cloud-platform \
  --metadata=BUCKET="$BUCKET",IMAGE="$IMAGE"
# ssh + docker run -e PHASE=rollout. RESEARCH HOOK: linear softmax over hand-coded patterns (Table S1).
```

`GATE: rollout >= 1000 positions/sec on c3-standard-8 (≈1000× ConvNet). Delete the VM.`

---

## 8. Phases 3–4 — Self-play fleet (Spot CPU workers → Spot A100 inference)

Inference GPU VM first, then the regional worker MIG (regional spreads Spot requests across zones for better fill). Workers read the current RL checkpoint pool via the controller's gating server and write games to GCS; the trainer (run on the inference VM or a second size-1 H100 MIG, your call) consumes them and emits new checkpoints. This is the CPU/GPU decoupling that keeps the GPU fed without Kubernetes.

```bash
# Inference server (A100, size-1 MIG so it self-heals on preemption)
gcloud compute instance-templates create alphago-infer-tmpl \
  --machine-type=a2-highgpu-1g \
  --provisioning-model=SPOT --instance-termination-action=DELETE --maintenance-policy=TERMINATE \
  --image-family=common-cu123 --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB --service-account="$SA" --scopes=cloud-platform \
  --metadata=BUCKET="$BUCKET",IMAGE="$IMAGE",startup-script='#!/bin/bash
docker pull '"$IMAGE"'; docker run --rm --gpus all -p 8080:8080 \
  -e PHASE=inference -e BUCKET='"$BUCKET"' '"$IMAGE"''
gcloud compute instance-groups managed create alphago-infer-mig \
  --template=alphago-infer-tmpl --size=1 --zone="$ZONE"
export INFER_HOST=$(gcloud compute instances list --filter="name~alphago-infer" \
  --format="value(networkInterfaces[0].networkIP)")

# Worker fleet (regional Spot MIG; scale size to your quota / pace)
gcloud compute instance-templates create alphago-selfplay-tmpl \
  --machine-type=c3-standard-22 \
  --provisioning-model=SPOT --instance-termination-action=DELETE \
  --image-family=common-cpu --image-project=deeplearning-platform-release \
  --service-account="$SA" --scopes=cloud-platform \
  --metadata=BUCKET="$BUCKET",IMAGE="$IMAGE",INFER_HOST="$INFER_HOST",startup-script='#!/bin/bash
docker pull '"$IMAGE"'; docker run --rm -e PHASE=selfplay-worker \
  -e BUCKET='"$BUCKET"' -e INFER_HOST='"$INFER_HOST"' '"$IMAGE"''
gcloud compute instance-groups managed create alphago-selfplay-mig \
  --template=alphago-selfplay-tmpl --size=8 --region="$REGION"

# Throttle / stop: resize alphago-selfplay-mig and alphago-infer-mig to 0.
```

RESEARCH HOOKS: REINFORCE update (reward ±1) and the checkpoint-pool opponent sampler (Phase 3); one-position-per-game sampling for the value net (Phase 4).

`GATE (Ph3): RL vs SL head-to-head win rate > 70% (paper > 80%).`
`GATE (Ph4): value-net MSE beats rollout-only on held-out games. Scale both MIGs to 0.`

---

## 9. Phases 5–6 — MCTS integration + eval (ephemeral Spot H100)

Reuses the SL template image; runs the gauntlet then self-deletes.

```bash
gcloud compute instances create alphago-eval \
  --zone="$ZONE" --machine-type=a3-highgpu-1g \
  --provisioning-model=SPOT --instance-termination-action=DELETE --maintenance-policy=TERMINATE \
  --image-family=common-cu123 --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB --service-account="$SA" --scopes=cloud-platform \
  --metadata=BUCKET="$BUCKET",IMAGE="$IMAGE",startup-script='#!/bin/bash
docker pull '"$IMAGE"'; docker run --rm --gpus all -e PHASE=eval -e BUCKET='"$BUCKET"' '"$IMAGE"'
gcloud compute instances delete $(hostname) --zone='"$ZONE"' --quiet'
```

`GATE: 100% vs GNU Go; consistent wins vs Pachi (10k sims).`

---

## 10. Cost guardrails (do this in Step 1, not last)

```bash
# Budget alert (needs your billing account id)
BILLING=$(gcloud billing projects describe "$PROJECT_ID" --format="value(billingAccountName)" | sed 's#.*/##')
gcloud billing budgets create --billing-account="$BILLING" \
  --display-name=alphago --budget-amount=1000USD \
  --threshold-rule=percent=0.5 --threshold-rule=percent=0.9
```

Invariants that keep this design cheap:
- Every GPU VM is **Spot** with `--instance-termination-action=DELETE` — nothing GPU-billed survives preemption or a forgotten shell.
- The only things that persist between phases are the `e2-small` controller (~$12/mo), GCS (a few hundred GB, ~$5–15/mo), and the AR image. All bursty compute is MIG-driven; **scale every GPU MIG to 0 at each GATE.**
- Realistic envelope at the plan's compute scale: ~150–250 H100 Spot-hours (SL + eval) at ≈$2.25/GPU-hr + ~A100/C3 self-play ≈ **$700–1,200 total**, dominated by self-play worker-hours, not the H100.

## Teardown

```bash
for MIG in alphago-sl-mig alphago-infer-mig; do gcloud compute instance-groups managed delete $MIG --zone="$ZONE" --quiet; done
gcloud compute instance-groups managed delete alphago-selfplay-mig --region="$REGION" --quiet
gcloud compute instances delete alphago-controller --zone="$ZONE" --quiet
# Keep $BUCKET (checkpoints + results) until the run is written up.
```

---

### Open infra decisions (flag, don't guess)
- **Trainer placement for Phases 3–4:** co-locate on the A100 inference VM, or a second size-1 H100 MIG? Default to co-location until GPU contention shows in W&B.
- **Flex-start vs Spot for the H100** if Spot capacity is thin: Flex-start (DWS) gives queued, non-preemptible blocks — swap `--provisioning-model=SPOT` for the Flex-start resize-request flow on the MIG if preemption churn stalls SL.
- **L4 inference** (`g2-standard-8`) is ~3× cheaper than A100 and ample for a 13-layer ConvNet; switch if inference is not the self-play bottleneck.
