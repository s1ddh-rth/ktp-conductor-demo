# Risk register

What can go wrong with the prototype between now and Monday morning,
and what we'd do about each. Loosely ranked by probability × impact.

A handful of items also discuss risks that would attach to the real
KTP project — these are flagged separately because they show
forward-thinking but aren't action items for the sprint.

---

## Sprint risks (this week)

### R1 — Training fails to converge

**Probability**: medium · **Impact**: high · **Headline**: the
single-largest sprint risk.

**Symptoms**: validation IoU stuck at 0, loss not decreasing, masks
look like random noise.

**Likely causes**:
- Class imbalance not handled (forgot focal loss, or pos_weight off).
- Mask format mismatch (expected 0/1, got 0/255 floats).
- Augmentation breaking the image–mask correspondence.
- Learning rate too high (loss diverges) or too low (loss flat).

**Mitigations**:
- Validate the dataset by visualising 10 random `(image, mask)` pairs
  *before* starting the training run. This catches 80% of failure
  modes in 2 minutes.
- Run a 1-epoch sanity check at full LR on 16 images and confirm loss
  decreases. If it doesn't, fix before kicking off the full run.
- If first run fails: drop to ResNet18 encoder, batch 4, LR 5e-4,
  shorter run. Iteration over correctness.

**Fallback**: if Saturday-evening training is broken, retry Sunday
morning on Colab; the timeline still allows it.

---

### R2 — Live tunnel goes down on Monday morning

**Probability**: medium · **Impact**: high

**Symptoms**: assessor gets DNS error or 5xx when clicking the link.

**Likely causes**:
- Laptop suspended (lid closed, power management).
- cloudflared service not running after reboot.
- Wi-Fi disconnected.
- Domain DNS not propagated.

**Mitigations**:
- Test from cellular hotspot Sunday evening *and* Monday morning.
- Install cloudflared as a systemd service, enabled at boot.
- `just no-sleep` masks sleep targets on Fedora.
- DNS propagation: configure the named tunnel Sunday, not Monday.
- Plug in mains power; close all other applications.

**Fallback**: a 90-second Loom recording, linked from the email
underneath the live URL, shows the demo working. If the live link
fails, the assessor still sees what was built.

---

### R3 — Model accuracy is poor, demo looks bad

**Probability**: medium · **Impact**: medium

**Symptoms**: segmentation masks are sparse, miss obvious cables, or
hallucinate cables on background features.

**Likely causes**:
- TTPLA → real-imagery domain shift (TTPLA is rural-transmission, the
  example images are something else).
- Mask threshold too high (sparse) or too low (hallucinations).
- Test images are out-of-distribution from training.

**Mitigations**:
- Pick example images carefully — they should be in TTPLA's
  distribution. Save the failure cases for the deliberate
  failure-gallery.
- Tune the threshold on a held-out validation image; expose the
  threshold as a frontend setting if needed.
- Lead the email and methodology with the framing "the focus is the
  pipeline architecture, methodology framing, and engineering
  practice; model quality on TTPLA-style imagery is secondary because
  the real research direction is LV-specific data we don't have access
  to yet."

**The accompanying `docs/methodology.md` already makes this argument
explicit.**

---

### R4 — TTPLA dataset format different from assumed

**Probability**: medium · **Impact**: medium

**Symptoms**: `scripts/ttpla_to_masks.py` crashes or produces empty
masks.

**Likely causes**:
- TTPLA ships annotations in a format we didn't expect (we assumed
  COCO-like; reality might be Pascal VOC or custom).
- File extensions vary (`.JPG` vs `.jpg`).
- Images in subdirectories, not flat.

**Mitigations**:
- The `scripts/ttpla_to_masks.py` script inspects the actual directory
  structure before assuming a layout.
- Allow `--annotations-format` flag to support alternate formats.
- Keep the script idempotent so a partial run can be resumed.

---

### R5 — GTX 1050 Ti runs hot, throttles or crashes during demo

**Probability**: low–medium · **Impact**: medium

**Symptoms**: requests time out, fans audible, GPU utilisation
mysteriously low.

**Mitigations**:
- Inference budget is comfortable at the v2 default of 768×768
  (~300–500 ms per tile on the 1050 Ti), so even a 4K image is a
  few-second wait. Thermal limits unlikely to bite.
- Set `nvidia-smi -pl <power-limit>` if needed to cap power draw.
- Demo laptop placed on a flat surface with airflow underneath.

---

### R6 — Disk fills up with uploads / weights / logs

**Probability**: low · **Impact**: low

**Mitigations**:
- 24-hour retention policy on uploads (already in `Settings`).
- Logs to stdout, not files (cloudflared captures to journald with
  systemd's default rotation).
- Pre-flight check: `df -h /` before sending the email.

---

### R7 — A dependency upgrade breaks something between push and demo

**Probability**: low · **Impact**: medium

**Mitigations**:
- `uv.lock` pins exact versions; `uv sync` is deterministic.
- After Sunday evening's final test, do not run `uv sync` again
  before the demo.

---

### R8 — Frontend bug under specific browsers (Safari, mobile)

**Probability**: low–medium · **Impact**: low–medium

**Symptoms**: tabs don't render, Three.js viewer black, Leaflet tiles
missing.

**Mitigations**:
- Test in Chromium, Firefox, and Safari (use BrowserStack or a friend
  with a Mac if no Mac available).
- Use ES modules and import-maps which are universally supported in
  recent browsers; avoid bleeding-edge JS features.
- Add a graceful-degradation banner: "this demo is tested on Chromium-
  family browsers."

---

## KTP-project risks (forward-looking)

These do not need action this week but are worth flagging for future
project planning.

### R9 — SSEN data access is slower than expected

The most-cited reason KTP projects under-deliver is data-access delays.
Phase 1 should explicitly negotiate data access in the first month.
Without LV-specific data, the project is research-without-application
risk.

### R10 — Domain shift between SSEN regions

Models trained on one DNO region may not generalise to another (rural
Wales vs urban London look very different from above). Geographic
generalisation must be a first-class evaluation concern from Phase 1.

### R11 — Operational integration debt

A model that works in a notebook but doesn't integrate with SSEN's
ArcGIS workflow is academic only. Phase 3's integration workstream
should start in Phase 2 to avoid a last-minute integration crunch.

### R12 — Adoption resistance

LV records are owned by people inside SSEN. A new automated source of
LV data will be received variably — some teams will welcome it, others
will see it as competing with their work. Knowledge-transfer and joint
work-stream activities in Phase 3 are the mitigation; this is a
classic KTP success factor.

### R13 — Regulatory ambiguity around AI-derived asset records

NUAR is built on the assumption that submissions are authoritative
asset records. Model predictions are not authoritative; the regulatory
treatment of AI-derived intermediate outputs is still being clarified.
The project must engage with this uncertainty rather than pretending
it's settled.

---

## Risk-tracking discipline

For the sprint: review this register Sunday evening, confirm each
R1–R8 mitigation has been done or has an explicit not-applicable
reason, and move on.

For the KTP: a register like this is a standard PM artefact; SSEN's
project office likely has a template. Adapt to it rather than
reinventing.
