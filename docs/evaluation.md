# Evaluation methodology

How the prototype is evaluated here, and how a production deployment
should be evaluated. The two are deliberately different: the
prototype's evaluation is constrained by what's available in 5 days;
the production evaluation reflects what would actually answer the
operator's question of "does this work."

---

## 1. Why pixel IoU is not enough

For thin-structure segmentation (cables, vessels, road centrelines),
pixel-IoU systematically under-rewards near-misses. A segmentation
that traces a cable but is offset by 2 pixels on a 3-pixel-wide cable
will have 0% IoU yet be operationally perfect. A more lenient metric
that respects "close enough is fine" matters.

The correct metric family for this problem is the **CCQ (Completeness,
Correctness, Quality)** triple, originating with Wiedemann, Heipke,
Mayer, and Jamet (1998) in road extraction. Given a buffer tolerance
ρ (typically 2–5 pixels):

- **Completeness** = length of correctly detected reference, in
  buffer ρ of prediction, divided by total reference length.
- **Correctness** = length of correctly predicted line, in buffer ρ
  of reference, divided by total prediction length.
- **Quality** = (Completeness × Correctness) / (Completeness +
  Correctness − Completeness × Correctness). This is the
  centerline-equivalent of IoU.

All three are reported at ρ = 3 px, alongside pixel IoU for
comparison with prior work that uses IoU as the primary metric.

---

## 2. Metrics we report

For the prototype's evaluation script (`training/evaluate.py`):

| Metric | At threshold | Comment |
|---|---|---|
| Pixel IoU | 0.5 | Comparable with TTPLA paper's reported numbers |
| Pixel precision | 0.5 | Helps diagnose "high recall, low precision" failure mode |
| Pixel recall | 0.5 | Helps diagnose missed cables |
| F1 | 0.5 | Standard for imbalanced segmentation |
| CCQ Completeness | 0.5, ρ=3px | Centerline-aware "how much did the model find" |
| CCQ Correctness | 0.5, ρ=3px | Centerline-aware "how much of what was found is real" |
| CCQ Quality | 0.5, ρ=3px | The headline thin-structure metric |
| Calibration (Brier) | n/a | Are confidence scores meaningful? |

**Threshold sweep.** For the headline IoU and Quality numbers, the
optimal threshold is found by sweeping τ ∈ {0.3, 0.4, 0.5, 0.6, 0.7}
on the validation set. The chosen threshold is fixed before test-set
evaluation to avoid optimism bias.

---

## 3. Data splits

### Prototype evaluation

The v1 trainer used a seed-42 random 80/10/10 partition over all
TTPLA images:

- **Train**: 80%, randomly selected with seed 42.
- **Validation**: 10%, selected for hyperparameter / threshold choices.
- **Test**: 10%, held out until the final evaluation pass.

### v2 update — TTPLA's canonical split is what we now use

The original claim above ("TTPLA does not ship with a canonical
split") was wrong. The dataset authors do publish a canonical
three-way split as `splitting_dataset_txt/{train,val,test}.txt`
totalling 1234 images (905 / 109 / 220), with 8 unassigned
post-publication images excluded. The v2 retrain uses this split
end-to-end, and the v2 evaluation runs against the 220-image test
bucket. Full report:
[`docs/evaluation_results_v2.md`](evaluation_results_v2.md).

The v2 canonical-split evaluation has been completed and achieves
**CCQ-Q 0.457 at τ=0.50**, **CCQ-Q 0.462 at τ=0.30**, with
**ECE 0.015 across both threshold choices** (ECE is a property of
the probability map, not the threshold). v2 numbers should be
treated as the headline; v1 numbers in
[`evaluation_results.md`](evaluation_results.md) and
[`evaluation_results_session.md`](evaluation_results_session.md)
are retained as the methodology-evolution record.

### Why random splitting is dangerous in practice

For the v1 prototype the random split was acceptable but
methodologically weak. Aerial imagery has strong spatial correlation
between images taken on the same flight: lighting, vegetation,
season, sensor settings cluster together. A random split puts
highly-similar images in train and test, so the test score
over-estimates real-world generalisation. The v1 session-grouped
evaluation tried to surface this, but the v1 trainer's random split
had already placed ≈90% of every session prefix into the training
set — so the session "test" bucket was largely composed of seen
imagery. v2's adoption of the canonical split fixes both problems
by construction.

**For a production evaluation**, the right protocol is **geographic
splitting**: train on imagery from one region, test on a held-out
region. This is the protocol used in modern aerial-imagery papers
(e.g. iSAID, FloodNet). The methodology document acknowledges this
gap explicitly.

---

## 4. Calibration

A 0.7-confidence prediction should be correct ≈ 70% of the time.
Models trained with focal loss are often over-confident, which is
worth detecting.

The evaluation computes **expected calibration error (ECE)** with 10
bins and saves the reliability diagram to
`docs/screenshots/eval/calibration.png`.

If ECE is large, **temperature scaling** (Guo et al., 2017) is the
standard fix: fit a single scalar temperature T on the validation set
and divide logits by T at inference time. This is a one-parameter
fit, takes seconds, and typically halves ECE without affecting
accuracy.

---

## 5. Failure analysis

Equally important to the headline numbers: a curated gallery of
failure cases with diagnoses.

The script picks 3 failures, deliberately chosen to illustrate
distinct mechanisms:

1. **A texture confusion failure.** Something the model classifies as
   a cable but isn't — typically a fine line of similar contrast
   (a fence, a cable shadow, a road marking).
2. **A scale failure.** A real cable too thin or too thick relative
   to the training distribution. This argues for multi-scale training
   augmentation.
3. **A context failure.** A cable visible to a human only because of
   the surrounding context (poles at both ends), where pure local
   features are inadequate. This argues for larger receptive fields
   or explicit pole/structure conditioning.

Each failure case appears in `docs/screenshots/eval/failures/` with
input | predicted-mask | overlay, and a one-sentence diagnosis.

---

## 6. Comparison with prior work

The original TTPLA paper (Abdelfattah et al., ACCV 2020) reports
instance-segmentation numbers, not binary semantic segmentation, so
direct comparison is partial:

| Source | Task | Metric | Reported |
|---|---|---|---|
| TTPLA paper (2020) | Instance segmentation, cables only | mAP@0.5 | (see paper) |
| TTPLA paper (2020) | Semantic segmentation, cables only | mIoU | (see paper) |
| Madaan et al. (2017) | Wire segmentation, synthetic + real | F1 | 0.84 |
| Yetgin & Gerek (2018) | Powerline detection (RGB) | Pixel accuracy | 0.96 |
| Ours | Semantic segmentation, cables only | mIoU, F1, CCQ-Q | (measured) |

Numbers from prior work are recorded with citations; values without a
source attribution are not paraphrased or extracted.

---

## 7. Deployment-time evaluation (production-relevant, not in-scope here)

For a real LV-mapping deployment, the evaluation question is not
"what's the IoU on TTPLA". It is "given a held-out 1km² area where
ground-truth digitised LV records exist, how complete and correct is
the predicted topology?" The metrics shift to:

- **Topology recall**: fraction of true LV edges with a corresponding
  predicted edge within a 5 m tolerance.
- **Topology precision**: fraction of predicted edges with a true
  counterpart within tolerance.
- **Building-coverage rate**: what fraction of LV-served buildings are
  correctly connected to the predicted network.
- **False-disconnection rate**: what fraction of buildings are left
  unconnected by the prediction (operationally serious — a missing
  connection is worse than an extra one).

These metrics are not computable on TTPLA; they require real LV
ground-truth data, which is itself an outcome of the larger project.

---

## 8. Resources

- Wiedemann, C., Heipke, C., Mayer, H., & Jamet, O. (1998). Empirical
  Evaluation of Automatically Extracted Road Axes.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On
  Calibration of Modern Neural Networks. *ICML*.
- Mosinska, A., Marquez-Neila, P., Kozinski, M., & Fua, P. (2018).
  Beyond the Pixel-Wise Loss for Topology-Aware Delineation. *CVPR*.
- Shit, S., Paetzold, J. C., et al. (2021). clDice — A Novel Topology-
  Preserving Loss Function for Tubular Structure Segmentation. *CVPR*.
