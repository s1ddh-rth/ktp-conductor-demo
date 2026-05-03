# 30-month research roadmap

A Knowledge Transfer Partnership (KTP) typically runs as a structured
research project against a set of jointly-agreed milestones. This
document sketches what those milestones could look like for an LV-
conductor-mapping engagement with a UK Distribution Network Operator,
anchored in what the prototype demonstrates and what the gaps suggest.

The roadmap is written speculatively — the real roadmap would be
co-developed with the academic supervisors and the industry business
owner. As a positioning document, it traces a coherent path beyond
the prototype to the project's whole arc.

---

## Phase 1 (months 1–6): foundation and data

**Goal.** Establish a labelled LV-imagery corpus and a transferable
baseline model.

**Workstreams.**

- **Data acquisition pact.** With SSEN, identify which aerial / drone
  / mounted-camera datasets exist, what the licensing framework is,
  and what's ingestible. Catalogue the corpus by region, resolution,
  modality (RGB, NIR, LiDAR), and existing labels. Negotiate access.
- **Active labelling pipeline.** Build a labelling tool (CVAT or
  Label Studio) configured for the specific class taxonomy SSEN
  needs. Train a small initial labelling team. Use uncertainty
  sampling against the prototype model to prioritise images that
  most-improve the model when labelled.
- **Self-supervised pretraining.** On SSEN's unlabelled archive (which
  is likely many times larger than the labelled corpus), pretrain a
  vision encoder using DINOv2 (Oquab et al., 2023) or MAE (He et al.,
  2022). This is the single largest expected gain available — a
  well-pretrained encoder typically improves downstream segmentation
  IoU by 3–8 points.
- **Geographic generalisation evaluation.** Establish a held-out
  region (a single SSEN substation district) that the model never
  trains on. All progress claims must be validated on this region.

**Deliverables.**
- Labelled LV imagery corpus (target: 5,000+ tiles).
- Self-supervised pretrained encoder, with paper-style ablation.
- Baseline LV segmentation model with reproducible training recipe.
- A first internal report to SSEN with quantitative results on the
  held-out region.

**Publication target.** A workshop paper at a CVPR or ICCV workshop
(EarthVision, AI4EO) describing the dataset and the self-supervised
recipe.

---

## Phase 2 (months 7–18): the methodology phase

**Goal.** Push the segmentation, topology completion, and LiDAR fusion
to publishable quality.

**Workstreams.**

- **Topology-aware segmentation losses.** Move from focal+Dice to
  CL-Dice (Shit et al., 2021) or skeleton-recall losses, which
  directly optimise the connectivity properties needed for downstream
  graph extraction. Compare quantitatively.
- **Hidden-conductor inference as a learning problem.** Replace the
  Steiner-tree baseline with a graph neural network conditioned on
  building footprints, transformer locations, and observed cable
  fragments. Likely a GraphSAGE or GAT operating on a building-pole
  graph with edge-prediction head. Training data: synthesised by
  masking known segments of digitised LV networks (a self-supervised
  task where ground truth is "what the network actually looked like").
- **RGB-LiDAR fusion.** Mid-level fusion architecture: dual encoders
  (RGB CNN + sparse 3D CNN for LiDAR), cross-attention between
  modalities, shared decoder. Compare with single-modality baselines.
- **Uncertainty quantification.** Integrate Bayesian uncertainty
  (deep ensembles or MC-dropout) so predictions come with confidence
  intervals SSEN's GIS team can act on.

**Deliverables.**
- Three method-papers' worth of results (segmentation losses,
  topology GNN, RGB-LiDAR fusion).
- A model with uncertainty quantification calibrated against held-out
  data.
- An interim system demonstration to SSEN's senior engineering team.

**Publication targets.** A main-track conference paper at WACV, BMVC,
or IROS on the topology-completion GNN. A second workshop paper on
RGB-LiDAR fusion for utility infrastructure.

### Reinforcement-learning extension to topology completion

The Steiner-tree approximation in `app/ml/graph_complete.py` is a
deterministic baseline; the topology-completion problem also admits
a sequential-decision framing that fits the reinforcement-learning
toolkit. An agent observes a partial cable graph (transformer +
known buildings + observed fragments + candidate poles) and chooses
edges one at a time, with a reward shaping that combines (a) length
budget, (b) building-coverage rate, (c) alignment with observed
fragments, and (d) penalties for crossing protected features
(railways, motorways) sourced from OS Open Zoomstack. This is
related to the wider literature on RL-for-combinatorial-optimisation
(Dai et al., 2017, *Learning Combinatorial Optimization Algorithms
over Graphs*; Khalil et al., 2017, *Learning to Optimize via
Posterior Sampling*) where graph neural networks are trained as
state representations for an actor-critic agent.

A KTP-scale execution would start with offline RL on synthesised
LV layouts (using the augmented urban scenes from Phase 1), evaluate
on a held-out SSEN region, and benchmark against the Steiner-tree
baseline this prototype already ships. The expected payoff is in
constraint-handling — RL can encode operational constraints
(minimum clearances, route preferences along existing rights-of-way)
that linear-programming-style formulations cannot — rather than in
raw topology accuracy.

---

## Phase 3 (months 19–30): productionisation and impact

**Goal.** Move the research from prototype to operational tool, and
demonstrate measurable impact.

**Workstreams.**

- **Operational integration.** Work with SSEN's GIS team to integrate
  outputs into their ArcGIS environment. Define the API contract;
  build the connector. Comply with SSEN's data security policies.
- **Active deployment loop.** Field-trial the system on a SSEN region
  the network operator chooses. Compare predicted LV topology with
  ground truth (digitised by surveyors after the fact) and report
  cost-per-corrected-record.
- **Knowledge transfer.** The KTP framework requires a measurable
  skills uplift in the company. Run training sessions for SSEN's
  data and asset-management teams on the model's capabilities and
  failure modes.
- **Robustness and adversarial testing.** Stress-test the model on
  edge cases: heavy vegetation occlusion, atypical pole types,
  underground-to-overhead transitions, edge-of-frame artefacts. Document
  the failure envelope.

**Deliverables.**
- A production deployment in at least one SSEN region.
- An impact report with measured savings (engineer-hours saved per
  km of LV mapped, accuracy uplift over manual digitisation).
- A final KTP report meeting Innovate UK's evaluation criteria.

**Publication target.** A journal paper (IEEE Transactions on Smart
Grid, or IEEE Transactions on Power Delivery) describing the full
system and operational results. This is the academic deliverable that
matters most for the Associate's career.

---

## Cross-cutting workstreams (all phases)

### Open science

The KTP is part-funded by Innovate UK; non-commercial elements should
default to open. Specifically:

- Code released under MIT or Apache 2.0 unless SSEN identifies a
  specific commercial concern.
- Models released with model cards (Mitchell et al., 2019) describing
  intended use, limitations, evaluation results, and known failure
  modes.
- Datasets that don't contain SSEN-proprietary information released
  under CC-BY-SA via a UK academic data portal.
- Anonymised aggregate results published in peer-reviewed venues.

### Compliance and data governance

Every workstream must comply with:

- UK GDPR on any imagery containing identifiable people or properties.
- ESQCR (Electricity Safety, Quality and Continuity Regulations 2002)
  requirements where outputs feed into operational decisions.
- NUAR (National Underground Asset Register) protocols where outputs
  intersect with shared utility data.
- IEC 61968 (Common Information Model for distribution management)
  data formats for any GIS-bound deliverables.

See `data_governance.md` for detail.

### Industrial adoption signals

Things that would prove the KTP succeeded beyond the academic outputs:

- A production deployment in at least one DNO region (target: SSEN
  by month 30).
- Adoption interest from a second DNO (UKPN, NPg, WPD, ENWL, SPEN).
- Patent or commercial-IP filings where appropriate.
- A skill increment in the partner company measurable via SSEN's
  internal capability assessments.
