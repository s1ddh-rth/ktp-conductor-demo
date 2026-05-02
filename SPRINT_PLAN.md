# Sprint plan — Saturday → Monday morning

> Internal sprint timeline retained for context. Documents the
> hour-by-hour cadence used during the build; left in the repository
> as a record of how the work was actually paced.

## Saturday

### Morning (2–3 h) — environment + data
- [ ] Install `uv` if not already: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [ ] Clone this scaffold to your laptop, `git init`, push to a fresh GitHub repo
- [ ] `uv sync --all-extras` — installs everything
- [ ] Verify GPU: `uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
- [ ] Open Colab; mount Drive
- [ ] Download TTPLA into Drive (the GitHub release has links — confirm before downloading)
- [ ] If TTPLA ships polygon JSON, write `scripts/ttpla_to_masks.py` to rasterise to PNG masks
- [ ] Verify dataset structure on Colab: `ttpla/images/*.jpg` and `ttpla/masks/*.png`

### Afternoon (4 h) — kick off training, build local
- [ ] Start training on Colab T4: `python -m training.train --data /content/drive/MyDrive/ttpla --epochs 50 --out /content/drive/MyDrive/ktp-weights`
- [ ] While that runs: locally start the dev server with no weights and verify it serves the static frontend
  - `just dev`
  - Open http://localhost:8000 — landing page should render even without a model
- [ ] Sanity check the routes: hit `/health`, see "model missing" warning in logs
- [ ] Place a sample LAS file in `app/static/examples/thatcham_sample.laz` — Environment Agency open data is ideal (search "Environment Agency LiDAR composite Berkshire")
- [ ] Test `/api/lidar/sample` — should return classified points

### Evening (2 h) — first inference
- [ ] Training should be done by now (~2 h on T4)
- [ ] Download `unet_resnet34_ttpla.pth` from Drive to laptop's `weights/` folder
- [ ] Restart server (`just dev`) — health should now show "model loaded · cuda"
- [ ] Test `/api/segment` and `/api/vectorise` with a TTPLA validation image
- [ ] Verify the frontend renders the mask overlay and Leaflet linestrings

---

## Sunday

### Morning (3 h) — polish module 1, validate quality
- [ ] Eval script: compute IoU on a held-out 10% of TTPLA, save numbers + a few qualitative examples to `docs/screenshots/`
- [ ] Drop 3–4 well-chosen example images into `app/static/examples/`
  - one easy case (clear transmission line)
  - one harder case (occluded by trees)
  - one failure case (something the model genuinely gets wrong — being honest about failure is high-credibility)
- [ ] Add a "failures" gallery section to the methodology doc

### Afternoon (3 h) — module 2: hidden-cable inference
- [ ] Test the catenary scenario via the UI; tune the sag uncertainty band visually
- [ ] Test the topology scenarios; verify the fragment-bias visibly pulls the tree
- [ ] If the visual of topology is weak, hand-tune building positions in `app.js` to make a more compelling case
- [ ] Add a 4th scenario if you have time: a real Thatcham OSM building extract with a synthetic transformer

### Evening (3 h) — module 3: LiDAR + cloudflared
- [ ] Verify LiDAR module renders well; tune `linearity` threshold if cables aren't separating
- [ ] Set up cloudflared:
  - [ ] `cloudflared tunnel login`
  - [ ] `cloudflared tunnel create ktp-demo`
  - [ ] Buy/use a domain on Cloudflare (Namecheap → transfer is overkill; just buy directly on Cloudflare Registrar — `.dev` is ~£10/yr)
  - [ ] `cloudflared tunnel route dns ktp-demo ktp.<yourdomain>.dev`
  - [ ] Edit `~/.cloudflared/config.yml` (template in `scripts/`)
  - [ ] Test from phone on cellular: open `https://ktp.<yourdomain>.dev`
- [ ] Install as systemd service: `sudo cloudflared service install`

---

## Monday morning (2 h)

### Final checks
- [ ] `just no-sleep` — prevent laptop suspension
- [ ] Plug in laptop, close all other apps
- [ ] Restart server clean: `pkill uvicorn; just serve` in a tmux session
- [ ] Verify `/health` from external network (phone hotspot)
- [ ] Test all four tabs end-to-end
- [ ] Record a 90-second Loom walkthrough as a fallback
- [ ] Update README with screenshots
- [ ] Push final commit; tag as `v0.1`

### Email
- [ ] Draft the application email separately, kept out of the
  repository. Include: live URL, GitHub URL, methodology link,
  walkthrough video URL.
- [ ] Send between 9am and 11am Monday — the sweet spot for academic
  email.
- [ ] Keep laptop on and tunnel up for 48–72 hours after sending.

### Fallbacks if something breaks
- If model accuracy is poor: lead the email with "the focus of this prototype is the pipeline architecture and methodology framing — model quality on TTPLA is secondary, since the real research direction is LV-specific data and methods we don't have access to yet"
- If the live link is down when they click: the README has full reproduction instructions, and the Loom video shows it working end-to-end
- If catenary or topology are qualitatively weak on demo day: surface this explicitly in the methodology document rather than hiding it; honest scoping reads better than a polished omission
