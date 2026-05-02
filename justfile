# Convenience commands. Run `just` to list.

default:
    @just --list

# Install all deps (incl. training/dev extras)
install:
    uv sync --all-extras

# Run the dev server with autoreload
dev:
    uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Run the production server (no reload)
serve:
    uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# Lint
lint:
    uv run ruff check app training

# Format
fmt:
    uv run ruff format app training

# Run tests
test:
    uv run pytest

# Start the cloudflared tunnel (requires prior `cloudflared tunnel create ktp-demo`)
tunnel:
    cloudflared tunnel run ktp-demo

# Stop laptop sleep / lid-close suspension during demo period
no-sleep:
    sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
    @echo "sleep masked. run `just allow-sleep` to undo."

# Re-enable sleep after demo
allow-sleep:
    sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target

# Quick health check
ping:
    @curl -s http://localhost:8000/health | jq .
