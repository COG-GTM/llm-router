.PHONY: services dev bench bench-chat ui e2e

services:
	docker compose up -d

dev:
	uv run uvicorn app.main:app --port 8000

bench:
	k6 run bench/k6.js

bench-chat:
	k6 run bench/k6_chat.js

ui:
	cd frontend && npm ci && npm run build

e2e:
	bash scripts/run_e2e.sh
