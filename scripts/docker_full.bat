@echo off
REM Run from project root with stack up: docker compose up -d
docker compose exec spark bash -lc "cd /app && bash scripts/docker_full.sh"
