#!/usr/bin/env bash
# Run full stack inside the spark container (Cassandra must be up: docker compose up -d).
set -euo pipefail
cd /app
# Fast: only stream-clean a sample into review_small.json (does not rewrite huge review.json)
PREPARE_FAST_SAMPLE=1 python prepare_data.py
python scripts/setup_cassandra.py
python scripts/ingest_cassandra.py --truncate
python run_full_pipeline.py
python ml_visualizations.py
python data_visualization.py
echo "Done. Plots: outputs/plots/  Dashboard: streamlit run dashboard_professional.py (port 8501)"
