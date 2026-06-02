.PHONY: smoke baseline full upgrade recheck external paper

PYTHON ?= python
DATA_DIR ?= data/raw/eegmat

# Reproduce the full pipeline from raw EDF downloads through final figures
full:
	$(PYTHON) scripts/run_pipeline.py --download --data_dir "$(DATA_DIR)" --window_seconds 4 --overlap 0.5 --n_boot 2000

# Quick smoke test with synthetic data (no download needed)
smoke:
	$(PYTHON) scripts/smoke_test_synthetic.py

# Reproduce the baseline single-dataset outputs
baseline:
	$(PYTHON) scripts/run_pipeline.py --data_dir "$(DATA_DIR)" --output_dir outputs_reproduced --window_seconds 4 --overlap 0.5 --n_boot 500

# Multi-dataset analysis (MAT + STEW + DS007262)
upgrade:
	$(PYTHON) scripts/run_all_journal_upgrade.py

# PhD-level statistical audit
recheck:
	$(PYTHON) scripts/run_all_phd_revision_tests.py

# External validation on DS007262
external:
	$(PYTHON) scripts/multi_dataset_pipeline.py --datasets ds007262

# Re-build the paper PDF from source
paper:
	cd paper/tex && pdflatex main.tex && pdflatex main.tex

# Regenerate all figures at 600 DPI (publication quality)
figures:
	$(PYTHON) -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; plt.rcParams.update({'figure.dpi':600,'savefig.dpi':600})" && \
	$(PYTHON) -c "from src.eeg_cogstates.visualization import make_all_figures; make_all_figures('outputs_reproduced/features/eeg_features.csv', 'outputs_reproduced/statistics/feature_stat_tests.csv', 'outputs_reproduced/models', 'outputs_reproduced/figures')"

# Run unit tests
test:
	$(PYTHON) -m pytest tests/ -v

# Create Dockerfile for full reproducibility
docker:
	echo "FROM python:3.11-slim" > Dockerfile && \
	echo "WORKDIR /app" >> Dockerfile && \
	echo "COPY requirements.txt ." >> Dockerfile && \
	echo "RUN pip install --no-cache-dir -r requirements.txt" >> Dockerfile && \
	echo "COPY . ." >> Dockerfile && \
	echo "CMD [\"python\", \"finish_all.py\"]" >> Dockerfile
