.PHONY: help setup setup-system download process validate analyze all report report-clean clean data-stats

PYTHON = python3
VENV = venv
PIP = $(VENV)/bin/pip
PY = $(VENV)/bin/python

help:  ## Mostrar esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: $(VENV)/bin/activate  ## Configurar entorno virtual e instalar dependencias
$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,notebooks]"
	touch $(VENV)/bin/activate

setup-system:  ## Instalar dependencias del sistema (Debian 12)
	sudo apt-get update
	sudo apt-get install -y python3-dev python3-venv python3-pip \
		texlive-full texlive-lang-spanish biber latexmk \
		git make

download: setup  ## Descargar estructuras de PDB y AlphaFold
	$(PY) -m alphafold_comparison download

process: setup  ## Procesar y alinear pares de estructuras
	$(PY) -m alphafold_comparison process

validate: setup  ## Validar correspondencia de pares proteicos
	$(PY) -m alphafold_comparison validate

analyze: setup  ## Ejecutar análisis estructural
	$(PY) -m alphafold_comparison analyze

all: download process validate analyze  ## Ejecutar pipeline completo

report:  ## Compilar reporte LaTeX a PDF
	cd reports/latex && latexmk -pdf -interaction=nonstopmode main.tex

report-clean:  ## Limpiar artefactos de compilación LaTeX
	cd reports/latex && latexmk -C

clean:  ## Eliminar archivos generados (no datos)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

data-stats:  ## Mostrar estadísticas de datos
	@echo "=== Estadísticas de datos ==="
	@echo "Raw:"; du -sh data/raw/ 2>/dev/null || echo "  (no existe)"
	@echo "Processed:"; du -sh data/processed/ 2>/dev/null || echo "  (no existe)"
	@echo "Validated:"; du -sh data/validated/ 2>/dev/null || echo "  (vacío)"
	@echo "Results:"; du -sh data/results/ 2>/dev/null || echo "  (no existe)"
	@echo "PDB files:"; ls data/raw/pdb_files/ 2>/dev/null | wc -l
	@echo "AlphaFold files:"; ls data/raw/alphafold_models/ 2>/dev/null | wc -l
