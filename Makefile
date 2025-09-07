.PHONY: .venv install run clean

# Name of the virtual environment folder
VENV := .venv
PYTHON := python3

# Create virtual environment
venv:
	$(PYTHON) -m venv $(VENV)

# Install dependencies inside venv
install: venv 
	$(VENV)/bin/pip install -r requirements.txt

# Run the program inside venv
run:
	set -a && source .env && set +a && $(VENV)/bin/python main.py

# Remove venv, caches, and build artifacts
clean:
	rm -rf $(VENV) \
	       *__pycache__ \
	       .pytest_cache \
	       *.pyc \
	       *.pyo \
	       *.pyd \
	       *.log \
	       build \
	       dist \
	       *.egg-info
