# Conda Environment Setup Guide

## Quick Start

```bash
# Create environment (Python 3.10+ required for DeepLabCut 3.x)
conda create -n element-deeplabcut python=3.10
conda activate element-deeplabcut

# Install system dependencies
conda install -c conda-forge graphviz

# Install package
pip install -e ".[elements,tests]"
```

## With DeepLabCut

```bash
# Full setup with DeepLabCut
pip install -e ".[elements,dlc_default,tests]"
```

## Verify Installation

```bash
python -c "import element_deeplabcut; print('✅ Package installed')"
python -c "import datajoint as dj; print('✅ DataJoint available')"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Package conflicts | `conda env remove -n element-deeplabcut` and recreate |
| Graphviz not found | `conda install -c conda-forge graphviz` |

## Next Steps

1. Configure database: See [docs/src/testing.md](docs/src/testing.md) for database setup
2. Run tests: `pytest tests/ -v`
3. See [docs/src/docker.md](docs/src/docker.md) for Docker setup
