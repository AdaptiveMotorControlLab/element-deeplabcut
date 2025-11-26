# Testing Guide

## Setup

See [CONDA_ENV_SETUP.md](../CONDA_ENV_SETUP.md) for environment setup.

### Database

```bash
docker compose -f docker-compose-db.yaml up -d
```

Create `dj_local_conf.json`:

```json
{
  "database.host": "localhost",
  "database.user": "root",
  "database.password": "simple",
  "database.port": 3306
}
```

## Running Tests

### Unit Tests (No DLC needed)

```bash
pytest tests/test_pretrained_workflow.py -v
pytest tests/test_pipeline.py -v
```

### Inference Tests (Requires DLC + videos)

```bash
# Install DLC
pip install -e ".[elements,dlc_default,tests]"

# Set video directory
export DLC_ROOT_DATA_DIR=./test_videos
export DLC_PROCESSED_DATA_DIR=./test_videos/output

# Run
pytest tests/test_pretrained_workflow.py::test_pretrained_inference_workflow -v
```

## Quick Examples

### Pretrained Workflow

```python
from element_deeplabcut import model

# 1. Register model
model.PretrainedModel.populate_common_models(["superanimal_quadruped"])

# 2. Insert model
model.Model.insert_pretrained_model(
    model_name="my_model",
    pretrained_model_name="superanimal_quadruped",
    prompt=False,
)

# 3. Create task
model.PoseEstimationTask.generate(
    recording_key,
    model_name="my_model",
    analyze_videos_params={"video_inference": {"scale": 0.4}},
)

# 4. Run inference
model.PoseEstimation.populate()
```

### Trained Workflow

```python
# 1. Insert trained model
model.Model.insert_new_model(
    model_name="my_trained",
    dlc_config="path/to/config.yaml",
    shuffle=1,
    trainingsetindex=0,
    prompt=False,
)

# 2. Create task and run
model.PoseEstimationTask.generate(recording_key, model_name="my_trained")
model.PoseEstimation.populate()
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No module named 'element_lab'" | `pip install -e ".[elements]"` |
| Database connection failed | Check `docker ps` and credentials |
| "Pretrained model not found" | `model.PretrainedModel.populate_common_models()` |
| CUDA out of memory | Use `"batchsize": 1` or `"gputouse": None` |
| Slow inference | Use GPU, reduce `scale`, use shorter videos |
