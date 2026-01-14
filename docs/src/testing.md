# Testing Guide

Element DeepLabCut includes **functional/integration test scripts** to verify the end-to-end workflow with DeepLabCut 3.x (PyTorch) and DataJoint integration. These tests are designed as **smoke tests** that validate the complete pipeline without requiring heavy computation or real model training.

**Test Type**: These are **functional/integration tests**, not unit tests. They test the entire pipeline end-to-end, including:
- Database interactions (DataJoint schemas and tables)
- File system operations (DLC project creation, file I/O)
- DeepLabCut API calls (project creation, training dataset creation, inference)
- DataJoint table population (Model, PoseEstimation, etc.)
- Cross-component integration (how all pieces work together)

While they use mocking to avoid actual model training (which would take hours/days), they still exercise the full workflow and integration between components. This makes them **functional smoke tests** - they verify the pipeline works correctly without heavy computation.

## Unit Tests

You also have **unit tests** in the `tests/` directory that test individual functions in isolation:

- **`tests/test_pretrained_workflow.py`**: Unit tests for pretrained model logic
  - `test_pretrained_model_registration` - Tests model registration
  - `test_insert_pretrained_model` - Tests inserting pretrained models
  - `test_pretrained_vs_trained_detection` - Tests detection logic
  - `test_pretrained_model_validation` - Tests validation
  - `test_parameter_merging` - Tests parameter merging

- **`tests/test_pipeline.py`**: Unit tests for trained model logic
  - `test_generate_pipeline` - Tests schema structure
  - `test_recording_info` - Tests recording data retrieval
  - `test_pose_estimation` - Tests pose estimation data

**Run unit tests:**
```console
# Run all unit tests
pytest tests/

# Run only pretrained workflow unit tests
pytest tests/test_pretrained_workflow.py

# Run only trained workflow unit tests
pytest tests/test_pipeline.py
```

**Unit tests vs Functional tests:**
- **Unit tests** (`tests/`): Fast (< 1 min), no DLC required, test individual functions
- **Functional tests** (`test_*.py`): Slower, require DLC, test full end-to-end workflows

The tests cover both workflow modes:
- **Trained Models**: Complete workflow from project creation to trained inference
- **Pretrained Models**: Quick inference using pre-built models from the DLC Model Zoo

See [Workflows](./workflows.md) for details on the differences between these modes.

## Test Scripts

### `test_model_train_run.py` - Trained Model Workflow

This script tests the **trained model workflow** (see [Workflows](./workflows.md#trained-models-workflow) for details):

1. ✅ Creates or reuses a DLC project
2. ✅ Generates mock labeled data (no manual labeling required)
3. ✅ Creates mock training dataset compatible with DLC 3.x
4. ✅ Runs mocked training (creates fake snapshot files + `pytorch_config.yaml`)
5. ✅ Inserts model into `element_deeplabcut.model.Model` table
6. ✅ Runs trained inference with `PoseEstimation` pipeline
7. ✅ Stores pose results into DataJoint tables

**Usage:**

```console
# Full workflow: create project, train, infer
python test_model_train_run.py

# Use existing trained model (skip training)
python test_model_train_run.py --skip-training

# Only train, don't infer
python test_model_train_run.py --skip-inference

# Custom model name
python test_model_train_run.py --model-name my_model
```

**In Docker:**

```console
docker compose run --rm client python test_model_train_run.py
# Or: make test-trained
```

### `test_pretrained_model_run.py` - Pretrained Model Inference

This script tests the **pretrained model workflow** (see [Workflows](./workflows.md#pretrained-models-workflow) for details):

1. ✅ Tests pretrained model inference (SuperAnimal quadruped, topviewmouse, etc.)
2. ✅ Handles video file discovery and processing
3. ✅ Database cleanup and model verification
4. ✅ Docker-aware path handling

**Usage:**

```console
# Test with SuperAnimal quadruped model
python test_pretrained_model_run.py superanimal_quadruped

# Test with SuperAnimal topviewmouse model
python test_pretrained_model_run.py superanimal_topviewmouse
```

**In Docker:**

```console
docker compose run --rm client python test_pretrained_model_run.py superanimal_quadruped
# Or: make test-pretrained
```

## Test Types Explained

### Functional/Integration Tests (These Scripts)

`test_model_train_run.py` and `test_pretrained_model_run.py` are **functional/integration tests**:

- ✅ Test the **entire end-to-end workflow** (project creation → training → inference → database storage)
- ✅ Test **integration between components** (DataJoint, DeepLabCut, file system)
- ✅ Use **mocking** to avoid heavy computation (no real training, but real DLC API calls)
- ✅ Verify **data flow** through the entire pipeline
- ✅ Suitable for **CI/CD** to catch integration issues

**Why not unit tests?** Unit tests test individual functions in isolation. These tests verify that all components work together correctly, which is integration testing.

### Unit Tests (Separate)

For true unit tests (testing individual functions), see:
- `tests/test_pretrained_workflow.py` - Unit tests for pretrained model logic
- `tests/test_pipeline.py` - Unit tests for trained model logic

These unit tests:
- Test individual functions/methods in isolation
- Don't require DLC installation
- Run very fast (< 1 minute)
- Use mocks extensively to isolate components

## Prerequisites

### Local Setup

1. **Conda Environment** (recommended):

   ```console
   conda env create -f environment.yml
   conda activate element-deeplabcut
   pip install -e .
   ```

   See [Conda Environment Setup](../CONDA_ENV_SETUP.md) for details.

2. **Database Configuration**:

   Create `dj_local_conf.json` in the project root:

   ```json
   {
     "database.host": "localhost",
     "database.user": "root",
     "database.password": "your_password",
     "database.port": 3306,
     "custom": {
       "database.prefix": "test_"
     }
   }
   ```

3. **Test Videos** (optional):

   Place test video files in `./test_videos/` directory. Supported formats: `.mp4`, `.avi`, `.mov`

### Docker Setup

See [Docker Setup](./docker.md) for complete Docker installation and configuration.

## Test Features

### Mock Training

The tests use **mocked training** to avoid heavy computation:

- Creates fake snapshot files: `snapshot-1000.index`, `.meta`, `.data-00000-of-00001`, `.pth`
- Generates minimal valid `pytorch_config.yaml` with required keys
- Converts `dlc-models` paths to `dlc-models-pytorch` for PyTorch
- Sets `snapshotindex = 0` and `engine = "pytorch"` in config

**No real training occurs** - this is a functional test to verify the workflow.

### Mock Inference

The tests use **mocked inference** to avoid GPU requirements:

- Patches `get_scorer_name` to return dummy strings
- Patches `get_model_snapshots` to create mock snapshots if missing
- Patches `load_state_dict` to use `strict=False` for mock state dicts
- Lenient pickle validation for mock models

**Results are not accurate** (mock model weights), but the workflow is validated.

### Database Cleanup

Both test scripts include comprehensive database cleanup at the start:

- Deletes `PoseEstimation` and `PoseEstimationTask` entries
- Removes test models (names starting with `test_`)
- Removes `ModelTraining` entries
- Ensures test repeatability

### Docker-Aware Path Handling

The test scripts automatically detect Docker environment and adjust paths:

- **Docker**: Uses `/app/test_videos` (from project mount)
- **Local**: Uses `./test_videos`
- Can be overridden with `DLC_ROOT_DATA_DIR` environment variable

## Running Tests

### Local Execution

```console
# Activate conda environment
conda activate element-deeplabcut

# Run trained model test
python test_model_train_run.py

# Run pretrained model test
python test_pretrained_model_run.py superanimal_quadruped
```

### Docker Execution

```console
# Start services
docker compose up -d

# Run tests
docker compose run --rm client python test_model_train_run.py
docker compose run --rm client python test_pretrained_model_run.py superanimal_quadruped

# Or use Makefile
make test-trained
make test-pretrained
```

## Test Output

The test scripts provide detailed status output:

```
============================================================
Testing Trained Model Workflow (Training + Inference)
============================================================

[1/12] ✓ Checking database connection
   ✓ Database connection successful

[2/12] ✓ Cleaning up database (removing test data from previous runs)
   🗑️ Deleted 5 PoseEstimation entry/entries
   🗑️ Deleted 3 PoseEstimationTask entry/entries
   ✅ Total: 8 entry/entries cleaned

[3/12] ✓ Finding video files
   📹 Found 2 video file(s)
   ...

[12/12] ✓ Running inference
   ✅ Inference completed!
```

## Troubleshooting

### Database Connection Errors

**Error**: `pymysql.err.OperationalError: (1045, "Access denied")`

**Solution**: Check your `dj_local_conf.json` password matches your database password. For Docker, use `datajoint` (default).

### No Video Files Found

**Error**: `No video files found in ./test_videos`

**Solution**:
- Ensure video files are in `./test_videos/` directory
- Check file formats (supported: `.mp4`, `.avi`, `.mov`)
- In Docker, videos are automatically available at `/app/test_videos` (from project mount)

### DLC 3.x Compatibility Issues

The tests include workarounds for known DLC 3.x issues:

- **Labeled data format**: Fixed DataFrame index to be string paths
- **Snapshot discovery**: Mocks `get_model_snapshots` to always find/create snapshots
- **State dict loading**: Uses `strict=False` for mock state dicts
- **Metadata validation**: Lenient handling of pickle metadata mismatches

If you encounter other issues, check the test script comments for additional workarounds.

### Task Length Errors

**Error**: `pymysql.err.DataError: (1406, "Data too long for column 'task'")`

**Solution**: The test scripts automatically truncate task names to 32 characters. If you see this error, check that truncation is happening early in the script.

## CI/CD Integration

These tests are suitable for CI/CD pipelines:

- **Fast execution**: No real training or GPU required
- **Deterministic**: Mocked components ensure consistent results
- **Isolated**: Database cleanup ensures clean state
- **Docker-ready**: Can run in containerized environments

**Example GitHub Actions workflow:**

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Start services
        run: docker compose up -d
      - name: Run tests
        run: docker compose run --rm client python test_model_train_run.py
```

## Next Steps

- See [Docker Setup](./docker.md) for Docker configuration
- See [Tutorials](./tutorials/) for workflow examples
- See [Concepts](./concepts.md) for architecture details
- See [Conda Environment Setup](../CONDA_ENV_SETUP.md) for detailed setup instructions

