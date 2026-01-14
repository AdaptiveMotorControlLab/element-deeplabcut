# Workflow Modes

Element DeepLabCut supports two distinct workflow modes for pose estimation:

1. **Trained Models**: Models you train yourself using your own labeled data
2. **Pretrained Models**: Pre-built models from the DLC Model Zoo (e.g., SuperAnimal models)

This guide explains when to use each mode, how they differ, and how to implement them.

## Trained Models Workflow

### Overview

Trained models are custom models that you create by:

1. Creating a DeepLabCut project
2. Labeling training data (manually or programmatically)
3. Training the model on your labeled data
4. Using the trained model for inference

This workflow gives you maximum control and customization but requires more setup and time.

### When to Use Trained Models

Use trained models when:

- ✅ You need high accuracy for a specific experimental setup
- ✅ Your videos have unique characteristics (lighting, camera angle, species, etc.)
- ✅ You want to track specific body parts not covered by pretrained models
- ✅ You have time and resources for data labeling and training
- ✅ You need to fine-tune a model for your specific use case

### Workflow Steps

#### 1. Create DLC Project

Create a DeepLabCut project with your video data:

```python
import deeplabcut

# Create a new project
config_path = deeplabcut.create_new_project(
    "MyProject",
    "experimenter",
    videos=["path/to/video1.mp4", "path/to/video2.mp4"],
    working_directory="./dlc_projects",
    copy_videos=True,
)
```

#### 2. Label Training Data

Label frames in your videos to create training data:

```python
# Extract frames for labeling
deeplabcut.extract_frames(config_path, mode='automatic', algo='kmeans', numframes2pick=20)

# Label frames (opens GUI)
deeplabcut.label_frames(config_path)

# Or use programmatic labeling (see test_model_train_run.py for examples)
```

#### 3. Create Training Dataset

Generate the training dataset from labeled data:

```python
deeplabcut.create_training_dataset(config_path)
```

#### 4. Train Model

Train the model (or mock training for testing):

```python
# Real training (requires GPU, takes hours/days)
deeplabcut.train_network(config_path)

# Or use element-deeplabcut's ModelTraining table
from element_deeplabcut import train

# Insert training task
train.TrainingTask.insert1({
    "paramset_idx": 1,
    "video_set_id": 1,
})

# Populate to start training
train.ModelTraining.populate()
```

#### 5. Insert Model into Database

Once training is complete, insert the model:

```python
from element_deeplabcut import model
import deeplabcut

# Load config
dlc_config = deeplabcut.auxiliaryfunctions.read_config(config_path)

# Insert model
model.Model.insert_new_model(
    model_name="my_trained_model",
    dlc_config=dlc_config,
    shuffle=1,
    trainingsetindex=0,
    model_description="Model trained on my experimental setup",
)
```

#### 6. Run Inference

Use the trained model for pose estimation:

```python
# Create pose estimation task
model.PoseEstimationTask.insert1({
    "recording_id": 1,
    "model_name": "my_trained_model",
    "task_mode": None,  # Fresh run
})

# Populate to run inference
model.PoseEstimation.populate()
```

### Key Characteristics

- **Project Path**: Required - points to your DLC project directory
- **Training Data**: Required - labeled frames in your project
- **Model Snapshots**: Required - trained model weights (`.pth` files for PyTorch)
- **Config File**: Required - `config.yaml` from your DLC project
- **Training Time**: Hours to days (depending on dataset size and hardware)
- **Accuracy**: High for your specific setup
- **Flexibility**: Full control over body parts, training parameters, etc.

### Database Tables Used

- `train.VideoSet`: Training video sets
- `train.TrainingParamSet`: Training parameters
- `train.TrainingTask`: Training tasks
- `train.ModelTraining`: Training execution records
- `model.Model`: Model metadata (with `project_path` and training info)
- `model.PoseEstimationTask`: Inference tasks
- `model.PoseEstimation`: Inference results

## Pretrained Models Workflow

### Overview

Pretrained models are ready-to-use models from the DeepLabCut Model Zoo (e.g., SuperAnimal models) that can be used directly without any training. These models are trained on large, diverse datasets and work well for many common experimental setups.

### When to Use Pretrained Models

Use pretrained models when:

- ✅ You want to get started quickly without training
- ✅ Your experimental setup matches common scenarios (e.g., top-view mouse, quadruped animals)
- ✅ You don't have time/resources for data labeling and training
- ✅ You want to test pose estimation before committing to training
- ✅ Your videos are similar to the pretrained model's training data

### Available Pretrained Models

Common pretrained models include:

- **`superanimal_quadruped`**: For quadruped animals (mice, rats, etc.)
- **`superanimal_topviewmouse`**: For top-view mouse pose estimation
- **Other SuperAnimal models**: Various species and camera angles

See the [DLC Model Zoo](http://www.mackenziemathislab.org/dlc-modelzoo) for the full list.

### Workflow Steps

#### 1. Register Pretrained Model

First, register the pretrained model in the database:

```python
from element_deeplabcut import model

# Populate common pretrained models
model.PretrainedModel.populate_common_models()

# Or add a custom pretrained model
model.PretrainedModel.add(
    pretrained_model_name="superanimal_quadruped",
    source="SuperAnimal",
    version="1.0",
    species="quadruped",
    backbone_model_name="hrnet_w32",
    detector_name="fasterrcnn_resnet50_fpn_v2",
    default_params={
        "video_adapt": False,
        "scale": 0.4,
        "batchsize": 8,
    },
    description="SuperAnimal model for quadruped animals",
)
```

#### 2. Insert Model into Database

Insert the pretrained model as a usable model:

```python
model.Model.insert_pretrained_model(
    model_name="my_pretrained_model",
    pretrained_model_name="superanimal_quadruped",
    model_description="Using SuperAnimal quadruped model",
)
```

#### 3. Run Inference

Use the pretrained model for pose estimation:

```python
# Create pose estimation task
model.PoseEstimationTask.insert1({
    "recording_id": 1,
    "model_name": "my_pretrained_model",
    "task_mode": None,  # Fresh run
})

# Populate to run inference
model.PoseEstimation.populate()
```

### Key Characteristics

- **Project Path**: Not required - pretrained models don't use DLC projects
- **Training Data**: Not required - model is already trained
- **Model Snapshots**: Not required - weights are downloaded automatically by DLC
- **Config File**: Minimal - only inference parameters needed
- **Training Time**: Zero - model is ready to use
- **Accuracy**: Good for common scenarios, may need fine-tuning for specific setups
- **Flexibility**: Limited to predefined body parts and configurations

### Database Tables Used

- `model.PretrainedModel`: Lookup table of available pretrained models
- `model.Model`: Model metadata (with `project_path=""` and `_pretrained_model_name`)
- `model.PoseEstimationTask`: Inference tasks
- `model.PoseEstimation`: Inference results

**Note**: The `train` schema is **not used** for pretrained models.

## Comparison Table

| Feature | Trained Models | Pretrained Models |
|---------|---------------|-------------------|
| **Setup Time** | Days to weeks | Minutes |
| **Training Required** | Yes (hours to days) | No |
| **Data Labeling** | Required | Not required |
| **DLC Project** | Required | Not required |
| **Project Path** | Required in database | Empty string |
| **Model Snapshots** | Required (`.pth` files) | Downloaded automatically |
| **Accuracy** | High (for your setup) | Good (for common setups) |
| **Customization** | Full control | Limited to model's design |
| **Body Parts** | You define them | Model defines them |
| **Best For** | Specific experimental setups | Quick testing, common scenarios |
| **Database Tables** | `train` + `model` schemas | `model` schema only |

## Choosing the Right Mode

### Start with Pretrained Models

If you're new to pose estimation or want to test quickly:

1. Try a pretrained model that matches your setup
2. Run inference on a few test videos
3. Evaluate the results
4. If results are good enough → continue with pretrained models
5. If results need improvement → consider training a custom model

### Use Trained Models When

- Pretrained models don't match your experimental setup
- You need specific body parts not in pretrained models
- You need higher accuracy for your specific videos
- You have time and resources for training

### Hybrid Approach

You can use both modes in the same pipeline:

- Use pretrained models for initial exploration and quick results
- Train custom models for production use with higher accuracy
- Compare results from both approaches

## Code Examples

### Complete Trained Model Workflow

See `test_model_train_run.py` for a complete example of the trained model workflow, including:

- Project creation
- Mock labeled data generation
- Training dataset creation
- Model training (mocked for testing)
- Model insertion
- Inference execution

### Complete Pretrained Model Workflow

See `test_pretrained_model_run.py` for a complete example of the pretrained model workflow, including:

- Pretrained model registration
- Model insertion
- Inference execution

## Testing

Both workflow modes have dedicated test scripts:

- **Trained models**: `test_model_train_run.py`
- **Pretrained models**: `test_pretrained_model_run.py`

See the [Testing Guide](./testing.md) for details on running these tests.

## Next Steps

- See [Tutorials](./tutorials/) for step-by-step examples
- See [Concepts](./concepts.md) for architecture details
- See [Docker Setup](./docker.md) for containerized development
- See [Testing Guide](./testing.md) for functional tests

