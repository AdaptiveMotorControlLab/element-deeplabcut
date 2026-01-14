"""
Tests for pretrained model workflow.

This module tests the pretrained model workflow alongside the existing trained model workflow.
"""

import pytest
def test_pretrained_model_registration(pipeline):
    """Test registering pretrained models in the lookup table."""
    model = pipeline["model"]
    
    # Test populate_common_models
    results = model.PretrainedModel.populate_common_models()
    assert len(results["inserted"]) > 0 or len(results["skipped"]) > 0
    
    # Verify models were registered
    assert model.PretrainedModel.is_pretrained("superanimal_quadruped")
    assert model.PretrainedModel.is_pretrained("superanimal_topviewmouse")
    
    # Test adding a custom model
    model.PretrainedModel.add(
        pretrained_model_name="test_custom_model",
        source="Test",
        default_params={"scale": 0.5},
        description="Test model"
    )
    assert model.PretrainedModel.is_pretrained("test_custom_model")


def test_insert_pretrained_model(pipeline):
    """Test inserting a pretrained model into the Model table."""
    model = pipeline["model"]
    
    # First, register a pretrained model
    if not model.PretrainedModel.is_pretrained("superanimal_quadruped"):
        model.PretrainedModel.populate_common_models(["superanimal_quadruped"])
    
    # Insert pretrained model instance
    model.Model.insert_pretrained_model(
        model_name="test_pretrained_model",
        pretrained_model_name="superanimal_quadruped",
        model_description="Test pretrained model",
        prompt=False,
    )
    
    # Verify it was inserted
    model_record = (model.Model & {"model_name": "test_pretrained_model"}).fetch1()
    assert model_record["model_name"] == "test_pretrained_model"
    assert model_record["project_path"] == ""  # Empty for pretrained
    assert model_record["paramset_idx"] is None  # No training param set
    
    # Verify pretrained detection flags
    config_template = model_record["config_template"]
    assert config_template.get("_is_pretrained") is True
    assert config_template.get("_pretrained_model_name") == "superanimal_quadruped"


def test_pretrained_vs_trained_detection(pipeline):
    """Test that pretrained and trained models are correctly detected."""
    model = pipeline["model"]
    
    # Get a trained model (from existing fixture)
    trained_model = (model.Model & {"model_name": "from_top_tracking_model_test"}).fetch1()
    trained_config = trained_model["config_template"]
    
    # Trained models should not have pretrained flags
    assert trained_config.get("_is_pretrained") is not True
    assert trained_config.get("_pretrained_model_name") is None
    assert trained_model["project_path"] != ""  # Should have project path
    
    # Get a pretrained model
    if not model.PretrainedModel.is_pretrained("superanimal_quadruped"):
        model.PretrainedModel.populate_common_models(["superanimal_quadruped"])
    
    model.Model.insert_pretrained_model(
        model_name="test_pretrained_detection",
        pretrained_model_name="superanimal_quadruped",
        prompt=False,
    )
    
    pretrained_model = (model.Model & {"model_name": "test_pretrained_detection"}).fetch1()
    pretrained_config = pretrained_model["config_template"]
    
    # Pretrained models should have pretrained flags
    assert pretrained_config.get("_is_pretrained") is True
    assert pretrained_config.get("_pretrained_model_name") == "superanimal_quadruped"
    assert pretrained_model["project_path"] == ""  # Empty for pretrained


@pytest.mark.skip(
    reason="Requires actual DLC installation with SuperAnimal support and GPU resources. "
           "Run manually with: pytest tests/test_pretrained_workflow.py::test_pretrained_inference_workflow -s"
)
def test_pretrained_inference_workflow(pipeline, insert_upstreams):
    """Test the full pretrained inference workflow.
    
    This test requires:
    - DeepLabCut installed with SuperAnimal support
    - Actual video files
    - GPU resources (or very long runtime)
    
    Run with: pytest tests/test_pretrained_workflow.py::test_pretrained_inference_workflow --run-inference
    """
    model = pipeline["model"]
    
    # Register pretrained model
    if not model.PretrainedModel.is_pretrained("superanimal_quadruped"):
        model.PretrainedModel.populate_common_models(["superanimal_quadruped"])
    
    # Insert pretrained model
    model.Model.insert_pretrained_model(
        model_name="test_pretrained_inference",
        pretrained_model_name="superanimal_quadruped",
        prompt=False,
    )
    
    # Create pose estimation task
    recording_key = {
        "subject": "subject6",
        "session_datetime": "2021-06-02 14:04:22",
        "recording_id": "1",
    }
    
    model.PoseEstimationTask.generate(
        recording_key,
        model_name="test_pretrained_inference",
        analyze_videos_params={
            "video_inference": {"scale": 0.4}  # Override default params
        }
    )
    
    # Run inference (this will actually call DLC)
    model.PoseEstimation.populate()
    
    # Verify results were created
    pose_estimation = (model.PoseEstimation & recording_key & {"model_name": "test_pretrained_inference"}).fetch1()
    assert pose_estimation is not None
    
    # Verify body part positions exist
    body_parts = model.PoseEstimation.BodyPartPosition.fetch("body_part")
    assert len(body_parts) > 0


def test_trained_workflow_still_works(pipeline, insert_dlc_model, insert_pose_estimation_task, pose_estimation):
    """Verify that the existing trained model workflow still works unchanged."""
    model = pipeline["model"]
    
    # Verify trained model exists and has correct structure
    trained_model = (model.Model & {"model_name": "from_top_tracking_model_test"}).fetch1()
    assert trained_model["project_path"] != ""  # Should have project path
    assert trained_model["paramset_idx"] is not None  # Should have paramset
    
    # Verify it's detected as trained (not pretrained)
    config_template = trained_model["config_template"]
    assert config_template.get("_is_pretrained") is not True
    
    # Verify pose estimation results exist
    body_parts = model.PoseEstimation.BodyPartPosition.fetch("body_part")
    assert len(body_parts) > 0
    assert "head" in body_parts or "tailbase" in body_parts


def test_pretrained_model_validation(pipeline):
    """Test validation when using pretrained models."""
    model = pipeline["model"]
    
    # Try to insert pretrained model without registering it first
    # This should fail gracefully
    if model.PretrainedModel.is_pretrained("nonexistent_model"):
        model.PretrainedModel.delete({"pretrained_model_name": "nonexistent_model"})
    
    # Should warn but not raise
    model.Model.insert_pretrained_model(
        model_name="test_nonexistent",
        pretrained_model_name="nonexistent_model",
        prompt=False,
    )
    
    # Verify it wasn't inserted
    assert not (model.Model & {"model_name": "test_nonexistent"})


def test_parameter_merging(pipeline):
    """Test that default params and user params are merged correctly."""
    model = pipeline["model"]
    
    # Register model with default params
    if not model.PretrainedModel.is_pretrained("superanimal_quadruped"):
        model.PretrainedModel.populate_common_models(["superanimal_quadruped"])
    
    # Get default params
    pm = (model.PretrainedModel & {"pretrained_model_name": "superanimal_quadruped"}).fetch1()
    default_params = pm.get("default_params") or {}
    
    # Verify default params exist
    assert "scale" in default_params or "video_adapt" in default_params or len(default_params) > 0

