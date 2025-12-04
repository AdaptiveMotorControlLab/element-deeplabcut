.PHONY: help build up down shell test-trained test-pretrained clean \
	quad-superanimal quad-superanimal-auto-refine quad-superanimal-no-adapt quad-superanimal-predict quad-superanimal-refine \
	quad-superanimal-dataset quad-superanimal-train-fast quad-superanimal-train-slow install-dlc-gui

help:
	@echo "make build          - Build Docker image"
	@echo "make up             - Start services"
	@echo "make down           - Stop services"
	@echo "make shell          - Interactive shell"
	@echo "make test-trained   - Run trained model test (CPU, or GPU if available)"
	@echo "make test-pretrained - Run pretrained model test (CPU, or GPU if available)"
	@echo "make test-trained-gpu   - Run trained model test with GPU (requires nvidia-container-toolkit)"
	@echo "make test-pretrained-gpu - Run pretrained model test with GPU (requires nvidia-container-toolkit)"
	@echo "Workflow (EXACT ORDER) - PyTorch engine (default):"
	@echo "  1. make quad-superanimal           - Create project + SuperAnimal predictions + extract frames"
	@echo "     OR make quad-superanimal-auto-refine - Same as above (no auto-refine, use step 2)"
	@echo "     OR make quad-superanimal-no-adapt - Same but without video adaptation (if OOM errors)"
	@echo "  2. make quad-superanimal-refine    - Open DLC refine GUI → correct + SAVE (creates CollectedData_*.csv/.h5)"
	@echo "  3. make quad-superanimal-dataset   - Create training dataset (.mat files) from labels"
	@echo "  4. make quad-superanimal-train-fast - Fast training (10 epochs) - combines dataset + train"
	@echo "     OR make quad-superanimal-train-slow - Slow training (50 epochs)"
	@echo ""
	@echo "Note: Step 2 is REQUIRED - SuperAnimal predictions ≠ training labels until you save in GUI"
	@echo "Note: All targets use --engine pytorch (PyTorch ModelZoo with HRNet-W32 + Faster R-CNN)"
	@echo "make install-dlc-gui - pip install 'deeplabcut[gui]' in current environment"
	@echo "make clean          - Remove volumes"

build:
	docker compose build
	@echo "Built image: element-deeplabcut-client"

up:
	docker compose up -d

down:
	docker compose down

shell:
	docker compose run --rm client -i

# Helper to get the compose project network name
COMPOSE_PROJECT_NAME := $(shell basename $$(pwd) | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g')
NETWORK_NAME := $(COMPOSE_PROJECT_NAME)_element_network

test-trained:
	@echo "⚠️  Note: docker compose run doesn't support --gpus flag"
	@echo "   GPU access requires using docker run directly or nvidia-container-toolkit"
	@echo "   Tests will run on CPU if GPU is not available"
	@docker compose ps db > /dev/null 2>&1 || docker compose up -d db
	@echo "Waiting for database..."
	@sleep 3
	docker compose run --rm client -c "python test_trained_inference.py --gpu 0 --batch-size 4"

test-pretrained:
	@echo "⚠️  Note: docker compose run doesn't support --gpus flag"
	@echo "   GPU access requires using docker run directly or nvidia-container-toolkit"
	@echo "   Tests will run on CPU if GPU is not available"
	@docker compose ps db > /dev/null 2>&1 || docker compose up -d db
	@echo "Waiting for database..."
	@sleep 3
	docker compose run --rm client -c "python test_video_inference.py superanimal_quadruped --gpu 0 --detector-batch-size 4"

# GPU-enabled test targets (requires nvidia-container-toolkit)
test-trained-gpu:
	@echo "🚀 Running with GPU support (requires nvidia-container-toolkit)"
	@docker compose ps db > /dev/null 2>&1 || docker compose up -d db
	@echo "Waiting for database..."
	@sleep 3
	@IMAGE=$$(docker compose config 2>/dev/null | grep -A 5 '^  client:' | grep 'image:' | awk '{print $$2}' 2>/dev/null || docker images --format '{{.Repository}}:{{.Tag}}' | grep element-deeplabcut-client | head -1); \
	if [ -z "$$IMAGE" ]; then \
		echo "Building image..."; \
		docker compose build client; \
		IMAGE=element-deeplabcut-client; \
	fi; \
	docker run --rm --gpus all \
		--network $(NETWORK_NAME) \
		-e DJ_HOST=db -e DJ_USER=root -e DJ_PASS=$${DJ_PASS:-datajoint} -e DJ_PORT=3306 \
		-e DATABASE_PREFIX=$${DATABASE_PREFIX:-} -e DLC_ROOT_DATA_DIR=$${DLC_ROOT_DATA_DIR:-/app/test_videos} \
		-e PYTHONUNBUFFERED=1 -e NVIDIA_VISIBLE_DEVICES=all \
		-v $${DLC_DATA_DIR:-./test_videos}:/app/data \
		$$([ -f dj_local_conf.json ] && echo "-v $$(pwd)/dj_local_conf.json:/app/dj_local_conf.json:ro") \
		-v $$(pwd):/app \
		$$IMAGE \
		-c "python test_trained_inference.py --gpu 0 --batch-size 4"

test-pretrained-gpu:
	@echo "🚀 Running with GPU support (requires nvidia-container-toolkit)"
	@docker compose ps db > /dev/null 2>&1 || docker compose up -d db
	@echo "Waiting for database..."
	@sleep 3
	@IMAGE=$$(docker compose config 2>/dev/null | grep -A 5 '^  client:' | grep 'image:' | awk '{print $$2}' 2>/dev/null || docker images --format '{{.Repository}}:{{.Tag}}' | grep element-deeplabcut-client | head -1); \
	if [ -z "$$IMAGE" ]; then \
		echo "Building image..."; \
		docker compose build client; \
		IMAGE=element-deeplabcut-client; \
	fi; \
	docker run --rm --gpus all \
		--network $(NETWORK_NAME) \
		-e DJ_HOST=db -e DJ_USER=root -e DJ_PASS=$${DJ_PASS:-datajoint} -e DJ_PORT=3306 \
		-e DATABASE_PREFIX=$${DATABASE_PREFIX:-} -e DLC_ROOT_DATA_DIR=$${DLC_ROOT_DATA_DIR:-/app/test_videos} \
		-e PYTHONUNBUFFERED=1 -e NVIDIA_VISIBLE_DEVICES=all \
		-v $${DLC_DATA_DIR:-./test_videos}:/app/data \
		$$([ -f dj_local_conf.json ] && echo "-v $$(pwd)/dj_local_conf.json:/app/dj_local_conf.json:ro") \
		-v $$(pwd):/app \
		$$IMAGE \
		-c "python test_video_inference.py superanimal_quadruped --gpu 0 --detector-batch-size 4"

quad-superanimal:
	# Step 1: Create project + SuperAnimal predictions + extract frames (PyTorch engine)
	# After completion, you can run 'make quad-superanimal-refine' to open the GUI
	# Uses optimized parameters for tiny animals in 4K video
	# NOTE: If you get CUDA OOM errors, try 'make quad-superanimal-no-adapt' instead
	python real_quadruped_training_example.py \
		--engine pytorch \
		--create-project \
		--project-name quad_superanimal \
		--experimenter mariia \
		--videos "/home/mariiapopova/element-deeplabcut/test_videos/IMG_7654.mp4" \
		--run-superanimal \
		--batch-size 1 \
		--detector-batch-size 1 \
		--bbox-threshold 0.1 \
		--pcutoff 0.05 \
		--pseudo-threshold 0.05 \
		--scale-list 300 400 500 600 700 800 \
		--video-adapt \
		--adapt-iterations 500 \
		--detector-epochs-inference 3 \
		--pose-epochs-inference 5 \
		--gpu 0 \
		--extract-frames

quad-superanimal-auto-refine:
	# Step 1: Create project + SuperAnimal predictions + extract frames (PyTorch engine)
	# NOTE: This is the same as 'quad-superanimal' - no auto-refine GUI
	# After this, run 'make quad-superanimal-refine' to open the refine GUI
	# Uses optimized parameters for tiny animals in 4K video
	python real_quadruped_training_example.py \
		--engine pytorch \
		--create-project \
		--project-name quad_superanimal \
		--experimenter mariia \
		--videos "/home/mariiapopova/element-deeplabcut/test_videos/IMG_7654.mp4" \
		--run-superanimal \
		--batch-size 2 \
		--detector-batch-size 1 \
		--bbox-threshold 0.1 \
		--pcutoff 0.05 \
		--pseudo-threshold 0.05 \
		--scale-list 300 400 500 600 700 800 \
		--video-adapt \
		--adapt-iterations 500 \
		--detector-epochs-inference 3 \
		--pose-epochs-inference 5 \
		--gpu 0 \
		--extract-frames

quad-superanimal-predict:
	# Run SuperAnimal inference on existing project (PyTorch engine)
	# Project directory is auto-detected as the latest quad_superanimal-mariia-* under the repo root.
	# NOTE: Usually not needed - 'make quad-superanimal' already runs SuperAnimal inference.
	# Uses optimized parameters for tiny animals in 4K video
	python real_quadruped_training_example.py \
		--engine pytorch \
		--project-name quad_superanimal \
		--experimenter mariia \
		--run-superanimal \
		--batch-size 2 \
		--detector-batch-size 1 \
		--bbox-threshold 0.1 \
		--pcutoff 0.05 \
		--pseudo-threshold 0.05 \
		--scale-list 300 400 500 600 700 800 \
		--video-adapt \
		--adapt-iterations 500 \
		--detector-epochs-inference 3 \
		--pose-epochs-inference 5 \
		--gpu 0 \
		--extract-frames

quad-superanimal-no-adapt:
	# Step 1 (NO VIDEO ADAPTATION): Create project + SuperAnimal predictions + extract frames
	# Use this if 'make quad-superanimal' fails with CUDA OOM errors
	# Disables video adaptation to reduce GPU memory usage
	python real_quadruped_training_example.py \
		--engine pytorch \
		--create-project \
		--project-name quad_superanimal \
		--experimenter mariia \
		--videos "/home/mariiapopova/element-deeplabcut/test_videos/IMG_7654.mp4" \
		--run-superanimal \
		--batch-size 1 \
		--detector-batch-size 1 \
		--bbox-threshold 0.1 \
		--pcutoff 0.05 \
		--pseudo-threshold 0.05 \
		--scale-list 300 400 500 600 700 800 \
		--no-video-adapt \
		--gpu 0 \
		--extract-frames

quad-superanimal-refine:
	# Step 2 (REQUIRED): Open DLC refine GUI to convert SuperAnimal predictions into training labels
	# This is an interactive step: you'll correct keypoints in the GUI and save them.
	# After saving, predictions become CollectedData_*.csv/.h5 files (real labels for training).
	# PREREQUISITE: Run 'make quad-superanimal' first to get predictions + frames
	python real_quadruped_training_example.py \
		--engine pytorch \
		--project-name quad_superanimal \
		--experimenter mariia \
		--refine-labels

quad-superanimal-dataset:
	# Step 3: Create training dataset (.mat files) from labels
	# REQUIRES: CollectedData_*.csv/.h5 files exist (created in step 2 when you save in refine GUI)
	# This generates training-datasets/.../*.mat files needed for training
	# PREREQUISITE: Run 'make quad-superanimal-refine' and SAVE your labels in the GUI first
	python real_quadruped_training_example.py \
		--engine pytorch \
		--project-name quad_superanimal \
		--experimenter mariia \
		--create-dataset

quad-superanimal-train-fast:
	# Step 4: One-shot dataset creation + fast training (10 epochs, save every 5)
	# PREREQUISITES:
	#   1. make quad-superanimal (creates project + predictions + frames)
	#   2. make quad-superanimal-refine (converts predictions → labels via GUI)
	#   3. Then run this command
	python real_quadruped_training_example.py \
		--engine pytorch \
		--project-name quad_superanimal \
		--experimenter mariia \
		--create-dataset \
		--train \
		--epochs 10 \
		--save-epochs 5 \
		--train-batch-size 16 \
		--gpu 0

quad-superanimal-train-slow:
	# Step 4: One-shot dataset creation + slow training (50 epochs, save every 10)
	# PREREQUISITES:
	#   1. make quad-superanimal (creates project + predictions + frames)
	#   2. make quad-superanimal-refine (converts predictions → labels via GUI)
	#   3. Then run this command
	python real_quadruped_training_example.py \
		--engine pytorch \
		--project-name quad_superanimal \
		--experimenter mariia \
		--create-dataset \
		--train \
		--epochs 50 \
		--save-epochs 10 \
		--train-batch-size 16 \
		--gpu 0
	

install-dlc-gui:
	pip install 'deeplabcut[gui]'

clean:
	docker compose down -v
