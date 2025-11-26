# Dockerfile for element-deeplabcut client environment
# Uses DeepLabCut's official Docker image as base
FROM deeplabcut/deeplabcut:latest-jupyter

# Set working directory
WORKDIR /app

# Install additional system dependencies if needed
RUN apt-get update && apt-get install -y \
    git \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project
COPY . .

# Install element-deeplabcut and its dependencies
# The DLC image already has DeepLabCut installed, so we just need element-deeplabcut
RUN pip install -e .[dlc_default,elements,tests]

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set entrypoint to bash
ENTRYPOINT ["/bin/bash"]

# Default command - interactive bash shell
# Users can override with: docker compose run client -c "python your_script.py"
CMD ["-i"]

