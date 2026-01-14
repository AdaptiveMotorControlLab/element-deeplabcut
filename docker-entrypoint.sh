#!/bin/bash
# Docker entrypoint script for element-deeplabcut client
# Works with DeepLabCut Docker image which already has DLC installed

set -e

# Execute the command
exec "$@"


