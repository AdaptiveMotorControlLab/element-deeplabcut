# Docker Setup

Element DeepLabCut provides Docker support for running tests and development in a containerized environment. This is particularly useful for:

- Running functional tests without setting up a local environment
- Ensuring consistent development environments across different machines
- CI/CD pipeline integration
- Quick testing of new features

## Prerequisites

- Docker and Docker Compose installed
- Test video files in `./test_videos/` directory (optional, for testing)

## Quick Start

### Start Services

Start the database and client containers:

```console
docker compose up -d
```

Or use the Makefile:

```console
make up
```

This starts:
- **Database service** (`db`): MySQL 8.0 database with health checks
- **Client service** (`client`): Development/test container with DeepLabCut pre-installed

### Run Tests

Run the functional test suite. There are two test scripts:

**1. Trained Model Workflow Test** (`test_trained_inference.py`):
Tests the complete trained model workflow (project creation → training → inference):

```console
docker compose run --rm client python test_trained_inference.py
```

Or use the Makefile:

```console
make test-trained
```

**2. Pretrained Model Workflow Test** (`test_video_inference.py`):
Tests pretrained model inference (SuperAnimal models):

```console
docker compose run --rm client python test_video_inference.py superanimal_quadruped
```

Or use the Makefile:

```console
make test-pretrained
```

See [Testing Guide](./testing.md) for details on both test scripts.

### Interactive Shell

Get an interactive shell in the container:

```console
docker compose run --rm client bash
```

Or use the Makefile:

```console
make shell
```

## Configuration

### Environment Variables

You can customize the setup using environment variables:

```console
# Database password
export DJ_PASS=your_password

# Database port (default: 3306)
export DB_PORT=3307

# Database prefix (default: test_)
export DATABASE_PREFIX=test_

# MySQL version (default: 8.0)
export MYSQL_VER=8.0
```

Or create a `.env` file in the project root:

```env
DJ_PASS=datajoint
DB_PORT=3306
DATABASE_PREFIX=test_
MYSQL_VER=8.0
```

### Database Configuration

The container automatically connects to the `db` service. You can also use an external database by setting:

```console
export DJ_HOST=your_database_host
export DJ_USER=your_username
export DJ_PASS=your_password
```

The container will look for database configuration in this order:

1. Environment variables (`DJ_HOST`, `DJ_USER`, `DJ_PASS`)
2. `dj_local_conf.json` file (if mounted)
3. Default DataJoint configuration

## Volumes

The following directories are mounted as volumes:

- `.` → `/app` - Project directory (includes `./test_videos` at `/app/test_videos`)
- `./test_videos` → `/app/data` - Test video files (also accessible at `/app/test_videos` from project mount)
- `./dj_local_conf.json` → `/app/dj_local_conf.json` - Database configuration (read-only)

**Note**: Test scripts automatically detect Docker and use `/app/test_videos` (from project mount).


## Usage Examples

See [Testing Guide](./testing.md) for detailed test usage examples. Quick reference:

```console
# Run trained model test
make test-trained

# Run pretrained model test  
make test-pretrained

# Interactive shell
make shell
```

### Development Workflow

For development, the project directory is mounted as a volume, so code changes are immediately available. However, if you install new dependencies, you'll need to rebuild:

```console
docker compose build
```

## Makefile Commands

The included Makefile provides convenient shortcuts:

| Command | Description |
|---------|-------------|
| `make build` | Build Docker image |
| `make up` | Start services in background |
| `make down` | Stop and remove containers |
| `make shell` | Interactive shell in container |
| `make test-trained` | Run `test_trained_inference.py` |
| `make test-pretrained` | Run `test_video_inference.py` |
| `make clean` | Remove containers and volumes |

## Troubleshooting

### Database Connection Issues

If you see database connection errors:

1. Check that the database is healthy:

   ```console
   docker compose ps
   ```

2. Verify environment variables:

   ```console
   docker compose run --rm client env | grep DJ_
   ```

3. Test database connection manually:

   ```console
   docker compose run --rm client python -c "import datajoint as dj; dj.config.load('/app/dj_local_conf.json'); print('Connected:', dj.conn())"
   ```

### Permission Issues

If you encounter permission issues with mounted volumes:

```console
# Fix permissions for test_videos
sudo chown -R $USER:$USER test_videos/
```

### Rebuild After Code Changes

If you modify dependencies or the Dockerfile:

```console
docker compose build --no-cache
```

### Clean Up

Remove containers and volumes:

```console
# Stop and remove containers
docker compose down

# Remove volumes (WARNING: deletes database data)
docker compose down -v
```

## Docker Image Details

- **Base image**: `deeplabcut/deeplabcut:latest-jupyter`
- **Pre-installed**: DeepLabCut 3.x (PyTorch), Python 3.11, all dependencies
- **No conda setup needed**: Base image provides the environment

## Next Steps

- See [Workflows](./workflows.md) for trained vs pretrained model modes
- See [Testing Guide](./testing.md) for details on running functional tests
- See [Tutorials](./tutorials/) for workflow examples
- See [Concepts](./concepts.md) for architecture details

