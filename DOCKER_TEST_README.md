# Docker Testing

## Quick Start

```bash
# Start database and container
docker compose up -d

# Run tests
docker compose run --rm client python test_trained_inference.py
docker compose run --rm client python test_video_inference.py superanimal_quadruped

# Or use Makefile
make test-trained
make test-pretrained
```

## Configuration

Create `.env` file (optional):

```env
DJ_PASS=simple
DB_PORT=3306
DATABASE_PREFIX=test_
```

## Volumes

| Mount | Container Path | Description |
|-------|----------------|-------------|
| `./test_videos` | `/app/data` | Test videos |
| `.` | `/app` | Project directory |
| `./dj_local_conf.json` | `/app/dj_local_conf.json` | Database config |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Database connection | `docker compose ps` to check health |
| Permission issues | `sudo chown -R $USER:$USER test_videos/` |
| Code changes | `docker compose build` |
| Clean up | `docker compose down -v` |

## Development

```bash
make shell  # Interactive shell
```
