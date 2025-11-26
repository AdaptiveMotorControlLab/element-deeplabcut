.PHONY: help build up down shell test-trained test-pretrained clean

help:
	@echo "make build          - Build Docker image"
	@echo "make up             - Start services"
	@echo "make down           - Stop services"
	@echo "make shell          - Interactive shell"
	@echo "make test-trained   - Run trained model test"
	@echo "make test-pretrained - Run pretrained model test"
	@echo "make clean          - Remove volumes"

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

shell:
	docker compose run --rm client -i

test-trained:
	docker compose run --rm client -c "python test_trained_inference.py"

test-pretrained:
	docker compose run --rm client -c "python test_video_inference.py superanimal_quadruped"

clean:
	docker compose down -v
