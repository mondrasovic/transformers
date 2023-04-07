docker_build:
	docker build \
		--build-arg SRC_DIR=/transformers \
		-t mond/tf-transformers \
		-f Dockerfile \
		.

docker_run:
	docker run \
		--gpus all \
		--rm \
		-it \
		-v ${shell pwd}:/pandemic_calc \
		-v /home/mond/datasets/pt_to_en_translation:/dataset \
		-e "DATASET_DIR=/dataset" \
		-e "TF_CPP_MIN_LOG_LEVEL=2" \
		mond/tf-transformers \
		bash

show_model:
	@python ./src/show_model.py

train:
	@python ./src/train.py

evaluate:
	@python ./src/evaluate.py

test:
	@pytest ./src/tests

reformat:
	@isort --line-length 100 .
	@black --line-length 100 .

clean:
	@find . -name "__pycache__" | xargs rm -rf
	@find . -name ".pytest_cache" | xargs rm -rf
	@find . -name ".mypy_cache" | xargs rm -rf