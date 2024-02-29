include .env
export

SERVICE=pangolin
TAG = us-central1-docker.pkg.dev/spliceai-lookup-412920/docker/pangolin:latest
GCLOUD_PROJECT=spliceai-lookup-412920

all: build push update_sha deploy describe

build:
	docker build -f Dockerfile -t $(TAG) .

push:
	docker push $(TAG)

update_sha:
	docker pull $(TAG) 2>&1 | grep Digest | cut -c 9- > sha256.txt

deploy: build push

	gcloud --project $(GCLOUD_PROJECT) beta run deploy $(SERVICE) \
		--image $(TAG) \
		--concurrency 5 \
		--execution-environment gen2 \
		--region us-central1 \
		--set-env-vars "DB_PASSWORD=${SPLICEAI_LOOKUP_DB_PASSWORD}" \
		--add-volume=name=ref,type=cloud-storage,bucket=spliceai-lookup-reference-data,readonly=true \
		--add-volume-mount=volume=ref,mount-path=/ref \
		--allow-unauthenticated \
		--memory 8Gi \
		--cpu 4

test: build
	docker run -p 8080:8080 $(TAG)

test2: build
	gcloud beta code dev

run: build
	docker run -v ~/ref/:/ref  -it $(TAG) /bin/bash

describe:
	gcloud --project $(GCLOUD_PROJECT) run services describe $(SERVICE) --region us-central1 --format yaml | tee service.backup_copy.yaml

replace:
	gcloud --project $(GCLOUD_PROJECT) run services replace service.backup_copy.yaml
