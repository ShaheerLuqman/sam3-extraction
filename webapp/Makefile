.PHONY: dev backend build serve install

REPO := $(shell cd .. && pwd)
UVICORN := CUDA_VISIBLE_DEVICES=1 SAM3_CHECKPOINT=$(REPO)/checkpoints/sam3.pt \
	$(REPO)/.venv/bin/uvicorn webapp.backend.main:app --host 127.0.0.1 --port 8000 --workers 1

dev:                 ## uvicorn + vite dev server (HMR)
	./run.sh

backend:            ## backend only, no reload (model load is ~1-2 min)
	cd $(REPO) && $(UVICORN)

build:              ## build the frontend to webapp/frontend/dist
	cd frontend && npm run build

serve: build        ## single process: uvicorn serves /api and the built frontend
	cd $(REPO) && $(UVICORN)

install:            ## one-time: backend deps + frontend deps
	VIRTUAL_ENV=$(REPO)/.venv uv pip install fastapi "uvicorn[standard]" python-multipart
	cd frontend && npm install
