.PHONY: download-fineweb-baseline train-baseline summarize-baseline-log

# Baseline defaults (override at invocation time as needed)
VARIANT ?= sp1024
TRAIN_SHARDS ?= 10
NPROC_PER_NODE ?= 1
RUN_ID ?= baseline_sp1024
DATA_PATH ?= ./data/datasets/fineweb10B_sp1024/
TOKENIZER_PATH ?= ./data/tokenizers/fineweb_1024_bpe.model
VOCAB_SIZE ?= 1024
LOG_PATH ?= train.log
TRAIN_ENV ?=

download-fineweb-baseline:
	python3 data/cached_challenge_fineweb.py --variant $(VARIANT) --train-shards $(TRAIN_SHARDS)

train-baseline:
	$(TRAIN_ENV) RUN_ID=$(RUN_ID) \
	DATA_PATH=$(DATA_PATH) \
	TOKENIZER_PATH=$(TOKENIZER_PATH) \
	VOCAB_SIZE=$(VOCAB_SIZE) \
	torchrun --standalone --nproc_per_node=$(NPROC_PER_NODE) train_gpt.py

summarize-baseline-log:
	python3 scripts/summarize_train_log.py --log $(LOG_PATH)
