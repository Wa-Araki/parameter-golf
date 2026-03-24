<img width="3840" height="1280" alt="1920x640-discord" src="https://github.com/user-attachments/assets/90607b26-171f-476a-90ae-69b9dbb7cb30" />

<br>
<br>

**OpenAI Model Craft Challenge: Parameter Golf** is a challenge to train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s, evaluated by compression on the FineWeb validation set (tokenizer-agnostic, bits per byte).

This challenge is heavily inspired by the [NanoGPT Speedrunning](https://github.com/KellerJordan/modded-nanogpt) challenge, where participants compete to train a model that reaches 3.28 FineWeb validation loss as quickly as possible. We're excited to see how optimizing for a parameter-constrained setting pushes people toward unique architectures (test-time compute, aggressive parameter tying, depth recurrence, low-rank training, ...), compression schemes (low precision, QAT, bitnets, novel tokenizers, ...), and other creative submissions (test-time training, long context, megakernels ...). 

If you're familiar with [neural scaling laws](https://arxiv.org/abs/2001.08361), you can consider this challenge a form of L(N) optimization, where the objective is to optimize the lowest loss given a fixed number of parameters (N) unconstrained by data, compute, steps, or architecture. Challenges like the [NanoGPT Speedrun](https://github.com/KellerJordan/modded-nanogpt), which optimizes for a form of L(T) (~lowest time given constrained loss) or the [NanoGPT Slowrun](https://github.com/qlabs-eng/slowrun), which optimizes for L(D) (lowest loss given constrained dataset size), can be thought of as equivalent challenges in this family.

Ideally, we'd allow for submissions to use arbitrary computational resources. But in order to make the challenge not inaccessibly expensive, we're limiting *leaderboard submissions* to 10 minutes on 8xH100s. However, we'd still love to see submissions that don't meet the compute limitation requirements in our 'Non-record Submissions' section: We're excited to see people push the infinite frontier of parameter limited performance as well.

We also know compute is expensive, so **OpenAI is sponsoring $1,000,000 in compute credits** to help people get started training their models. To request a compute grant, use this form: [Request a Compute Grant](https://openai.com/index/parameter-golf/#credit-form).
When requesting compute, please make sure you choose the appropriate level, write sufficient justification, and **submit with an email tied to a OpenAI / ChatGPT account**.

## 最初の実行ガイド（日本語）

### まず何をするリポジトリか
- このリポジトリは、**16MB 以下に圧縮できる言語モデル**を学習し、FineWeb 検証セットでの圧縮性能を比較するための実験基盤です。
- 最小構成では、`Makefile` の 3 ターゲット（データ取得 / 学習 / ログ要約）を順番に実行するだけで、baseline 再現と結果確認まで到達できます。
- 学習本体は `train_gpt.py`、データ取得は `data/cached_challenge_fineweb.py`、結果要約は `scripts/summarize_train_log.py` が担当します。
- 比較の主指標は **`val_bpb`（低いほど良い）** です。補助的に **`val_loss`** と **圧縮後サイズ（`Serialized model int8+zlib` / `Total submission size int8+zlib`）** を見ます。
- baseline 実験ハーネスとして、`make download-fineweb-baseline` → `make train-baseline` → `make summarize-baseline-log` の導線が整備されています。
- 2 本目の改善タスクとして、**baseline 近傍の差分実験**（例: `LR_WARMUP_ITERS` のみ変更して A/B 比較）を、同じハーネスで実行できます。
- まずは「baseline が再現できること」を優先し、そこから 1 要素ずつ改善を試すのが安全です。
- 現段階で最初からやらないこと: 蒸留・新規アーキテクチャ大改造・トークナイザ変更など、比較条件を大きく崩す改変は初回では避けます。

### 最短の実行手順
1. **環境確認**
   - 実行コマンド
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     python -m pip install --upgrade pip
     pip install -r requirements.txt
     ```
   - 何が起きるか  
     仮想環境を作成し、`train_gpt.py` とデータ取得・要約スクリプトに必要な依存を入れます。
   - 成功したら何が見えるか  
     `(.venv)` プロンプトになり、`pip install` がエラーなく完了します。
   - 失敗時にまず疑う点  
     Python バージョン差異、CUDA/PyTorch 環境不一致、依存インストール途中のネットワーク不調。

2. **データ取得**
   - 実行コマンド
     ```bash
     make download-fineweb-baseline
     ```
     （必要なら例: `TRAIN_SHARDS=80 make download-fineweb-baseline`）
   - 何が起きるか  
     `data/cached_challenge_fineweb.py` が `sp1024` 前提でデータを取得し、学習用データとトークナイザを配置します。
   - 成功したら何が見えるか  
     `./data/datasets/fineweb10B_sp1024/` と `./data/tokenizers/fineweb_1024_bpe.model` を学習コマンドが参照可能な状態になります。
   - 失敗時にまず疑う点  
     ネットワーク、ディスク容量不足、`TRAIN_SHARDS` 指定ミス、途中中断でデータが不完全なままになっていること。

3. **baseline 実行**
   - 実行コマンド
     ```bash
     make train-baseline
     ```
     （軽い疎通確認: `NPROC_PER_NODE=1 RUN_ID=baseline_smoke TRAIN_ENV="ITERATIONS=200 MAX_WALLCLOCK_SECONDS=0" make train-baseline`）
   - 何が起きるか  
     `torchrun` 経由で `train_gpt.py` が起動し、学習ログ（`train_loss` など）と最終評価（`val_loss`, `val_bpb`, 圧縮サイズ）を標準出力へ出します。
   - 成功したら何が見えるか  
     終盤に `final_int8_zlib_roundtrip_exact ... val_bpb:...` とサイズ情報が出て、run が正常終了します。
   - 失敗時にまず疑う点  
     GPU 数と `NPROC_PER_NODE` の不整合、`DATA_PATH` / `TOKENIZER_PATH` の参照先不一致、VRAM 不足、`torchrun` 実行環境の問題。

4. **改善版の実行（2 本目の改善タスク）**
   - 実行コマンド（README 既存導線に合わせた最小差分）
     ```bash
     # Baseline
     NPROC_PER_NODE=1 RUN_ID=baseline_ref make train-baseline | tee baseline_ref.log

     # 改善版（LR warmup のみ追加）
     NPROC_PER_NODE=1 RUN_ID=baseline_lr_warmup_200 \
     TRAIN_ENV="LR_WARMUP_ITERS=200" \
     make train-baseline | tee baseline_lr_warmup_200.log
     ```
   - 何が起きるか  
     baseline と改善版を同条件で 2 本走らせ、差分を `LR_WARMUP_ITERS` のみに限定した A/B ができます。
   - 成功したら何が見えるか  
     2 本のログファイルが生成され、後段の要約で `lr_warmup_iters` を含む実行条件差分を確認できます。
   - 失敗時にまず疑う点  
     ログ保存先の権限、シェル改行（`\`）の崩れ、`TRAIN_ENV` のクオート漏れで環境変数が反映されないこと。

5. **ログ要約 / 結果確認**
   - 実行コマンド
     ```bash
     make summarize-baseline-log LOG_PATH=baseline_ref.log
     make summarize-baseline-log LOG_PATH=baseline_lr_warmup_200.log
     ```
   - 何が起きるか  
     `scripts/summarize_train_log.py` がログを正規表現で解析し、`final.val_bpb`・`artifact.total_submission_bytes`・実行条件を JSON で出力します。
   - 成功したら何が見えるか  
     比較に必要な値（`final`, `artifact`, `run_conditions`）が JSON でまとまって表示されます。
   - 失敗時にまず疑う点  
     `LOG_PATH` ミス、学習途中停止で最終行が欠損、ログ形式が想定外で必要キーが `null` になっていること。

### 何をチェックすべきか
- データ配置:
  - `./data/datasets/fineweb10B_sp1024/` が存在するか
  - `./data/tokenizers/fineweb_1024_bpe.model` が存在するか
- 学習完走:
  - 学習コマンドが異常終了していないか
  - 最終評価行（`final_int8_zlib_roundtrip_exact ...`）まで出ているか
- ログ:
  - `tee` した `train.log` / `baseline_ref.log` / `baseline_lr_warmup_200.log` があるか
- 指標:
  - `val_bpb` が取得できているか（主指標）
  - `val_loss` が取得できているか（補助指標）
  - `Total submission size int8+zlib` が確認できるか（16,000,000 bytes 制約）
- baseline と改善版の比較で最初に見る点:
  - まず `final.val_bpb`（改善有無）
  - 次に `artifact.total_submission_bytes`（制約違反がないか）
  - その後 `run_conditions` を見て、公平比較（差分が意図した 1 点のみ）になっているか

### 実行の全体像
- 流れは **データ取得 → 学習 → 評価 → 要約** です。
- データ取得: 学習と評価に必要な FineWeb キャッシュと tokenizer をそろえる段階です。
- 学習: `train_gpt.py` を実行して重みを更新し、最後に圧縮評価まで行います。
- 評価: `val_loss` / `val_bpb` と圧縮サイズを出し、提出可能性を判定します。
- 要約: ログから比較可能な JSON を作り、baseline と改善版の差を機械的に確認します。
- 初回はスコア改善よりも、**baseline を同じ手順で再現し、指標を取得できること**を完了条件にしてください。

### この README の読み方
- 最初はこの「最初の実行ガイド（日本語）」だけを読み、上の 1〜5 をそのまま実行してください。
- 次に、`## Getting Started` の **Quick Baseline Reproduction (CUDA)** と **Baseline-near Differential Experiment** を読みます。
- クラウド実行（Runpod）や提出ルールなどの詳細は、README 後半の既存セクションを参照してください。

## Participant Form

If you enjoy solving very difficult technical problems, please introduce yourself via the [Challenge Participant Form](https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf). It helps us attribute challenge submissions and reach out about opportunities with OpenAI. _Completing the form is not required to participate._

Many researchers at OpenAI first distinguished themselves through elite mathematics and programming competitions. The Model Craft Challenge is designed in that spirit: testing the ability to tackle unfamiliar problems with creativity and rigor, qualities we believe are essential for frontier AI research.

In June, we plan to hire a small cohort of early-career researchers, targeting current undergraduate students and recent graduates, including Olympiad medalists and elite competitors. For exceptional participants, the challenge may also serve as a way to stand out to OpenAI researchers and recruiters.

The challenge runs from March 18th to April 30th. 

Happy training!

## Leaderboard

| Run | Score | Author | Summary | Date | Info |
|-----|------:|--------|---------|------|------|
| 11L EMA + GPTQ-lite + warmdown3500 | 1.1228 | signalrush | On PR #374: GPTQ-lite clip search + EMA, plus warmdown3500 and QAT@0.15 | 2026-03-22 | [info](records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md) |
| 11L Partial RoPE + LN Scale + EMA + XSA4 | 1.1248 | jfprincz | On PR #287: Partial RoPE (16/64) + layerwise LN scale | 2026-03-21 | [info](records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md) |
| 11L XSA4 + EMA + Int6 MLP3x | 1.1271 | jfprincz | On PR #198: XSA on the last 4 layers + EMA replacing SWA | 2026-03-20 | [info](records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md) |
| 11L Efficient Partial XSA | 1.1307 | unnir | On PR #198: Efficient Partial XSA on the deepest 3 layers | 2026-03-20 | [info](records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/README.md) |
| 10L Int5-MLP + BigramHash(10240) | 1.1428 | thwu1 | 10 layers, mixed int5/int6 quantization, BigramHash(10240), SWA(0.4), WD=0.04 | 2026-03-20 | [info](records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md) |
| Int6 MLP3x + SmearGate + BigramHash | 1.1458 | Raahil Shah | 3x MLP + SmearGate + BigramHash + OrthoInit + Muon WD + SWA | 2026-03-20 | [info](records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/README.md) |
| 11L MLP3x + Int6 QAT | 1.1502 | aruniyer | 11 layers, 3x MLP, int6 QAT, zstd-22, WD=0.04, sliding eval | 2026-03-20 | [info](records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/README.md) |
| SmearGate + OrthoInit + Muon WD | 1.1556 | aquariouseworkman | SmearGate + BigramHash + 3x MLP + int6 STE QAT + sliding eval | 2026-03-19 | [info](records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/README.md) |
| 10L Int6 QAT + Zstd MLP2.6x | 1.1586 | yahya010 | 10 layers, int6 QAT + zstd-22, MLP 1344, Muon 0.99, sliding eval | 2026-03-19 | [info](records/track_10min_16mb/2026-03-19_Seq2048_FP16Emb_TunedLR/README.md) |
| Mixed Quant + Sliding Window Eval | 1.1630 | aquariouseworkman | Int6 block weights + int8 embeddings + 3x MLP + sliding eval | 2026-03-19 | [info](records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/README.md) |
| Muon WD + 10 layer | 1.1748 | notapplica | Includes prev. wins + Spectral embed init + resid mix | 2026-03-19 | [info](records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/README.md) |
| Sliding Window Eval | 1.1925 | Matthew Li | Sliding window evaluation at stride=64, increasing context for eval | 2026-03-19 | [info](records/track_10min_16mb/2026-03-19_SlidingWindowEval/README.md) |
| Lora TTT | 1.1928 | samacqua | Test-time training with LORAs | 2026-03-19 | [info](records/track_10min_16mb/2026-03-17_LoRA_TTT/README.md) |
| 4k seq length| 1.2014 | Spokane Way | 4k seq length + better hypers | 2026-03-19 | [info](records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/README.md) |
| 2048 seq length | 1.206 | Spokane Way | 2048 seq length (train + val) | 2026-03-18 | [info](records/track_10min_16mb/2026-03-18_LongContextSeq2048/README.md) |
| int6 mixed precision | 1.2147 | Nan Liu | 10 layers, mixed int8/int6 | 2026-03-18 | [info](records/track_10min_16mb/2026-03-19_10L_MixedPrecision/README.md) |
| fp16 Embed | 1.2197 | Renier Velazco | FP16 Tied Embedding + LR/Warmdown Tuning | 2026-03-18 | [info](records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md) |
| Naive Baseline | 1.2244 | Baseline | 9layer 512dim 1024vocab TiedEmbeddings 4 KV heads | 2026-03-18 | [info](records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md) |

#### Notable Non-Record Runs

| Run | Score | Author | Summary | Date | Info |
|-----|------:|--------|---------|------|------|
| 4-Hour Baseline | 1.2074 | Will DePue | Testing unlimited compute, 4 hours on 8xH100 | 2026-03-18 | [info](records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md) |

## Getting Started

### Quick Baseline Reproduction (CUDA)

If you want the shortest path to rerun the baseline on a CUDA machine, use the Make targets below.

1) **Setup**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) **Download FineWeb baseline data (1 command)**
```bash
make download-fineweb-baseline
```

By default this fetches tokenizer variant `sp1024` with `TRAIN_SHARDS=10` (first 1B train tokens) plus full validation.  
To change shard count, override at runtime (example: `TRAIN_SHARDS=80 make download-fineweb-baseline`).

3) **Launch baseline training (1 command)**
```bash
make train-baseline
```

Useful overrides for quick iteration:
```bash
NPROC_PER_NODE=1 RUN_ID=baseline_smoke TRAIN_ENV="ITERATIONS=200 MAX_WALLCLOCK_SECONDS=0" make train-baseline
```

4) **Summarize results from train.log (1 command)**
```bash
make summarize-baseline-log LOG_PATH=path/to/train.log
```

This prints JSON with:
- final `val_bpb` / `val_loss` (from `final_int8_zlib_roundtrip_exact` when present)
- key run conditions (dataset shards, world size, train tokens, seq_len, iterations, seed, tokenizer path)
- compressed model size (`Serialized model int8+zlib`) and total submission bytes.

### Baseline-near Differential Experiment (small, low-risk)

If you want to run a **single small ablation near baseline** (without changing architecture/tokenizer/data), use this checklist.

Candidate ideas (all baseline-near and low complexity):

| Idea | What changes | Why it might help | Cost | Risk |
|---|---|---|---|---|
| A. LR warmup in main training | Add short linear LR ramp for first N train steps (`LR_WARMUP_ITERS`) | Reduces early-step instability / overshoot with large baseline LRs | Very low (1 env var + scheduler branch) | Low |
| B. Warmdown length tweak | Change `WARMDOWN_ITERS` (e.g. 1200 → 1600) | Smoother final optimization near wallclock cap | Very low | Low/medium (can under-train if too long) |
| C. Sequence length tweak | Change `TRAIN_SEQ_LEN` (e.g. 1024 → 1536/2048) | More context per example can improve validation compression | Low/medium (retune throughput) | Medium (tokens/sec and memory tradeoff) |

Recommended first differential run (lowest risk + easiest A/B):
- **Choose A: main-training LR warmup only**.
- Keep everything else equal to baseline.

Example commands (baseline vs diff):

```bash
# Baseline
NPROC_PER_NODE=1 RUN_ID=baseline_ref make train-baseline | tee baseline_ref.log

# Differential run: only LR warmup changed
NPROC_PER_NODE=1 RUN_ID=baseline_lr_warmup_200 \
TRAIN_ENV="LR_WARMUP_ITERS=200" \
make train-baseline | tee baseline_lr_warmup_200.log

# Summaries for comparison
make summarize-baseline-log LOG_PATH=baseline_ref.log
make summarize-baseline-log LOG_PATH=baseline_lr_warmup_200.log
```

Metrics to compare vs baseline:
- Primary: `final.val_bpb` (lower is better).
- Secondary quality: `final.val_loss`.
- Constraint checks: `artifact.total_submission_bytes` (must stay under 16,000,000 bytes).
- Fairness checks: ensure run conditions match (seed/world size/iterations/train_batch_tokens/train_seq_len), with only `lr_warmup_iters` changed.

### Training Your First Model (Mac with Apple Silicon)

If you have an Apple laptop or desktop with Apple Silicon, we've set up a simple MLX training script to help you start iterating locally.

If you don't have a Mac with Apple Silicon, you can run an adapted version of this script without MLX support. Just ask [Codex](https://openai.com/codex/) to refactor it; the change is straightforward. It may still be fairly slow, so we recommend jumping straight to cloud GPUs with Runpod.

First, clone the repository, create a fresh Python environment, and install the packages needed for the MLX path plus dataset download:

```bash
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
```

Download our cached version of FineWeb with the 1024-token vocabulary:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

This populates `./data/datasets/fineweb10B_sp1024/` and `./data/tokenizers/`.
By default this downloads the full validation split plus 80 training shards (8B tokens). For a smaller local smoke subset, pass `--train-shards 1`, for example `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1`.

Then run a small MLX training job:

```bash
RUN_ID=mlx_smoke \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py
```

Validation always runs on the full `fineweb_val_*` split, which is the fixed first-50k-document set. The smoke command above skips periodic validation and just prints the final `val_loss` and `val_bpb` once at the end.

### Scaling Up to a Remote Machine

Once you're happy with your local tests, or you want more compute, switch to a remote CUDA machine.

You can rent GPUs from anywhere, but OpenAI is partnering with Runpod to make setup as easy as possible.  

#### Launching a 1xH100 Pod

1. First, [create a Runpod account](https://console.runpod.io/deploy). You should also set up an SSH key in the Settings tab on the left so you can connect to your remote machine. If you're new to this, ask Codex to help you set it up.

2. Once you've set up your account, create a new GPU Cloud Pod. You can choose whichever GPU SKU you'd like. Final leaderboard submissions must run in under 10 minutes on 8xH100s (specifically the SXM variant), but we strongly recommend testing and running experiments on cheaper SKUs first, since an 8xH100 box can cost around $20/hour.

3. Let's start with a 1xH100 pod. Deploy using the official Parameter Golf template: [Launch Template](https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th). Enable SSH terminal access, leaving the other settings at their defaults. Deploy your pod and SSH into it once it's up. You should land in `/workspace/`.

On your remote machine, clone the repo onto local disk. All Python dependencies are already pre-installed in the image.

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
```

Download our cached version of FineWeb. We'll use the 1024-token vocabulary for now.

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

This defaults to the full validation split plus 80 training shards (8B tokens). If you only want a smaller subset while iterating, pass `--train-shards N`, for example `--train-shards 1`.

Launch your first training run. Note that we're passing `nproc_per_node=1` because we're running on a single H100 GPU in this case.

```bash
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

By default, `train_gpt.py` keeps its ~10 minute wallclock cap. If you want a longer run, override it explicitly, for example `MAX_WALLCLOCK_SECONDS=0`.

By default, this command prints `train_loss` step logs during training and prints `val_loss`, `val_bpb`, and compressed model size in the final `final_int8_zlib_roundtrip` lines at the end. If you want periodic validation logs during the run, set `VAL_LOSS_EVERY`, for example `VAL_LOSS_EVERY=200`. For the baseline config, the final `val_bpb` should land around ~1.2 with a compressed model size under 16MB.

For dataset export, tokenizer export, and docs-cache rebuild instructions, see [data/README.md](data/README.md).

Evaluation will be in the RunPod environment with all packages installed. `requirements.txt` is provided as a reference if you want to self-setup.

## FAQ

**What exactly counts toward the 16MB artifact size?**

The submission artifact is computed as code bytes plus compressed model bytes. All counted code should live in the `train_gpt.py` script.
The cap is decimal 16MB, i.e. 16,000,000 total bytes, not 16 MiB / 16,777,216 bytes.
No external downloads, training dataset access, or network calls are allowed during evaluation. The artifact must be fully self-contained and reproducible.

**Are scores independently verified by OpenAI?**

We're not automatically verifying every submission, but we will verify the top leaderboard entries over time. Any non-reproducible results can be disqualified, and issues reproducing submissions should be raised on the PR. If you find an issue with a record on the leaderboard or find a record isn't reproducible, please let us know and add an Github Issue describing your findings.

**What counts as 'external compute'? For example, is it fair to tune my hyperparameters offline?**

There's no perfectly clear answer here and it's hard to draw a clean line around what does or does not count as external compute. For now, we're reserving the right to disqualify runs that are not in the spirit of the challenge. Tuning your Adam hyperparameters across a bunch of runs is fine, but if there's evidence that you're sneaking in additional compute unfairly, such as brute-forcing ridiculous seeds, we won't allow it. Use your best judgment and there's no penalty for asking questions.

**What are the restrictions on evaluation?**

We won't accept submissions that take more than 10 minutes on 8xH100 to evaluate (Note: This limit is in addition to the 10 minutes of training time allowed!), but otherwise you're free to evaluate however. As with modded-nanogpt, we allow evaluation at any sequence length. And, obviously, you aren't allowed to access any training data during evaluation, unless you pay for those bits in the <16MB limit. We encourage competitors to push the bounds of evaluation methods as aggressively as with training methods. You CANNOT access validation data during training, e.g. by compressing it into your 16mb with "paid prefix".

If it isn't abundantly obvious: You can't cheat on your test loss. You can't cheat by training on the validation set before you evaluate on the validation set. The validation language around test-time training has been confusing people: you are only allowed to test-time train on validation set tokens _you've already evaluated your model on_, since those tokens have already been graded!

**What is the process for accepting new submissions?**

Since all submissions are public, we're accepting record submissions chronologically depending on their PR creation time. The leaderboard may take time to update due to verification and review of submissions, so pay consideration to what the current SOTA PR is when submitting. As explained below, submissions should exceed the SOTA record with sufficient statistical significance in order to be accepted for the leaderboard. Otherwise, submissions may be accepted as 'non-record submissions' given they are sufficiently unique or interesting.

**Can I import XYZ package or library?**

Yes, you're free to import any package or library you want, so long as it does not unjustly violate the rules on evaluation, compute, training time, code size or otherwise. Just include a requirements.txt in your records folder and mention setup instructions in your README.md. Since you don't pay for bits imported in Python libraries, limitations clearly apply: You can't sneak in extra compute, capabilities, or massively increase effective code size with custom libraries, but importing FlashAttention, etc. is completely fine.


## Submission Process

New SOTA records must fulfill the following criteria:

1. They must beat the existing SOTA by at least 0.005 nats. As in modded-nanogpt, because of inter-run variance all submissions must provide enough run logs to show at `p < 0.01` that they achieved the required 0.005-nat improvement. For submissions that improve speed through systems optimization without changing the ML, this requirement is waived.

2. If changes are made to the tokenizer or dataset, prove with certainty that the val_bpb is correctly calculated. Submissions that edit the tokenizer will be examined much more carefully, since bugs may unjustly improve your score.

3. Reproducibly run in under 10 minutes on 8xH100s.

All submissions should be made as a pull request that only adds a new folder to the appropriate `/records` subfolder and includes the following files. Submissions without the full set of requirements will not be accepted.

1. A README.md file that explains the submission in reasonable detail.

2. A `submission.json` file (see the example runs) that includes your name, GitHub ID, `val_bpb`, and related metadata.

3. A train log, automatically produced by your script. Please demonstrate a statistically significant win. Most often, submitting an average over 3 training runs is sufficient.

4. A `train_gpt.py` script and any other dependencies. Note: this must successfully compile and run within the records folder. Broken scripts will not be accepted.

### Non-record Submissions

Submissions are also open to unique and interesting approaches that might not beat the existing SOTA, but still satisfy the 16MB artifact limit. We strongly encourage participants to submit implementations for weird or out-of-the-box ideas, in-progress or unoptimized solutions, so long as they run successfully, or even interesting negative results. We're excited to see what you come up with. We'll still maintain a high bar for non-record submissions, so be sure to justify your ideas and results in detail when submitting.

We also accept non-record submissions to an unlimited compute track for runs that are not intended to meet the 10-minute cutoff. Just note as such in your README file.

Non-record submissions should be made in the same fashion as SOTA records, as described above.

#### PRs on Core Code

The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but the best models should stay in the `/records` folder.

## Support


Join the [OpenAI Discord server](https://discord.com/invite/openai) and visit the Parameter Golf channels (#parameter-golf-discussions, #parameter-golf-announcements) and ask questions.

This repository adapts code from `modded-nanogpt`, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.
