# Dynamic Speculative Decoding

## Why is Dynamic SD needed?

SD methods need to verify K tokens for each sequence during decoding. As BS increases, the effective BS becomes BS\*K which increases the compute requirement during verification. When this BS\*K goes beyond a critical BS then SD negatively impacts the decode speed (TPOT). DSD helps by tuning the K to an optimal value such that we continue to reap the benefits from SD.

## Use cases

* Variable concurrency workload using same deployment. K would decrease as concurrency increases.
* During RL rollout where we start off with high BS but then end up with small BS due to very few long tail request which end up generating a lot of tokens stalling the progress of the current rollout. Here K would go up during the end of rollout.

## `--speculative-config` schema

To use Dynamic SD, add `num_speculative_tokens_per_batch_size` to the config of an SD method as a list of lists. An entry `[start_bs, end_bs, optimal_K]` selects `optimal_K` draft tokens when the number of requests scheduled in an engine step is in the inclusive range `[start_bs, end_bs]`. This batch size can differ from the client's concurrency limit. `num_speculative_tokens` caps the effective draft depth, so set it to at least the largest `optimal_K` in the table. For example:

```bash
--speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'
```

implies that:

* K=3 will be used when the scheduled batch size is in range [1, 64]
* K=1 will be used when the scheduled batch size is in range [65, 128]
* K=0 will be used when the scheduled batch size is in range [129, 512], i.e., no draft tokens will be produced.

## Online Examples

For native MTP models, see the [dynamic MTP example and workload tuning
guide](mtp.md#dynamic-mtp-by-batch-size). Measure fixed depths and a
non-speculative baseline before choosing batch-size thresholds.

### Dynamic SD Eagle Drafter

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'
```

### Dynamic SD Eagle3 Drafter

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "eagle3",
    "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 5,
    "num_speculative_tokens_per_batch_size": [
      [1, 16, 5],
      [17, 32, 4],
      [33, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'

```

## Limitations

* Tested with Eagle, Eagle-3, DFlash, and MTP (`XiaomiMiMo/MiMo-7B-Base`). Other SD methods and model families may or may not work out of the box.
* Not compatible with data parallelism (`--data-parallel-size > 1`). Each DP rank schedules independently, so ranks can pick different K values, causing DP collective divergence and deadlocks. When DP is enabled, vLLM automatically disables `num_speculative_tokens_per_batch_size` and falls back to the static `num_speculative_tokens` value.
