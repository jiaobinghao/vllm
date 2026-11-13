# MTP (Multi-Token Prediction)

MTP is a speculative decoding method where the target model includes native
multi-token prediction capability. Unlike draft-model-based methods, you do not
need to provide a separate draft model.

MTP is useful when:

- Your model natively supports MTP.
- You want model-based speculative decoding with minimal extra configuration.

## Gemma 4 Assistant Models

Gemma 4 assistant checkpoints use vLLM's Gemma 4 MTP path. They are not generic
draft models, even though they are passed through the `model` field in
`--speculative-config`.

Use `"method": "mtp"` when serving Gemma 4 with an assistant checkpoint:

```bash
vllm serve google/gemma-4-E2B-it \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --speculative-config '{"method":"mtp","model":"gg-hf-am/gemma-4-E2B-it-assistant","num_speculative_tokens":1}'
```

The E2B, E4B, 12B, 26B-A4B, and 31B Gemma 4 IT assistant checkpoints are supported.
Tower-based variants use `model_type: gemma4_assistant` and the encoder-free
Gemma 4 Unified variant (12B) uses `model_type: gemma4_unified_assistant`.
vLLM maps both to `Gemma4MTPModel` internally and wires the assistant layers
to share KV cache with the target model.

If an older vLLM release logs `SpeculativeConfig(method='draft_model', ...)`
for a Gemma 4 assistant checkpoint, that release is treating the assistant as a
generic draft model and may fail during initialization for multimodal Gemma 4
targets. Upgrade to a version with Gemma 4 MTP support instead.

## Offline Example

The MiMo examples on this page enable `trust_remote_code` to load the checkpoint's
[custom configuration](https://huggingface.co/XiaomiMiMo/MiMo-7B-Base/blob/main/configuration_mimo.py).
Review that code before enabling this option.

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="XiaomiMiMo/MiMo-7B-Base",
    trust_remote_code=True,
    tensor_parallel_size=1,
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 1,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## Online Example

```bash
vllm serve XiaomiMiMo/MiMo-7B-Base \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

## Dynamic MTP by batch size

At high concurrency, MTP's drafting and verification work can cost more than the
decode steps it saves. [Dynamic speculative decoding](dynamic_speculative_decoding.md)
lets you reduce the draft depth as the scheduled batch grows, while retaining a
larger depth for smaller batches.

Choose a batch-size schedule by comparing three configurations with the same
model, hardware, prompt and output lengths, sampling settings, and server limits.
Record the vLLM version and model revision with the results:

1. A baseline without `--speculative-config`.
2. Fixed MTP depths, starting with `num_speculative_tokens=1` and then larger
   values supported by the model.
3. A dynamic schedule chosen from those measurements.

### Dynamic MTP example

The following schedule illustrates reducing the draft depth as the batch grows.
The thresholds are examples, not recommended defaults or measured optima:

```bash
vllm serve XiaomiMiMo/MiMo-7B-Base \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 2048 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 2048 \
    --no-enable-prefix-caching \
    --speculative-config '{
        "method": "mtp",
        "num_speculative_tokens": 3,
        "num_speculative_tokens_per_batch_size": [
            [1, 4, 3],
            [5, 16, 1],
            [17, 64, 0]
        ]
    }'
```

Each inclusive range selects `K` from the number of requests scheduled in an
engine step. This can differ from the client's concurrency limit as requests
arrive, queue, and finish. Set `num_speculative_tokens` to at least the largest
`K` in the table: it caps the effective draft depth.

`K=0` stops proposing speculative tokens for that batch size. It does not unload
the MTP weights or guarantee the same cost as starting without speculative
decoding; the draft path can still perform work to maintain its state. Measure
against the non-speculative baseline as well as fixed MTP depths.

### Measure the tradeoff

For a controlled comparison, restart the server with each configuration, keeping
the other options above unchanged. Disabling prefix caching prevents repeated
warmup prompts from turning measured requests into cache hits. Run a benchmark
such as:

```bash
vllm bench serve \
    --backend vllm \
    --model XiaomiMiMo/MiMo-7B-Base \
    --trust-remote-code \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 128 \
    --random-range-ratio 0.0 \
    --num-prompts 128 \
    --max-concurrency 8 \
    --request-rate inf \
    --num-warmups 16 \
    --temperature 0 \
    --ignore-eos \
    --seed 0 \
    --save-result \
    --result-dir results \
    --result-filename mtp-dynamic-c8-r1.json
```

Repeat at several concurrency levels, including values on both sides of each
schedule boundary, and use a distinct result filename for each configuration
and repetition. Compare output tokens/s and mean and tail time per output token
(TPOT); check time to first token (TTFT), failed requests, and memory usage too.
Repeat measurements and vary their order before choosing thresholds.

Random prompts are useful for a repeatable load test, but their MTP acceptance
can differ from real text. Confirm the selected schedule on representative
prompts, using the [acceptance metrics](acceptance_metrics.md) alongside latency
and throughput. Include a workload that drains from a large batch to a small
one so that the schedule is exercised when `K` increases again.

Dynamic speculative decoding is currently disabled with data parallelism
(`--data-parallel-size > 1`); vLLM falls back to the static
`num_speculative_tokens`. See the dynamic decoding guide's
[limitations](dynamic_speculative_decoding.md#limitations) before deploying a
schedule.

## Notes

- MTP only works for model families that support MTP in vLLM.
- `num_speculative_tokens` controls speculative depth. A small value like `1`
  is a good default to start with.
- If your model does not support MTP, use another method such as EAGLE or draft
  model speculation.
- When the MTP head shares the target's unquantized lm_head (most MTP models),
  `"draft_lm_head_quantization": "fp8"` or `"nvfp4"` in `--speculative-config`
  gives the drafter a quantized copy of that head (Model Runner V2). The target
  still verifies with its own head, so outputs are unchanged; only the
  acceptance length can move. It helps most on bandwidth-bound GPUs with a
  large vocabulary. The copy costs about one byte (FP8) or 0.56 bytes (NVFP4)
  per lm_head weight, e.g. 0.6 or 0.35 GiB for a 248k x 2560 head; it is
  counted in the model weights, so with `--kv-cache-memory-bytes` leave room
  for it.
