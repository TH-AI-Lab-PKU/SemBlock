# GSM8K SOTA Boundary Ablation

All runs use the GSM8K SOTA B=32 decoding setup unless noted: LLaDA-8B-Instruct, `semantic_hybrid`, `steps=16`, `gen_length=512`, cache on, math head threshold 0.60, newline delimiter threshold 0.3, and `gsm8k_landing_control=true`.

| Variant | Boundary signal | Limit | Strict EM | Flexible EM | Delta vs SOTA | Avg NFE | Avg blocks | Avg block len | Note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sota_hybrid | math head + NL delimiter | 300 | 0.8533 | 0.8533 | 0.0000 | 104.48 | 26.87 | 19.06 | Existing SOTA limit-300 reference |
| delimiter_only_adablock | NL delimiter only | 300 | 0.8233 | 0.8300 | -0.0300 | 102.75 | 24.94 | 20.53 | Remove learned math boundary head |
| math_head_only | math head only | 300 | 0.8367 | 0.8400 | -0.0167 | 112.08 | 34.93 | 14.66 | Remove delimiter confidence from the hybrid score |
