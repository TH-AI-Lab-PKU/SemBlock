# Core Code Manifest

This folder contains exactly 20 core code or script files selected for paper
writing and method reconstruction. They are copied under `core_code/` with their
original repository paths preserved.

## Training and Data Processing

1. `llada/train_boundary_segmenter.py`  
   Main boundary-head training entry point.

2. `llada/train_local_boundary_corrector.py`  
   Local correction head training for math/GSM-style boundary recovery.

3. `llada/build_boundary_gate_dataset.py`  
   Dataset builder for boundary-gate supervision.

4. `llada/build_task_conditioned_phase_boundary_jsonl.py`  
   Task-conditioned boundary data construction.

5. `llada/build_leetcode_humaneval_bridge_jsonl.py`  
   Code-line data bridge between LeetCode-style data and HumanEval.

6. `llada/build_targeted_nonhe_aug.py`  
   Targeted non-HumanEval code augmentation logic.

7. `llada/build_local_boundary_correction_dataset.py`  
   Local boundary correction dataset generation.

## Math / GSM8K Line

8. `llada/math_oracle_utils.py`  
   Shared math-head and oracle utilities.

9. `llada/math_oracle_benchmark.py`  
   Math/GSM benchmark runner and scoring logic.

10. `llada/rescore_hendrycks_math_cache.py`  
    Offline MATH rescoring / cache post-processing.

11. `llada/run_hendrycks_math_b016_b064_20260516.sh`  
    MATH B0=16/B0=64 run script.

## Code / HumanEval Line

12. `llada/analyze_humaneval_failures.py`  
    HumanEval failure analysis logic.

13. `llada/postprocess_humaneval.py`  
    HumanEval generation post-processing.

14. `llada/run_opencompass_llada_1p5_humaneval_b016_b064_20260519.sh`  
    Final LLaDA-1.5 HumanEval B0=16/B0=64 OpenCompass runner.

15. `llada/opencompass_llada_1p5_humaneval_b16_confidence.py`  
    LLaDA-1.5 HumanEval B0=16 config.

16. `llada/opencompass_llada_1p5_humaneval_b64_confidence.py`  
    LLaDA-1.5 HumanEval B0=64 config.

## NL / GUM / IFEval Line and Ablations

17. `llada/run_ifeval_gsm8k_queue_20260516.sh`  
    IFEval/GSM8K B0 queue runner.

18. `llada/experiments/semantic_boundary_indep/20260518_ifeval_boundary_naturalness_curve/summarize_ifeval_boundary_curve.py`  
    IFEval natural-boundary / GUM semantic-head summary script.

19. `llada/experiments/semantic_boundary_indep/20260518_ifeval_semantic_degradation_curve/summarize_ifeval_semantic_degradation_curve.py`  
    IFEval boundary perturbation / degradation summary script.

20. `llada/experiments/semantic_boundary_indep/20260519_gsm8k_sota_boundary_ablation/summarize_gsm8k_sota_boundary_ablation.py`  
    GSM8K SOTA boundary-ablation summary script.
