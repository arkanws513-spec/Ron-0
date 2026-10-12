# Ron-0 Project Information / Handoff Memory

Last updated: 2026-10-12
Repository: https://github.com/arkanws513-spec/Ron-0
Default branch: `main`

## Purpose
This file is the durable project handoff note for future ChatGPT conversations. Read it before continuing Ron-0 development so the work can resume without relying on chat history. Update it after meaningful repository changes, training results, or decisions.

## User's current request and working style
- The user wants concrete changes made in the repository, not just instructions.
- Communicate in Egyptian Arabic, be direct, and distinguish completed work from work that is still running or unverified.
- Do not claim training or tests passed until the relevant GitHub Actions run confirms it.
- The goal is to improve Ron's original/native core and make training/evaluation more systematic. Centralizing tests does not itself improve model weights; model quality must be checked separately.

## Completed changes on 2026-10-12
Added a central test entry point:
- `scripts/run_all_tests.py`
- Link: https://github.com/arkanws513-spec/Ron-0/blob/main/scripts/run_all_tests.py
- Runs the Python pytest suite, checks key JavaScript files with `node --check`, runs Ron core/browser runtime tests, and checks required repository paths.
- It stops on the first failed command so the calling workflow cannot continue past a failed test gate.

Updated workflows to invoke the central suite:
- `.github/workflows/ci.yml`
- `.github/workflows/train-ron-10m.yml`
- `.github/workflows/train-native.yml`
- `.github/workflows/train-ron-10m-200-eval.yml`
- `.github/workflows/train-ron-100m.yml`
- `.github/workflows/distill-ron-180m-pythia.yml`

Recent commit SHAs (each is on main; later commits may follow):
- Central test runner: `60fb675d8f14ac8d2d0aecc22fed3ac13da8ac10`
- CI workflow: `1c08c54165ab57822221f6ce90bb3951ac9353e8`
- 10M workflow: `01b4c972252a184bc605cfc0f4a2d10c63e32a37`
- Native workflow: `6ecb22fe86357faa7f072f44c95340214dbaadc8`
- Hard 200-case workflow: `dca4773f6f8258c598e1362ac770df407a8c75d0`
- 100M workflow: `7c310bbeeecb947993c99706509ef11c1281b18d`
- 180M distillation workflow: `f0410bda555c755052e77b93806927640fa9fdb0`

## Verification status at the time this note was written
After the changes, GitHub Actions had queued these runs; no pass/fail result had yet been confirmed:
- Ron CI #735: https://github.com/arkanws513-spec/Ron-0/actions/runs/38208156488
- Ron CI #734: https://github.com/arkanws513-spec/Ron-0/actions/runs/38208153992
- Ron 100M free training proof #7: https://github.com/arkanws513-spec/Ron-0/actions/runs/38208153999
- Ron 180M Pythia distillation #3: https://github.com/arkanws513-spec/Ron-0/actions/runs/38208156435

Important: re-check current run statuses and logs before saying the changes passed. These run statuses are historical snapshots, not guaranteed current statuses.

## Important repository files/workflows
- `scripts/run_all_tests.py`: centralized test entry point.
- `ron/model.py`, `ron/model_config.py`, `ron/native_provider.py`: native model/runtime areas.
- `ron/evaluation.py`, `ron/generation_quality.py`: evaluation and generation quality.
- `scripts/evaluate_native_checkpoint.py`: conversational checkpoint evaluation.
- `scripts/promote_native_checkpoint.py`: quality-gated checkpoint promotion.
- `training/eval_cases.json`: evaluation cases.
- `tests/`: broad Python and runtime test suite.
- `.github/workflows/train-ron-10m.yml`: 10M training, SFT, blind conversational evaluation, gated promotion.
- `.github/workflows/train-ron-10m-200-eval.yml`: separate harder 200-case training/evaluation path.
- `.github/workflows/train-native.yml`: native baseline training.
- `.github/workflows/train-ron-100m.yml`: bounded 100M parameter training proof.
- `.github/workflows/distill-ron-180m-pythia.yml`: bounded distillation experiment.
- `.github/workflows/ci.yml`: continuous integration.

## Existing training safety/continuation design (verify current files before changing)
The 10M workflow attempts to restore the last promoted checkpoint, can restore a separate experimental continuation for training, evaluates the candidate, and promotes only when the conversational quality gate and promotion script both succeed. Failed candidates should not replace the approved live checkpoint. Keep this separation intact unless there is a deliberate, reviewed reason to change it.

## Next actions
1. Check the latest Actions runs for the central suite and all workflows modified above; inspect logs for failures.
2. Fix central runner issues if any checked JavaScript file does not exist or any tests fail. Check the actual current repository tree before assuming paths exist.
3. Confirm that every relevant training workflow installs the dependencies needed by `scripts/run_all_tests.py` (including pytest and Node.js where required).
4. Check whether any other workflow that trains or evaluates the native core should also call the central test runner; include it only if compatible with its environment and dependencies.
5. Inspect actual evaluation reports and promotion outcomes separately from the overall workflow conclusion. A workflow success or a completed optimizer run is not, by itself, proof of improved conversational quality or promoted live weights.
6. Keep this file updated with verified run links, outcomes, fixes, and the next step.

## How to resume in a new conversation
Start by opening this file from `main`, then inspect current GitHub Actions and repository files. Treat the live repository and current workflow logs as the source of truth when they disagree with this note.
