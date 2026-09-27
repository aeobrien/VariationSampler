# Variation Sampler — Development Workflow

This document defines the exact development process. It is the process contract.

This project has two operating modes: **supervised development sessions** (building and modifying code with the project owner present) and **autonomous iteration loops** (the system running unattended, tuning hyperparameters, generating reports). The workflow defines both modes and the handoffs between them.

---

## Mode 1: Supervised Development Sessions

These are interactive Claude Code sessions where the project owner is present. All code changes, architecture decisions, and infrastructure work happen here.

### Session Start Protocol

1. Read `docs/session-log.md` to understand current state
2. Read `ROADMAP.md` to identify the current phase and task
3. If resuming after an autonomous loop batch: read the latest report(s) in `reports/` and any listening notes in `docs/listening-notes/`
4. Check git status — ensure correct branch, clean working tree
5. Run tests to confirm everything works before making changes
6. Begin work on the current task

### Development Cycle

Every piece of work follows:

1. **Branch** — Create `phase-N/description` from main
2. **Build** — Implement the task following CLAUDE.md conventions
3. **Test** — Write tests, run them, verify all pass
4. **Verify** — For audio-producing code: generate sample outputs, confirm they are listenable and reasonable
5. **Commit** — Atomic commits with clear messages
6. **Report** — Summarise what was built, any issues, and what's next

### Session End Protocol

1. Ensure all work is committed
2. Update `docs/session-log.md` with:
   - What was completed
   - Test results
   - Issues encountered and resolutions
   - What the next session should start on
   - If an autonomous loop should be kicked off: what config to use and how many iterations
3. Merge to main if phase/task is complete

---

## Mode 2: Autonomous Iteration Loop

The automated loop runs on a GPU VM without the project owner present. It trains, evaluates, consults Claude for hyperparameter adjustments, and produces reports. It does NOT modify code — only config values.

### Loop Mechanics

```
┌─────────────────────────────────────────────────────────┐
│                    ITERATION N                           │
│                                                         │
│  1. Load config (JSON)                                  │
│  2. Train / continue training                           │
│  3. Generate test samples from held-out inputs          │
│  4. Compute all metrics                                 │
│  5. Write iteration report (JSON + audio examples)      │
│  6. Send report to Claude API                           │
│  7. Claude returns new config JSON (hyperparams only)   │
│  8. Apply guardrail checks:                             │
│     - Auto-rollback if key metrics regressed            │
│     - Hard stop if N iterations without improvement     │
│     - Hard stop if M total iterations reached           │
│  9. If not stopped → next iteration                     │
│  10. If stopped → write batch summary report            │
└─────────────────────────────────────────────────────────┘
```

### What Claude May Change (Automated Loop)

Only these config values. Nothing else.

| Parameter | Description |
|---|---|
| `mask_p_attack` | Mask probability for attack frames |
| `mask_p_tail` | Mask probability for tail/sustain frames |
| `editable_codebooks` | Which RVQ codebooks are eligible for editing (e.g. [6,7,8]) |
| `temperature` | Sampling temperature for token generation |
| `top_p` | Nucleus sampling threshold |
| `k_candidates` | Number of candidates to generate before acceptance filtering |
| `acceptance_band_low` | Lower bound of perceptual distance acceptance band |
| `acceptance_band_high` | Upper bound of perceptual distance acceptance band |
| `learning_rate` | Training learning rate |
| `batch_size` | Training batch size |
| `attack_frames` | Number of frames treated as "attack" region |

### What Claude May NOT Change (Automated Loop)

- Source code files
- Model architecture
- Loss function
- Data pipeline
- Evaluation metric implementations
- Splitting protocol
- Anything not in the config JSON

### Stopping Conditions

The loop stops automatically when ANY of these are true:

1. **Iteration cap reached:** M total iterations in this batch (default: 10)
2. **Stagnation:** N consecutive iterations with no improvement on primary metrics (default: 3)
3. **Regression:** Any key metric regresses past a hard threshold (defined per-metric in config)
4. **Error:** Training fails, OOM, or any unrecoverable error

When the loop stops, it writes a **batch summary report** (see below).

### Auto-Rollback

If an iteration's metrics regress past defined thresholds:
1. Revert config to the last "good" state
2. Log the failed config and why it was rolled back
3. Count toward stagnation counter
4. Continue with reverted config

---

## Handoff: Autonomous Loop → Supervised Session

When the autonomous loop stops, it produces a **batch summary report**. This is the primary artifact for the project owner to review and pass to Claude in the next supervised session.

### Batch Summary Report Contents

The report must contain everything needed to understand what happened and decide what to do next. It is written to `reports/batch-NNN-summary.json` and `reports/batch-NNN-summary.md` (human-readable version).

**Report sections:**

1. **Batch metadata**
   - Batch ID, start/end timestamps, number of iterations run
   - Stopping reason (cap, stagnation, regression, error)
   - Git commit at batch start

2. **Config trajectory**
   - Starting config → each iteration's config changes → final config
   - Which changes improved metrics, which were rolled back

3. **Metric trajectory**
   - Per-iteration values for all key metrics (similarity, variation, artifact scores)
   - Trend direction for each metric across the batch
   - Comparison to real round-robin baselines (are we in-band? approaching? diverging?)

4. **Best and worst outputs**
   - File paths to the 5 best-scoring and 5 worst-scoring generated variations from the batch
   - Corresponding input samples for A/B comparison
   - Per-sample metric breakdown

5. **Claude's diagnoses**
   - The full diagnosis and proposed changes from each iteration's Claude API call
   - Which proposals were accepted vs rolled back

6. **Recommended next actions**
   - Claude's assessment of what to try next (from the final iteration's response)
   - Whether the current trajectory suggests code/architecture changes are needed vs more hyperparameter tuning

7. **Listening pack**
   - A curated set of audio files for the project owner's listening check-in
   - Organised as: `listening-pack/batch-NNN/{input_name}/{input.wav, var_01.wav, var_02.wav, ..., rapid_sequence.wav}`
   - The `rapid_sequence.wav` is the machine-gun test: all variations played as 16th notes at 120 BPM

### Project Owner's Check-In Process

After reviewing the batch report:

1. **Listen** to the listening pack (20–30 minutes). Focus on:
   - Does the best output sound like a different hit, or a processed version of the same hit?
   - Does the worst output reveal a systematic failure mode?
   - Are variations "interesting" like real round-robins, or mechanical/algorithmic?

2. **Write listening notes** in `docs/listening-notes/batch-NNN.md`:
   ```markdown
   # Listening Notes — Batch NNN

   ## Date
   YYYY-MM-DD

   ## Overall Impression
   [1–3 sentences: gut feeling]

   ## Best Outputs
   [What sounds good, which samples, what qualities]

   ## Worst Outputs
   [What sounds wrong, which samples, what failure modes]

   ## Specific Observations
   - [observation 1]
   - [observation 2]

   ## Direction
   [What should change: more/less variation, different frequency range,
    different instruments to focus on, etc.]
   ```

3. **Start a supervised session** with Claude Code. Provide:
   - The batch summary report
   - Your listening notes
   - Any specific direction or questions

4. In the supervised session: make code/architecture changes, then configure and kick off the next autonomous batch.

---

## Handoff: Supervised Session → Autonomous Loop

At the end of a supervised session that precedes an autonomous batch:

1. Commit all code changes to main
2. All tests passing
3. Write the starting config for the batch to `configs/batch-NNN.json`
4. Define stopping conditions and guardrail thresholds in the config
5. Document in `docs/session-log.md`: what was changed, why, what the next batch should test
6. Provide clear launch instructions (the exact command to start the loop)

---

## Debugging Protocol

When something fails:

1. **Read the full error.** Don't skim. Copy the exact error and traceback.
2. **Identify the root cause.** Trace to origin. Don't guess.
3. **Fix the cause, not the symptom.** If a metric is wrong because alignment is off, fix alignment — don't adjust the threshold.
4. **Re-run the specific failing test** to confirm the fix.
5. **Re-run the full test suite.**
6. **Only proceed when everything passes.**

### Audio-Specific Debugging

When generated audio sounds wrong:

1. **Listen first.** Characterise the problem in words: "smeared transient", "metallic ringing", "sounds filtered", "too similar to input", "sounds like a different drum entirely".
2. **Check the codegram diff.** How many tokens changed? Which codebooks? Which time positions? Is the edit pattern what you expect?
3. **Check acceptance filtering.** What was the perceptual distance? Was it inside the band? If the filter passed something that sounds bad, the filter needs recalibrating, not the model.
4. **Check DAC reconstruction.** Encode the input through DAC and decode it — does the roundtrip sound clean? If not, the problem is the codec, not the variation model.
5. **Check the training data.** Is the sample from a domain the model has seen? Is the alignment correct? Is there a velocity layer mismatch?

---

## Communication Protocol

### What to report to the project owner

- **Phase start:** "Starting Phase N: [name]. Building [components]."
- **Phase complete:** "Phase N complete. [summary]. [test count] tests passing. [next steps]."
- **Blocker:** "Blocked on [issue]. Options: [A] or [B]. My recommendation is [X] because [reason]."
- **Decision needed:** "This requires a decision: [context]. Recommendation: [X]. Want to proceed?"
- **Gate reached:** "Gate [A/B/C] evaluation ready. Here's what the metrics and audio show: [summary]. Recommend [continue/adjust/pivot]."
- **Listening check-in needed:** "Batch [N] complete. [iteration count] iterations, stopped because [reason]. Listening pack ready at [path]. Key metrics: [summary]."

### What NOT to do

- Don't silently skip tests
- Don't make architecture changes without flagging them
- Don't change conventions in CLAUDE.md without discussion
- Don't modify the automation loop's allowed parameter set without discussion
- Don't dismiss poor listening results because metrics look good
- Don't continue past a go/no-go gate without explicit project owner approval

---

## Go/No-Go Gate Protocol

Gates are phase boundaries in the roadmap. Each requires explicit project owner approval to pass.

### Gate Evaluation Process

1. Generate the gate's required evaluation outputs (defined in ROADMAP.md)
2. Compute all metrics and comparison to baselines
3. Prepare a listening pack specifically for the gate criteria
4. Write a gate evaluation report: `reports/gate-X-evaluation.md`
5. Present findings to project owner with a clear recommendation: continue, adjust, or pivot
6. **Wait for explicit approval before proceeding past the gate**

### Gate Failure

If a gate fails:
- Document exactly what failed and why
- Propose concrete alternatives (different codebook range, more training data, architecture change, or pivot to non-ML approach)
- Do not treat a gate failure as a setback — it's the system working as designed. The gates exist to prevent wasted effort.

---

## Session Log Format

```markdown
# Session Log

## Session N — YYYY-MM-DD

### Phase
Phase X: [name]

### Completed
- [What was built or changed]
- [Specific details]

### Test Results
- tests/test_alignment.py: N passed
- tests/test_metrics.py: N passed
- [etc.]

### Audio Verification
- [What was listened to, what it sounded like]

### Issues Encountered
- [Issue]: [Resolution]

### Mistakes to Remember
- [If any — also add to CLAUDE.md Mistakes to Avoid]

### Next Steps
- [What the next session should do]
- [If autonomous batch: config path, iteration count, what to watch for]
```
