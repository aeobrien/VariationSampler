# VARIATION SAMPLER — Vision Statement

## Intent

Variation Sampler is a machine learning system that generates perceptually convincing round-robin variations from a single one-shot drum sample. Given one recording of a drum hit, it produces 4–8 additional versions that sound like the same instrument struck again under the same conditions — not like the same recording replayed, and not like the original sample has been processed.

The project exists to solve a specific, well-defined perceptual problem: the "machine-gun effect," where rapid retriggering of an identical sample produces an obviously artificial sound. Round-robin sampling eliminates this, but currently requires expensive, time-consuming studio recording. Variation Sampler aims to approximate that result from minimal source material using a model trained on thousands of real round-robin sets.

V1 is a proof of concept. Its purpose is to answer a single question: can a machine learning model generate drum sample variations that are subtle enough to preserve source identity while different enough to break perceptual repetition? Everything beyond that question — tooling, distribution, velocity layers, broader applications — is contingent on a clear yes.

## Motivation

A single drum sample, played back identically every time it is triggered, sounds mechanical. The human auditory system is extraordinarily sensitive to exact repetition — even when we can't consciously identify what's different between two recordings of the same drum being struck, our brains register the sameness. Professional sample libraries solve this with round-robin sampling, but that solution is only available for instruments that have been meticulously recorded in studios with that specific intent.

If you have a one-shot sample — a field recording, a sound design element, a single hit from a minimal sample pack — you're stuck with the machine-gun effect. There is currently no way to generate convincing round-robin variations from a single source sample. This project attempts to create one.

The project owner is a musician who works extensively with drum samples and has direct, daily experience of this limitation. The motivation is practical and personal: making one-shot samples usable in contexts where repetition artifacts are unacceptable, without being limited to the instruments that commercial libraries have chosen to deeply sample.

## Audience

V1 has an audience of one: the project owner, acting as both developer and evaluator. The immediate purpose is hypothesis validation, not product delivery.

If v1 succeeds, the natural audience is producers and sound designers who work with drum samples — particularly those using one-shots, field recordings, or found sounds that lack round-robin variants. The tool would be most valuable to anyone who wants organic, human-feeling percussion without being constrained to pre-existing deeply-sampled libraries. But that audience and its needs are explicitly deferred until the core feasibility question is answered.

## Scope

**In scope for v1:**
- Round-robin variation generation from a single one-shot drum sample.
- Output of 4–8 variations per input as 44.1 kHz / 16-bit / stereo WAV files.
- Training on existing commercial round-robin drum libraries.
- Cloud-based training and inference on Google Cloud.
- Proof-of-concept validation starting with a single instrument category (snares), expanding to other drum families if initial results are promising.
- Automated development iteration loop with quantitative metrics and periodic human listening evaluation.
- Go/no-go gates designed for early, cheap failure if the hypothesis doesn't hold.

**Explicitly out of scope for v1:**
- Velocity layer generation. This is a fundamentally different problem (synthesis, not variation) and requires separate research.
- Ableton instrument packaging. Existing tooling handles this independently and can be connected later.
- Real-time synthesis or live performance use.
- Any user-facing application, interface, or distribution mechanism.
- Cross-domain generalisation to non-percussive audio.
- Product polish, optimisation, or deployment concerns.

## Design Principles

**Perceptual quality is the ultimate authority.** Metrics serve the ear, never the other way around. If a variation measures well on spectral distance metrics but sounds wrong to a trained listener, it is wrong. If it measures poorly but sounds convincingly like a different hit of the same source, the metrics need re-examining. Every technical decision — architecture, training objective, acceptance thresholds — is subordinate to how the output sounds in actual musical use.

**Feasibility before everything.** V1 exists to answer whether this idea works at all. Speed-to-answer matters more than polish, elegance, or completeness. The project is structured to reach a clear yes or no as cheaply and quickly as possible, with explicit go/no-go gates designed for early termination if the hypothesis fails.

**Narrow and concrete over broad and speculative.** The ambitious long-term vision (instant instrument creation from found sounds, velocity layer generation, democratised deep sampling) is real and motivating, but it earns its place only after the core hypothesis is validated. Scope discipline is not a compromise — it is how the project avoids spending months building infrastructure for something that might not work.

**Subtlety is the product.** The variations must be subtle. They should not sound like different drums, or like the original has been processed, filtered, or effected. They should sound like what would happen if the same performer hit the same instrument again and made a new recording. The target is the narrow perceptual band where real round-robin recordings naturally sit — different enough to break repetition, similar enough to preserve identity.

## Definition of Done

V1 is complete when the following criteria are satisfied:

**The machine-gun test.** Generated variations, when played in rapid sequence (16th notes at 120 BPM or comparable), sound like the same drum source struck multiple times — not like the same recording replayed. This is evaluated by expert listening assessment from the project owner, which takes precedence over any quantitative metric.

**Two-axis validation.** Variations must simultaneously satisfy both:
- *Similarity:* Each variation is perceptibly the same instrument — same pitch, timbre, decay profile, energy envelope. It does not sound like a different drum or like the source has been processed.
- *Variation:* Each variation is perceptibly different from the input and from every other variation. The magnitude of difference falls within the range observed in real round-robin recordings from commercial drum libraries, as measured by multi-resolution spectral distance and MFCC distance against a calibrated baseline.

**Cross-source validation.** The above criteria are met not only on samples from libraries represented in training data, but on held-out libraries with different recording conditions. The model has learned variation patterns, not studio fingerprints.

**Concrete output.** Given a single one-shot drum sample, the system reliably produces 4–8 variations as 44.1 kHz / 16-bit / stereo WAV files that pass the above tests.

## Mental Model

The project is best understood through a DNA metaphor. A round-robin set is like a family of siblings: they share the same genetic code (the fundamental character of the instrument and how it was struck), but each individual expresses that code slightly differently. The model's job is not to understand the physics of why a drumstick hitting a snare at a marginally different angle produces a marginally different sound. Its job is to learn what that family of differences looks like in the finished audio — the statistical fingerprint of "same source, different instance" — and reproduce the pattern convincingly.

This is analogous to how a large language model learns the patterns of language without being taught grammar explicitly. The model learns from the output distribution — thousands of examples of what "the same drum hit multiple times" sounds like — without needing to model the physical process that generated the variation. It is not physical modelling or re-synthesis. It is pattern recognition applied to the subtle, almost imperceptible differences that make repeated sounds feel human.

## Ethical Considerations

**Training data provenance.** The model will be trained on commercial sample libraries that the project owner has legitimately purchased. The output is not intended to reproduce or redistribute those samples — it is intended to learn general patterns of micro-variation that transfer to novel, unrelated source material. Nonetheless, the project should be mindful of licensing terms and ensure that the trained model does not memorise or regurgitate specific commercial samples.

**Honest capability claims.** If v1 succeeds, any future communication about the tool should be honest about what it does and doesn't do. It generates plausible variations, not physically accurate simulations. The variations are statistically convincing, not ground-truth recordings. This distinction matters for anyone using the output in contexts where authenticity claims are relevant.