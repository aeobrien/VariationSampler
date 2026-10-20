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

This is analogous to how a large language model doesn't understand grammar as a linguist does — it learns the statistical patterns of how words follow other words, and from that emerges something that functions like understanding. Variation Sampler doesn't need to model drum physics. It needs to learn the statistical patterns of how round-robin recordings differ from each other, and from that should emerge something that functions like natural variation.

The model sees thousands of examples of "here are five recordings of the same snare hit — notice how they differ." From this, it learns a general template of what "same but different" means for percussive audio. When given a new, unseen one-shot, it applies that learned template to generate plausible siblings.

## What Success Enables

If v1 validates the core hypothesis, the path forward includes:

- **Velocity layer generation (v2).** Training the model to generate not just round-robin variations at the same intensity, but variations at different strike velocities — effectively turning a single sample into a dynamically playable instrument. This is a fundamentally harder problem (synthesis rather than variation) and requires its own research phase.

- **Instant instrument creation.** Combined with existing tooling for generating Ableton Sampler instrument files, a validated Variation Sampler could take a handful of field recordings and produce a fully playable, velocity-layered, round-robin-equipped virtual instrument in minutes rather than hours of studio recording.

- **Democratised deep sampling.** Any sound source — a desk, a cooking pot, a found object — becomes a candidate for a production-quality sampled instrument, lowering the barrier from "access to a recording studio and a week of session time" to "one recording on a phone."

These possibilities are real and motivating, but they are contingent. V1 comes first.

## Ethical Considerations

**Training data.** V1 trains on commercial drum libraries. The model learns statistical patterns of variation, not the specific sounds themselves — but the relationship between training on commercial content and generating new content from that training is worth being thoughtful about. The output is not a reproduction of any training sample; it is a novel variation of a user-provided input, informed by learned patterns. This is analogous to how a session drummer's technique is informed by every drummer they've listened to, without their playing being a reproduction of any specific performance.

**Commercial implications.** If successful, this technology could reduce demand for deeply-sampled commercial drum libraries. That's a real economic effect worth acknowledging, even though the immediate scope is personal use. The counterargument is that it could also expand the market by making sample-based production more accessible and by creating demand for high-quality source samples to feed into the system.

**Transparency.** If this tool is ever distributed beyond personal use, it should be clear about what it does — that variations are ML-generated, not recorded. Users of the resulting instruments should know what they're working with.
