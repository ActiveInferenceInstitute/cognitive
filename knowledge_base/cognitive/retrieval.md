---
title: Retrieval
description: "Memory retrieval as the cue-driven reactivation of stored information, covering recall versus recognition, encoding specificity, retrieval practice, reconsolidation, and its reading as inference in Active Inference."
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - memory
  - cognition
  - recall
semantic_relations:
  - type: relates
    links:
      - "[[memory_systems]]"
      - "[[episodic_memory]]"
      - "[[pattern_completion]]"
      - "[[active_inference]]"
---

# Retrieval

## Overview
Retrieval is the stage of memory at which stored information is brought back into use, as conscious recall, recognition, or an unconscious influence on behaviour. It is distinct from encoding (forming a trace) and storage (maintaining it). In the memory-organisation pages it appears as the "get out" stage of the encoding, storage, retrieval sequence; this page uses the cognitive-psychology sense.

## Core claims
- **Cue dependence.** Retrieval starts from a cue, either external (a question, a smell) or internal (a goal, a prior thought). Success depends on how well the cue overlaps with the information encoded with the trace, the encoding specificity principle.
- **Recall versus recognition.** Free or cued recall requires generating the target; recognition requires only judging an offered item as old or new. Recognition is usually easier, and dual-process accounts attribute it to a recollection component plus a familiarity component.
- **Retrieval changes memory.** Retrieving an item strengthens it more than restudying does (the testing effect), and can impair related competitors (retrieval-induced forgetting). A reactivated trace may become labile and be stored again, which is the reconsolidation phenomenon first established in rodent fear memory.
- **Reconstruction.** Retrieval is generally treated as reconstruction from partial traces plus schema knowledge, which explains systematic distortions and false memories.
- **Neural basis.** Cued retrieval of episodes is associated with hippocampal pattern completion and prefrontal control processes that select among candidate memories and monitor the result (see [[source_memory]]).

## Role in Active Inference
Active Inference treats memory contents as part of a generative model, so retrieval can be read as inference: a partial cue is an observation, and recall is the posterior over the latent states that best explain it. This matches the computational notion of pattern completion in [[pattern_completion]]. The reading is an interpretive framework and an active research area, not a derivation specific to retrieval; the confidence people report about a retrieved item corresponds naturally to the precision of that posterior ([[precision_weighting]]).

## Related concepts
- [[episodic_memory]] - the memory system most often probed by cued recall
- [[pattern_completion]] - the computational operation behind cue-driven reactivation
- [[memory_consolidation]] - stabilisation processes that retrieval can reopen
- [[working_memory]] - holds retrieved content for ongoing use
- [[source_memory]] - judging where a retrieved item came from

## References
- Tulving, E., & Thomson, D. M. (1973). Encoding specificity and retrieval processes in episodic memory. Psychological Review.
- Roediger, H. L., & Karpicke, J. D. (2006). Test-enhanced learning: taking memory tests improves long-term retention. Psychological Science.
- Nader, K., Schafe, G. E., & LeDoux, J. E. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. Nature.
