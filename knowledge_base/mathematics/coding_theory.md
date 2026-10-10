---
title: Coding Theory
description: The mathematics of representing information as codewords for compression and reliable transmission, covering source coding bounds, error-correcting codes, Shannon's channel theorem, and the links to message passing.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - information_theory
  - error_correction
  - compression
semantic_relations:
  - type: relates
    links:
      - "[[information_theory]]"
      - "[[channel_capacity]]"
      - "[[message_passing]]"
      - "[[entropy]]"
---

# Coding Theory

## Overview
Coding theory studies how to map messages to sequences of symbols so that they are short, resistant to noise, or both. It has two halves. Source coding removes redundancy to compress data; channel coding adds structured redundancy so that errors can be detected and corrected. Shannon's 1948 results fix the limits of both in terms of entropy and channel capacity.

## Results and examples
- **Kraft inequality.** A binary prefix code with codeword lengths $\ell_i$ exists if and only if $\sum_i 2^{-\ell_i}\le1$. The optimal expected length $L$ satisfies $H(X)\le L<H(X)+1$ bits, and Huffman coding attains the optimum for symbol-by-symbol codes. Coding with the wrong distribution $q$ costs extra bits equal to the KL divergence $D_{\mathrm{KL}}(p\Vert q)$.
- **Channel coding theorem.** For a channel of capacity $C$ ([[channel_capacity]]) there are codes with vanishing error probability at any rate below $C$, and none above it.
- **Block codes.** A code with minimum Hamming distance $d$ detects $d-1$ errors and corrects $\lfloor(d-1)/2\rfloor$. The Hamming $(7,4)$ code has 4 data bits and 3 parity bits and $d=3$ (verified by enumerating all 16 codewords), so it corrects any single-bit error. Reed-Solomon and BCH codes extend this algebraically.
- **Sparse-graph codes.** LDPC and turbo codes approach capacity and are decoded by iterative message passing on a factor graph, the sum-product algorithm described in [[belief_propagation]]. Gallager introduced LDPC codes in 1962, and they were rediscovered in the 1990s.
- **Minimum description length.** Choosing the model giving the shortest total code for data plus model connects coding to statistical inference.

## Role in Active Inference
The connection is mathematical overlap, not a derivation. Message-passing inference on factor graphs and LDPC decoding are the same algorithm applied to different graphs, and the complexity term of variational free energy, $D_{\mathrm{KL}}(q\Vert p)$, has the coding interpretation of the extra nats needed to encode data using the prior instead of the posterior. Efficient-coding hypotheses in neuroscience use Shannon's results to ask how sensory codes should be shaped, and are related to Active Inference but are not the same claim: the latter concerns agents minimising surprise, including through action, whereas coding theory is silent about action.

## Related concepts
- [[information_theory]] — the framework that sets coding limits
- [[entropy]] — the bound on lossless compression
- [[kl_divergence]] — the penalty for coding with a mismatched model
- [[message_passing]] — the shared decoding and inference algorithm
- [[channel_capacity]] — the limit on reliable communication
- [[algorithmic_information]] — description length without a probability model

## References
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27, 379-423 and 623-656.
- MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press.
- Gallager, R. G. (1962). Low-density parity-check codes. *IRE Transactions on Information Theory*.
