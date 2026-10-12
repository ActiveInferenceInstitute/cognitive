---
title: Semantic Information
description: Information understood as meaningful content that is about something, in contrast to the statistical quantity of Shannon's theory, including the Bar-Hillel and Carnap measure and Floridi's truth-requiring account.
type: concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - philosophy_of_information
  - semantics
  - epistemology
  - logic
semantic_relations:
  - type: relates
    links:
      - "[[information_theory]]"
      - "[[mutual_information]]"
      - "[[philosophy_of_mind]]"
      - "[[logic]]"
      - "[[correspondence_theory]]"
---

# Semantic Information

## Overview
Shannon's mathematical theory of communication quantifies the statistical structure of messages without regard to what they mean (see [[information_theory]]). Theories of semantic information instead ask how much information a sentence or proposition carries about the world, and under what conditions something counts as information at all. The Stanford Encyclopedia of Philosophy entry by Sebastian Sequoiah-Grayson and Luciano Floridi organises the field around three approaches, which it presents in turn.

## Bar-Hillel and Carnap
Yehoshua Bar-Hillel and Rudolf Carnap's "An Outline of a Theory of Semantic Information" (1952) is the natural starting point. It is a quantitative theory designed to calculate the amount of semantic information carried by a sentence in a given language (monadic predicate logic). It rests on the inverse range principle: the amount of information carried by a sentence is inversely proportional to the likelihood that it is true. With $p(s)$ the a priori probability of a sentence $s$, the entry gives two measures,
$$\mathrm{cont}(s)=1-p(s),\qquad \mathrm{inf}(s)=\log_2\frac{1}{1-\mathrm{cont}(s)}=\log_2\frac{1}{p(s)}.$$
The theory has two famous problems. The *Bar-Hillel-Carnap paradox* is that a contradiction, false in all possible worlds, is assigned maximal information. The *scandal of deduction* is that the theory assigns minimal information to logical truths and valid deductions, although accounts of information still owe an explanation of why these seem informative.

## Floridi's strongly semantic information
Floridi's response to the paradox is the veridicality thesis: semantic information is well-formed, meaningful and truthful data. A proposition's quantity of information is measured by its distance from the situation it is meant to model, not by probability, so that contradictions and tautologies both carry none. Other authors use truthlikeness (verisimilitude) instead, which allows false statements to carry some information.

## Further approaches
The entry also discusses the relation of information to meaning and modes of presentation (the "Frege puzzle" for information), and the theory of information channels, where information flow is modelled by relations among situation types, and which is used to model epistemic phenomena such as knowledge.

## Relation to Active Inference
The relation is moderate. Active Inference rests on Shannon-style and Bayesian quantities such as surprise and mutual information (see [[mutual_information]]), and these are not semantic in the sense above. Debates about whether a system's internal states carry content about the world, and whether that content must be true, bear on how the generative model is interpreted (see [[philosophy_of_mind]]). Which notion of information applies to those states remains contested.

## Related concepts
- [[information_theory]] — the statistical theory from which semantic theories depart
- [[mutual_information]] — a statistical measure of the information one variable carries about another
- [[philosophy_of_mind]] — questions about mental content
- [[logic]] — the setting in which Bar-Hillel and Carnap's measures are defined
- [[correspondence_theory]] — truth as correspondence, relevant to the veridicality thesis

## References
- Sequoiah-Grayson, S., & Floridi, L. (2005). Semantic Conceptions of Information. In E. N. Zalta & U. Nodelman (Eds.), The Stanford Encyclopedia of Philosophy (first published 5 October 2005, substantive revision 14 January 2022). https://plato.stanford.edu/entries/information-semantic/
- Bar-Hillel, Y., & Carnap, R. (1953). Semantic information. The British Journal for the Philosophy of Science, 4(14), 147-157.
- Floridi, L. (2011). The Philosophy of Information. Oxford University Press.
