---
title: Conditional Reasoning
description: Reasoning with if-then statements, covering valid and invalid inference forms, the Wason selection task, and competing accounts of how people interpret conditionals.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - reasoning
  - logic
  - cognition
semantic_relations:
  - type: relates
    links:
      - "[[deductive_reasoning]]"
      - "[[reasoning_problem_solving]]"
      - "[[propositional_logic]]"
      - "[[conditional_probability]]"
      - "[[bayes_theorem]]"
---

# Conditional Reasoning

## Overview
Conditional reasoning is inference from statements of the form "if $p$ then $q$". Given such a rule and a further premise, a reasoner draws, or declines to draw, a conclusion. It is one of the most heavily studied topics in the psychology of reasoning because it separates logical validity from the way people ordinarily interpret everyday conditionals.

## Inference forms
Two inferences are valid in classical propositional logic. Modus ponens: from $p \to q$ and $p$, infer $q$. Modus tollens: from $p \to q$ and $\neg q$, infer $\neg p$. Two are invalid: affirming the consequent (from $q$ infer $p$) and denying the antecedent (from $\neg p$ infer $\neg q$). People endorse modus ponens almost universally, modus tollens less often, and are frequently tempted by the two fallacies, especially when the rule is read as a biconditional or when background knowledge suggests other causes of $q$.

## The selection task
In Wason's selection task, participants see four cards and must choose which to turn over to test a rule such as "if a card has a vowel on one side, it has an even number on the other". Few choose the cards corresponding to modus tollens, namely the odd number. Performance improves markedly when the rule has a familiar social or deontic content, such as a permission or obligation, which has been explained by pragmatic reasoning schemas, social-contract reasoning and information-gain accounts.

## Competing accounts
Mental-model theory holds that reasoners represent the possibilities in which the conditional is true and often fail to represent all of them. Probabilistic accounts hold that people evaluate the conditional by the conditional probability $P(q \mid p)$, not by the material conditional, an idea going back to Ramsey's test. Evidence from a range of tasks supports the probabilistic view for everyday conditionals, though the debate is not settled.

## Relation to Active Inference
The connection is conceptual. If conditionals are assessed through $P(q \mid p)$, then reasoning with them resembles evaluating a likelihood or transition probability in a generative model. Active Inference does not itself specify a theory of conditional reasoning, and applying it here would be an interpretation, not a derivation.

## Related concepts
- [[deductive_reasoning]] — the broader class of reasoning in which conditionals are central
- [[reasoning_problem_solving]] — the wider study of thinking
- [[propositional_logic]] — the formal system defining validity
- [[conditional_probability]] — the quantity used in probabilistic accounts
- [[bayes_theorem]] — reverses a conditional, bearing on the affirming-the-consequent fallacy

## References
- Wason, P. C. (1968). Reasoning about a rule. Quarterly Journal of Experimental Psychology, 20, 273-281.
- Evans, J. St. B. T., & Over, D. E. (2004). If. Oxford University Press.
- Johnson-Laird, P. N., & Byrne, R. M. J. (2002). Conditionals: A theory of meaning, pragmatics, and inference. Psychological Review, 109, 646-678.
