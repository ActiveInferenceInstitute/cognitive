---
title: Enriched Categories
description: Categories whose hom-sets are replaced by objects of a monoidal category, so that distances, linear structure, truth values or higher morphisms become part of the definition of a category.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - category_theory
  - enrichment
  - monoidal_categories
semantic_relations:
  - type: relates
    links:
      - "[[category_theory]]"
      - "[[monoidal_categories]]"
      - "[[functors]]"
      - "[[metric_geometry]]"
      - "[[information_theory]]"
---

# Enriched Categories

## Overview
In an ordinary category the morphisms between two objects form a set. In an enriched category they form an object of some other category $\mathcal{V}$, which lets the morphisms carry structure: a distance, an addition, a truth value, or a topology. The base $\mathcal{V}$ must be a [[monoidal_categories|monoidal category]] so that composition can be expressed. Kelly (1982) gave the standard development, and Lawvere (1973) showed that metric spaces are one of the most illuminating examples.

## Definition
Let $(\mathcal{V},\otimes,I)$ be a monoidal category. A **$\mathcal{V}$-enriched category** $\mathcal{C}$ consists of
- a collection of objects;
- for each pair of objects $a,b$, a hom-object $\mathcal{C}(a,b)\in\mathcal{V}$;
- composition morphisms $\mathcal{C}(b,c)\otimes\mathcal{C}(a,b)\to\mathcal{C}(a,c)$;
- identity morphisms $I\to\mathcal{C}(a,a)$,

subject to associativity and unit axioms expressed in $\mathcal{V}$. A **$\mathcal{V}$-functor** assigns objects to objects and gives morphisms $\mathcal{C}(a,b)\to\mathcal{D}(Fa,Fb)$ in $\mathcal{V}$ that respect composition and identities. Every enriched category has an underlying ordinary category whose morphisms $a\to b$ are the morphisms $I\to\mathcal{C}(a,b)$ in $\mathcal{V}$.

## Examples
- **Ordinary categories** are enriched over sets with the cartesian product.
- **Preadditive categories** are enriched over abelian groups, and **linear categories** over vector spaces. Hom-sets then carry addition and scalar multiplication, and composition is bilinear.
- **2-categories** are enriched over the category of small categories, so the morphisms between two objects themselves form a category.
- **Preorders** are enriched over the two-element poset $\{\text{false}\le\text{true}\}$ with conjunction as the tensor product.
- **Lawvere metric spaces** are enriched over the poset $([0,\infty],\ge)$ with addition as the tensor product and $0$ as the unit. The hom-object $\mathcal{C}(a,b)=d(a,b)$ is a distance, composition becomes the triangle inequality $d(a,b)+d(b,c)\ge d(a,c)$, and identities become $d(a,a)=0$. Symmetry is not required, so $d(a,b)\ne d(b,a)$ is allowed (Lawvere, 1973).

## Why it is useful
Enrichment lets constructions from ordinary category theory be carried over to settings with extra structure. Kelly's account develops weighted limits and colimits, Kan extensions and the Yoneda lemma for general $\mathcal{V}$. In the metric case, Lawvere (1973) showed that Cauchy completeness of a metric space is an instance of a general notion of completeness for enriched categories. Categories enriched over simplicial sets are also one of the standard models of higher categories, which is where they connect to the "Higher Categories" entry of [[category_theory]].

## Relation to Active Inference
The link is an analogy. Categorical formulations of probability and inference use monoidal categories, and hom-objects that carry a real-valued cost are natural to consider. The Kullback-Leibler divergence has the asymmetric shape of a Lawvere distance, but it does not satisfy the triangle inequality, so it does not make a space of distributions into an enriched category over $([0,\infty],\ge)$ in the sense above. Active Inference does not require enriched category theory.

## Related concepts
- [[category_theory]] — the unenriched theory
- [[monoidal_categories]] — the structure needed on the base of enrichment
- [[functors]] — maps between categories, here respecting the enrichment
- [[metric_geometry]] — the classical setting of the metric example
- [[information_theory]] — divergences and their lack of a triangle inequality

## References
- Lawvere, F. W. (1973). Metric spaces, generalized logic, and closed categories. Rendiconti del Seminario Matematico e Fisico di Milano, 43, 135-166. https://doi.org/10.1007/BF02924844
- Kelly, G. M. (1982). Basic Concepts of Enriched Category Theory. Cambridge University Press. Reprinted in Reprints in Theory and Applications of Categories, No. 10 (2005).
- nLab authors. Enriched category. nLab. https://ncatlab.org/nlab/show/enriched+category
