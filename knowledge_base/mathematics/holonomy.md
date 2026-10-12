---
title: Holonomy
description: The transformation produced by parallel transport around a closed loop in a space with a connection, its relation to curvature, and its appearance as the Berry phase in physics.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - differential_geometry
  - connections
  - geometric_phase
semantic_relations:
  - type: relates
    links:
      - "[[fiber_bundles]]"
      - "[[berry_connection]]"
      - "[[geometric_phase]]"
      - "[[differential_geometry]]"
      - "[[lie_groups]]"
wikidata: Q907926
---

# Holonomy

## Overview
Holonomy measures how far a connection fails to be trivial. Given a connection on a vector bundle or principal bundle over a manifold $M$, a vector carried along a closed loop $\gamma$ by parallel transport generally returns changed. The resulting linear map $P_\gamma$ is the holonomy of the loop. On a flat plane every loop gives the identity, whereas on a sphere the vector is rotated by an angle set by the enclosed area.

## Definition and structure
Fix a base point $p\in M$. The set of maps $P_\gamma$ over all piecewise smooth loops based at $p$ forms a group, the holonomy group $\mathrm{Hol}_p$, which is a subgroup of the structure group of the bundle. Restricting to loops that can be contracted to a point gives the restricted holonomy group, which is a connected Lie subgroup.

Holonomy is tied to curvature. For an infinitesimal loop spanning $u$ and $v$, the holonomy differs from the identity by the curvature $F(u,v)$ times the enclosed area, to leading order. The Ambrose-Singer theorem makes this precise: the Lie algebra of the holonomy group is spanned by the values of the curvature, parallel-transported back to $p$. A connection with vanishing curvature has trivial restricted holonomy, although a flat connection on a space with non-trivial fundamental group can still have non-trivial holonomy around non-contractible loops.

For a $U(1)$ connection $A$ the holonomy of a loop is the phase $\exp(i\oint_\gamma A)$, and by Stokes' theorem $\oint_\gamma A=\iint F$ over a surface bounded by $\gamma$. This is the form taken by the Berry phase: when the parameters of a quantum Hamiltonian are carried slowly around a loop, an eigenstate acquires a phase equal to the holonomy of the Berry connection $A=i\langle n|\,dn\rangle$ (Berry, 1984).

## Role in Active Inference
Holonomy has no established derivation within Active Inference. It enters the vault's mathematical material through gauge-theoretic and geometric treatments of path integrals, where a phase factor accumulated around a closed path is the holonomy of a connection on a bundle of states. The link is a mathematical analogy to physics and geometry, not a result about inference. Information geometry also equips families of beliefs with affine connections and so supports a notion of holonomy, but its use in modelling agents remains exploratory.

## Related concepts
- [[fiber_bundles]] — the spaces on which connections and holonomy are defined
- [[berry_connection]] — the connection whose holonomy gives the Berry phase
- [[geometric_phase]] — the physical phase effect of non-trivial holonomy
- [[differential_geometry]] — curvature and parallel transport
- [[lie_groups]] — holonomy groups are Lie subgroups of the structure group
- [[path_integral_theory]] — context in which phase factors around loops appear

## References
- Kobayashi, S., & Nomizu, K. (1963). Foundations of Differential Geometry, Volume 1. Interscience Publishers.
- Nakahara, M. (2003). Geometry, Topology and Physics (2nd ed.). Institute of Physics Publishing.
- Berry, M. V. (1984). Quantal phase factors accompanying adiabatic changes. Proceedings of the Royal Society of London A, 392, 45-57.
