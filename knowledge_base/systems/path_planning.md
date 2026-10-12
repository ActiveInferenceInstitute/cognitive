---
title: Path Planning
description: The problem of computing a collision-free route from a start to a goal, with graph-search methods such as Dijkstra and A*, potential fields, and sampling-based planners such as roadmaps and rapidly exploring random trees.
type: concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - robotics
  - planning
  - search
semantic_relations:
  - type: relates
    links:
      - "[[optimal_control]]"
      - "[[dynamic_programming]]"
      - "[[belief_space_planning]]"
      - "[[graph_theory]]"
      - "[[swarm_robotics]]"
      - "[[policy_selection]]"
      - "[[expected_free_energy]]"
---

# Path Planning

## Overview
Path planning is the problem of finding a route for a robot, vehicle or other agent from a start state to a goal state that avoids obstacles and, usually, minimises a cost such as length or time. It is the geometric core of the broader problem of motion planning. The standard textbook treatment is LaValle (2006).

## Formulation
Planning is carried out in the **configuration space**: the set of all configurations of the agent, such as the position and orientation of a mobile robot or the joint angles of an arm. Obstacles in the workspace become forbidden regions of the configuration space, which turns the problem into finding a continuous curve in the free region that connects two points (Lozano-Perez, 1983). When the space is discretised into a graph with edge costs, planning becomes a shortest-path problem.

## Main families of methods
- **Graph search.** Dijkstra's algorithm (1959) finds shortest paths from a source in a graph with non-negative edge weights. A* (Hart, Nilsson and Raphael, 1968) orders its search by $f(n)=g(n)+h(n)$, where $g$ is the cost so far and $h$ a heuristic estimate of the remaining cost. If $h$ never overestimates the true remaining cost, A* returns an optimal path. A good heuristic lets it examine far fewer nodes than Dijkstra's algorithm.
- **Potential fields.** The goal attracts the robot and obstacles repel it, and the robot descends the resulting potential (Khatib, 1986). The approach is simple and fast but can become trapped in local minima.
- **Sampling-based planners.** In high-dimensional configuration spaces explicit enumeration is impractical. Probabilistic roadmaps sample free configurations, connect nearby samples and then search the resulting graph (Kavraki et al., 1996). Rapidly exploring random trees grow a tree from the start by repeatedly extending towards random samples, and handle kinematic and dynamic constraints (LaValle and Kuffner, 2001). These methods are probabilistically complete, meaning that the probability of finding a path, if one exists, tends to one as the number of samples grows.

## Connection to optimal control
Minimum-cost paths satisfy the principle of optimality, so path planning is a special case of the problems studied in [[optimal_control]] and solved by [[dynamic_programming]]. Planning under uncertainty about the robot's own state leads to belief-space formulations (see [[belief_space_planning]]).

## Relation to Active Inference
Active Inference frames planning as inference over policies: candidate sequences of actions are scored by expected free energy and selected with probability given by a softmax of that score (see [[policy_selection]] and [[expected_free_energy]]). A classical planner such as A* minimises a path cost on a known map. Expected free energy adds an information-seeking term, so an agent may choose a path in order to reduce uncertainty about the map. The two share the idea of searching over future trajectories, but they are different formalisms, and neither reduces to the other.

## Related concepts
- [[optimal_control]] — minimum-cost trajectories with dynamics
- [[dynamic_programming]] — the optimality principle behind graph search
- [[belief_space_planning]] — planning when the state is uncertain
- [[graph_theory]] — graphs and shortest paths
- [[swarm_robotics]] — planning for groups of robots
- [[policy_selection]] — action sequence selection in Active Inference
- [[expected_free_energy]] — the quantity scored in Active Inference planning

## References
- Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. Numerische Mathematik, 1(1), 269-271. https://doi.org/10.1007/BF01386390
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. IEEE Transactions on Systems Science and Cybernetics, 4(2), 100-107. https://doi.org/10.1109/TSSC.1968.300136
- Lozano-Perez, T. (1983). Spatial planning: A configuration space approach. IEEE Transactions on Computers, C-32(2), 108-120. https://doi.org/10.1109/TC.1983.1676196
- Khatib, O. (1986). Real-time obstacle avoidance for manipulators and mobile robots. The International Journal of Robotics Research, 5(1), 90-98. https://doi.org/10.1177/027836498600500106
- Kavraki, L. E., Svestka, P., Latombe, J.-C., & Overmars, M. H. (1996). Probabilistic roadmaps for path planning in high-dimensional configuration spaces. IEEE Transactions on Robotics and Automation, 12(4), 566-580. https://doi.org/10.1109/70.508439
- LaValle, S. M., & Kuffner, J. J. (2001). Randomized kinodynamic planning. The International Journal of Robotics Research, 20(5), 378-400. https://doi.org/10.1177/02783640122067453
- LaValle, S. M. (2006). Planning Algorithms. Cambridge University Press.
