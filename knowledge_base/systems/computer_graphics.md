---
title: Computer Graphics
description: The field concerned with synthesising and manipulating images by computer, covering geometric modelling, the projection and rasterisation pipeline, shading and light transport, and real-time rendering for interactive and virtual environments.
type: concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - computer_graphics
  - rendering
  - visualization
  - virtual_environments
semantic_relations:
  - type: relates
    links:
      - "[[virtual_reality]]"
      - "[[augmented_reality]]"
      - "[[spatial_computing]]"
      - "[[metaverse]]"
      - "[[human_computer_interaction]]"
      - "[[linear_algebra]]"
      - "[[visual_perception]]"
      - "[[generative_models]]"
---

# Computer Graphics

## Overview
Computer graphics is the study and practice of generating images from data. Its central problem is rendering: given a description of a scene, meaning its geometry, materials, lights and a camera, compute the image that a viewer at that camera would see. The same techniques underlie film and games, scientific and information visualisation, user interfaces, and the displays of virtual and augmented reality (see [[virtual_reality]], [[augmented_reality]] and [[metaverse]]).

## Core components
- **Modelling.** Objects are represented as polygon meshes, parametric surfaces, point sets, volumes or implicit functions. Transformations such as translation, rotation and scaling are expressed as matrices, usually in homogeneous coordinates, so that a sequence of transformations is a matrix product (see [[linear_algebra]]).
- **Viewing and projection.** A perspective projection maps three-dimensional points to the image plane, with a depth value retained for deciding which surfaces are visible.
- **Rasterisation.** In the pipeline used for real-time rendering, triangles are projected to the screen, converted to fragments, and shaded, with a depth buffer resolving visibility. Programmable shader stages on graphics hardware compute the position and appearance of each element.
- **Ray tracing.** Rays are traced from the camera into the scene and their intersections with surfaces determine colour, with secondary rays used for shadows, reflection and refraction. Whitted's recursive ray tracing is a classic example.
- **Light transport.** Physically based rendering aims to solve how light is exchanged between surfaces. Kajiya (1986) formulated the rendering equation, in which the outgoing radiance from a point $x$ in direction $\omega_o$ is its emitted radiance plus the reflected incoming radiance:

$$L_o(x,\omega_o)=L_e(x,\omega_o)+\int_{\Omega} f_r(x,\omega_i,\omega_o)\,L_i(x,\omega_i)\,(\omega_i\cdot n)\,d\omega_i,$$

where $f_r$ is the bidirectional reflectance distribution function, $n$ the surface normal and $\Omega$ the hemisphere of directions about $n$. Most offline renderers estimate this integral by Monte Carlo sampling.

## Relation to cognition and Active Inference
Graphics inverts the problem faced by the visual system. A renderer maps a scene description to an image, while perception must infer a scene from images, which is why the idea of vision as inverse graphics has been used in discussions of perception (see [[visual_perception]]). In the language of Active Inference, a renderer plays the role of the likelihood mapping in a generative model that predicts sensory input from hidden causes (see [[generative_models]]). This is an analogy of function. The free energy principle makes no claim about graphics, and the cited sources do not discuss perception. In immersive systems, graphics determines the sensory input that is delivered to a user, and the rendering latency and fidelity affect how well the user's predictions about the effects of their movements are met.

## References
- Kajiya, J. T. (1986). The rendering equation. ACM SIGGRAPH Computer Graphics, 20(4), 143-150. doi:10.1145/15886.15902
- Whitted, T. (1980). An improved illumination model for shaded display. Communications of the ACM, 23(6), 343-349. doi:10.1145/358876.358882
- Foley, J. D., van Dam, A., Feiner, S. K., & Hughes, J. F. (1990). Computer Graphics: Principles and Practice (2nd ed.). Addison-Wesley.
- Akenine-Möller, T., Haines, E., & Hoffman, N. (2018). Real-Time Rendering (4th ed.). A K Peters/CRC Press.
