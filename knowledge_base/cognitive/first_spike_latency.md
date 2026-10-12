---
title: First Spike Latency
description: The delay between stimulus onset and a neuron's first action potential as a temporal code, evidence from vision and touch, and its relation to rate coding and predictive models.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neural_coding
  - spike_timing
  - latency
semantic_relations:
  - type: relates
    links:
      - "[[neural_coding]]"
      - "[[firing_rate]]"
      - "[[spike_timing]]"
      - "[[temporal_coding]]"
      - "[[spike_trains]]"
---

# First Spike Latency

## Overview
First spike latency is the time between the onset of a stimulus and the first action potential that a neuron emits in response. It has been proposed as a neural code in which information about the stimulus is carried by when a neuron first fires, and not by how many spikes follow. In general, stronger or more preferred stimuli evoke shorter latencies, so latency can serve as a monotonic readout of stimulus strength or feature match.

## Why latency may matter
Behaviour can be too fast to rely on averaging spike counts over long windows. Thorpe, Fize and Marlot (1996) reported that human observers can detect an animal in a briefly shown natural image, and that event-related potentials differ between target and non-target images about 150 ms after onset. Given the number of processing stages and the typical firing rates of cortical neurons, this suggests that each stage can contribute little more than a few spikes. Thorpe, Delorme and Van Rullen (2001) argued that the order in which neurons in a population fire first can carry information in this regime, a proposal known as rank-order coding.

Experimental support has been found in several sensory systems. Gollisch and Meister (2008) showed that the relative latencies of retinal ganglion cell spikes after a flash encode spatial image features, and that the relative timing carries more information about the stimulus than the spike counts do in their experiments. In the somatosensory system, first-spike timing of tactile afferents has been reported to carry information about contact properties.

## Limitations
Latency also depends on non-stimulus factors, such as contrast, adaptation state and the background activity of the neuron, so a downstream reader needs a reference, for example the population's onset time. Whether the brain actually uses latency as a code, as opposed to a correlate of rate, is debated. Rate and latency often co-vary, and the two are difficult to separate experimentally (see [[firing_rate]]).

## Relation to Active Inference
The link is weak. Active Inference process theories are generally expressed in terms of firing rates as expectations, with timing treated only through the dynamics of the underlying variables. Spike-latency codes are compatible with a predictive-coding picture, since a well-predicted input may need fewer spikes, but no standard Active Inference account assigns a specific role to first-spike latency.

## Related concepts
- [[firing_rate]] — the alternative rate description
- [[spike_timing]] — the broader question of temporal precision
- [[temporal_coding]] — codes carried by timing
- [[neural_coding]] — the general framework
- [[spike_trains]] — the event sequences in which latency is measured

## References
- Thorpe, S., Fize, D., & Marlot, C. (1996). Speed of processing in the human visual system. Nature, 381, 520-522.
- Thorpe, S., Delorme, A., & Van Rullen, R. (2001). Spike-based strategies for rapid processing. Neural Networks, 14, 715-725.
- Gollisch, T., & Meister, M. (2008). Rapid neural coding in the retina with relative spike latencies. Science, 319, 1108-1111.
