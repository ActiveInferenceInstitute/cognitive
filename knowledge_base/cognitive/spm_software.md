---
title: SPM Software
description: Statistical Parametric Mapping, the free and open source software developed at the Functional Imaging Laboratory at UCL for analysing brain imaging data with the general linear model and for dynamic causal modelling.
type: concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - neuroimaging
  - statistics
  - software
  - dynamic_causal_modelling
aliases:
  - SPM
  - Statistical Parametric Mapping
semantic_relations:
  - type: relates
    links:
      - "[[neuroimaging]]"
      - "[[fmri_studies]]"
      - "[[meg]]"
      - "[[hypothesis_testing]]"
      - "[[bayes_factors]]"
      - "[[variational_methods]]"
      - "[[free_energy_principle]]"
---

# SPM Software

## Overview
SPM is a software package for the analysis of brain imaging data. Its name comes from the method it implements, statistical parametric mapping: the construction and assessment of spatially extended statistical processes used to test hypotheses about functional imaging data. It is developed by the Functional Imaging Laboratory (FIL) at the UCL Queen Square Institute of Neurology in London, is described by its maintainers as free and open source, and was first released to the research community in 1991. It was originally developed by Karl Friston for the analysis of PET data, and the current releases handle sequences of images from fMRI, PET, SPECT, EEG and MEG.

## Statistical parametric mapping
The core idea is to fit the same model independently at every voxel and then summarise the results as an image. In the general linear approach described by Friston et al. (1994), the time series at each voxel $v$ is modelled as

$$y_v = X\beta_v + \varepsilon_v,$$

where $X$ is a design matrix whose columns are experimental regressors (for example stimulus time courses convolved with a haemodynamic response), $\beta_v$ are the unknown effects and $\varepsilon_v$ is noise. A contrast $c^\top\beta_v$ is tested with a $t$ or $F$ statistic at each voxel, giving a statistical parametric map. Because thousands of voxels are tested, inference must correct for multiple comparisons. SPM does this using the smoothness of the image, treated through the theory of random fields, so that thresholds control the family-wise error over the whole volume instead of per voxel.

## Extensions
The package has grown beyond mass-univariate analysis. It includes spatial preprocessing steps (realignment, normalisation to a template, smoothing), Bayesian estimation of the general linear model, and dynamic causal modelling (DCM), in which a generative model of how neuronal states cause the measured signal is fitted to data and models are compared by approximate model evidence (Friston, Harrison and Penny, 2003).

## Relation to Active Inference
The connection is one of shared authorship and shared methods, not of identical theory. The group that developed SPM also developed the free energy principle, and DCM uses variational Bayesian inference with a free-energy bound on model evidence, the same quantity that Active Inference agents minimise. SPM itself is a data-analysis tool: using it does not commit an analyst to Active Inference as a theory of the brain, and a statistical parametric map says nothing by itself about predictive coding or belief updating. Modelling choices (the design matrix, the DCM architecture) carry the theoretical assumptions.

## Related concepts
- [[neuroimaging]] — the measurement modalities that SPM analyses
- [[fmri_studies]] — the most common application of voxel-wise modelling
- [[meg]] — one of the modalities supported alongside EEG
- [[hypothesis_testing]] — the inferential framework behind thresholded maps
- [[bayes_factors]] — model comparison used in DCM
- [[variational_methods]] — the approximate inference behind DCM

## References
- Friston, K. J., Holmes, A. P., Worsley, K. J., Poline, J.-P., Frith, C. D., & Frackowiak, R. S. J. (1994). Statistical parametric maps in functional imaging: a general linear approach. Human Brain Mapping, 2(4), 189-210. https://doi.org/10.1002/hbm.460020402
- Friston, K. J., Harrison, L., & Penny, W. (2003). Dynamic causal modelling. NeuroImage, 19(4), 1273-1302. https://doi.org/10.1016/S1053-8119(03)00202-7
- Friston, K. J., Ashburner, J. T., Kiebel, S. J., Nichols, T. E., & Penny, W. D. (Eds.). (2007). Statistical Parametric Mapping: The Analysis of Functional Brain Images. Academic Press (Elsevier). ISBN 9780123725608.
- Functional Imaging Laboratory, UCL Queen Square Institute of Neurology. SPM: Statistical Parametric Mapping. https://www.fil.ion.ucl.ac.uk/spm/
