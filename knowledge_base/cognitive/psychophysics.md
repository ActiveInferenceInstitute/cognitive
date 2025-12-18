---

title: Psychophysics

type: concept

status: stable

tags:
  - psychophysics
  - perception
  - thresholds

semantic_relations:

  - type: relates

    links:
      - perception_processing
      - information_theory
      - active_inference

---

# Psychophysics

Psychophysics is the scientific study of the quantitative relationship between physical stimuli and perceptual experiences. It provides empirical foundations for understanding how sensory systems encode, process, and interpret environmental information. Modern psychophysics integrates signal detection theory with Bayesian inference, providing normative accounts of perception under uncertainty.

## Classical Psychophysics

### Threshold Measurement

#### Absolute Threshold

The minimum stimulus intensity required for detection:

```python
class PsychophysicalExperiment:
    """Framework for conducting psychophysical experiments."""

    def __init__(self, stimulus_generator, response_collector):
        self.stimulus_gen = stimulus_generator
        self.response_col = response_collector

    def measure_absolute_threshold(self, participant, method='staircase'):
        """Measure absolute detection threshold."""

        if method == 'staircase':
            threshold = self.staircase_procedure(participant)
        elif method == 'constant_stimuli':
            threshold = self.constant_stimuli_method(participant)

        return threshold

    def staircase_procedure(self, participant, start_intensity=0.5, step_size=0.1):
        """Adaptive staircase method for threshold estimation."""

        current_intensity = start_intensity
        direction = 1  # 1 = increasing, -1 = decreasing
        reversals = []

        for trial in range(100):  # Maximum trials
            # Present stimulus
            stimulus = self.stimulus_gen.generate_stimulus(current_intensity)
            response = participant.detect_stimulus(stimulus)

            # Update staircase
            if response == 'detected':
                if direction == 1:  # Was increasing, now reverse
                    direction = -1
                    reversals.append(current_intensity)
                current_intensity -= step_size
            else:  # Not detected
                if direction == -1:  # Was decreasing, now reverse
                    direction = 1
                    reversals.append(current_intensity)
                current_intensity += step_size

            # Reduce step size after reversals
            if len(reversals) > 10:
                step_size *= 0.5

        # Threshold is mean of last few reversals
        threshold = np.mean(reversals[-6:])

        return threshold
```

#### Difference Threshold (Just Noticeable Difference)

The smallest detectable difference between two stimuli:

```python
def measure_difference_threshold(self, participant, reference_intensity):
    """Measure just noticeable difference."""

    # Generate comparison stimuli around reference
    intensity_range = np.linspace(
        reference_intensity * 0.5,
        reference_intensity * 1.5,
        20
    )

    responses = {}

    for intensity in intensity_range:
        # Present reference and comparison
        stimulus_pair = self.stimulus_gen.generate_comparison(
            reference_intensity, intensity
        )

        response = participant.discriminate_stimuli(stimulus_pair)
        responses[intensity] = response

    # Fit psychometric function
    threshold = self.fit_psychometric_function(
        intensity_range, responses, reference_intensity
    )

    return threshold

def fit_psychometric_function(self, intensities, responses, reference):
    """Fit cumulative Gaussian to discrimination data."""

    # Convert responses to proportions
    differences = intensities - reference
    proportions = [np.mean(responses[intensity]) for intensity in intensities]

    # Fit cumulative Gaussian
    from scipy.optimize import curve_fit

    def cumulative_gaussian(x, mu, sigma):
        return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))

    params, _ = curve_fit(cumulative_gaussian, differences, proportions,
                         p0=[0, np.std(differences)])

    threshold = params[1]  # Standard deviation parameter

    return threshold
```

## Signal Detection Theory

### Core Concepts

#### Sensitivity (d')

The ability to discriminate signal from noise:

```math
d' = \frac{\mu_s - \mu_n}{\sigma}
```

Where:
- μ_s is the mean signal distribution
- μ_n is the mean noise distribution
- σ is the standard deviation (assumed equal)

```python
def calculate_sensitivity(self, signal_responses, noise_responses):
    """Calculate d' sensitivity measure."""

    # Calculate hit rate and false alarm rate
    hit_rate = np.mean(signal_responses)
    fa_rate = np.mean(noise_responses)

    # Prevent extreme values for z-score calculation
    hit_rate = np.clip(hit_rate, 0.01, 0.99)
    fa_rate = np.clip(fa_rate, 0.01, 0.99)

    # Z-scores
    z_hit = norm.ppf(hit_rate)
    z_fa = norm.ppf(fa_rate)

    # Sensitivity
    d_prime = z_hit - z_fa

    return d_prime, hit_rate, fa_rate
```

#### Decision Criterion (β)

Response bias in detection decisions:

```python
def calculate_criterion(self, hit_rate, fa_rate, d_prime):
    """Calculate decision criterion β."""

    # Likelihood ratio at criterion
    beta = norm.pdf(norm.ppf(fa_rate)) / norm.pdf(norm.ppf(hit_rate))

    # Alternative calculation using z-scores
    z_hit = norm.ppf(hit_rate)
    z_fa = norm.ppf(fa_rate)

    beta_alt = np.exp((z_fa**2 - z_hit**2) / (2 * d_prime))

    return beta, beta_alt
```

### Bayesian Formulation

Signal detection as Bayesian inference:

```python
class BayesianSignalDetector:
    """Signal detection as Bayesian inference."""

    def __init__(self, prior_signal_prob=0.5):
        self.prior_signal = prior_signal_prob
        self.prior_noise = 1 - prior_signal_prob

        # Likelihood parameters (to be estimated)
        self.signal_mean = 0.0
        self.noise_mean = 0.0
        self.signal_std = 1.0
        self.noise_std = 1.0

    def detect_signal(self, observation, cost_matrix=None):
        """Detect signal using Bayesian decision theory."""

        # Likelihoods
        likelihood_signal = norm.pdf(observation, self.signal_mean, self.signal_std)
        likelihood_noise = norm.pdf(observation, self.noise_mean, self.noise_std)

        # Posterior probabilities
        posterior_signal = (likelihood_signal * self.prior_signal) / (
            likelihood_signal * self.prior_signal + likelihood_noise * self.prior_noise
        )

        posterior_noise = 1 - posterior_signal

        # Decision based on maximum posterior
        decision = 'signal' if posterior_signal > posterior_noise else 'noise'

        # Expected loss decision (if costs provided)
        if cost_matrix is not None:
            expected_loss_signal = self.calculate_expected_loss(
                'signal', posterior_signal, posterior_noise, cost_matrix
            )
            expected_loss_noise = self.calculate_expected_loss(
                'noise', posterior_signal, posterior_noise, cost_matrix
            )

            decision = 'signal' if expected_loss_signal < expected_loss_noise else 'noise'

        return decision, posterior_signal, posterior_noise

    def calculate_expected_loss(self, decision, p_signal, p_noise, cost_matrix):
        """Calculate expected loss for a decision."""

        if decision == 'signal':
            # Loss if signal present: cost_matrix[0,0] (correct rejection? No)
            # Standard cost matrix: rows = true state, columns = decision
            # cost_matrix[0,0] = loss for deciding signal when signal present (correct)
            # cost_matrix[0,1] = loss for deciding noise when signal present (miss)
            # cost_matrix[1,0] = loss for deciding signal when noise present (false alarm)
            # cost_matrix[1,1] = loss for deciding noise when noise present (correct rejection)

            expected_loss = (p_signal * cost_matrix[0,0] + p_noise * cost_matrix[1,0])
        else:  # decision == 'noise'
            expected_loss = (p_signal * cost_matrix[0,1] + p_noise * cost_matrix[1,1])

        return expected_loss
```

## Psychometric Functions

### Fitting and Analysis

```python
def fit_psychometric_function(stimulus_intensities, responses, model='logistic'):
    """Fit psychometric function to data."""

    # Convert to proportions
    proportions = [np.mean(resp) for resp in responses]

    # Choose model
    if model == 'logistic':
        psychometric_func = logistic_psychometric
    elif model == 'gaussian':
        psychometric_func = gaussian_psychometric
    elif model == 'weibull':
        psychometric_func = weibull_psychometric

    # Fit parameters
    from scipy.optimize import curve_fit

    # Initial parameter guesses
    if model == 'logistic':
        p0 = [np.median(stimulus_intensities), 1.0, 0.5, 0.5]
    elif model == 'gaussian':
        p0 = [np.median(stimulus_intensities), 1.0, 0.5]
    elif model == 'weibull':
        p0 = [np.median(stimulus_intensities), 1.0, 2.0, 0.5]

    # Perform fit
    params, covariance = curve_fit(
        psychometric_func, stimulus_intensities, proportions, p0=p0
    )

    # Calculate threshold (50% point)
    if model == 'logistic':
        threshold = params[0]
    elif model == 'gaussian':
        threshold = params[0]
    elif model == 'weibull':
        threshold = params[0]

    # Calculate slope
    if model == 'logistic':
        slope = params[1]
    elif model == 'gaussian':
        slope = 1 / params[1]  # Gaussian width
    elif model == 'weibull':
        slope = params[1]  # Weibull slope

    return {
        'parameters': params,
        'threshold': threshold,
        'slope': slope,
        'model': model
    }

def logistic_psychometric(x, mu, sigma, gamma, lambda_):
    """Logistic psychometric function."""
    return gamma + (1 - gamma - lambda_) / (1 + np.exp(-(x - mu) / sigma))
```

## Active Inference and Psychophysics

### Perception as Inference

Psychophysics through the lens of active inference:

```python
class ActiveInferencePsychophysics:
    """Psychophysical experiments in active inference framework."""

    def __init__(self, generative_model):
        self.model = generative_model

    def psychophysical_judgment(self, stimulus, context):
        """Make psychophysical judgment using active inference."""

        # Encode stimulus in generative model
        observation = self.encode_stimulus(stimulus, context)

        # Infer hidden causes (perceptual interpretation)
        posterior_beliefs = self.perform_inference(observation)

        # Generate response based on beliefs
        response = self.generate_response(posterior_beliefs, context)

        return response, posterior_beliefs

    def encode_stimulus(self, stimulus, context):
        """Encode physical stimulus in model observation space."""

        # Sensory preprocessing
        sensory_features = self.sensory_preprocessing(stimulus)

        # Context integration
        contextualized_features = self.context_integration(
            sensory_features, context
        )

        return contextualized_features

    def perform_inference(self, observation):
        """Perform variational inference on observation."""

        # Initialize beliefs
        beliefs = self.initialize_prior_beliefs()

        # Minimize variational free energy
        for iteration in range(50):
            # Predict observation
            prediction = self.model.generate_prediction(beliefs)

            # Calculate prediction error
            prediction_error = observation - prediction

            # Update beliefs
            beliefs = self.update_beliefs(beliefs, prediction_error)

        return beliefs

    def generate_response(self, beliefs, context):
        """Generate psychophysical response from beliefs."""

        # Expected value calculation
        expected_intensity = self.calculate_expected_intensity(beliefs)

        # Decision threshold application
        if context['task'] == 'detection':
            response = 'detected' if expected_intensity > context['criterion'] else 'not_detected'
        elif context['task'] == 'discrimination':
            reference_intensity = context['reference']
            response = 'greater' if expected_intensity > reference_intensity else 'lesser'

        return response
```

### Adaptive Psychophysics

Active inference enables adaptive experimental design:

```python
class AdaptivePsychophysics:
    """Adaptive psychophysical testing using active inference."""

    def __init__(self, participant_model, experiment_design):
        self.participant = participant_model
        self.design = experiment_design
        self.history = []

    def run_adaptive_experiment(self, n_trials=100):
        """Run adaptive psychophysical experiment."""

        results = []

        for trial in range(n_trials):
            # Select optimal stimulus intensity
            stimulus_intensity = self.select_optimal_stimulus()

            # Present stimulus and get response
            stimulus = self.generate_stimulus(stimulus_intensity)
            response = self.get_participant_response(stimulus)

            # Update participant model
            self.update_participant_model(stimulus_intensity, response)

            # Record trial
            trial_result = {
                'trial': trial,
                'stimulus_intensity': stimulus_intensity,
                'response': response
            }
            results.append(trial_result)
            self.history.append(trial_result)

        return results

    def select_optimal_stimulus(self):
        """Select stimulus that maximizes information gain."""

        # Candidate intensities
        candidate_intensities = np.linspace(0.1, 10.0, 50)

        # Calculate expected information gain for each
        information_gains = []
        for intensity in candidate_intensities:
            gain = self.calculate_expected_information_gain(intensity)
            information_gains.append(gain)

        # Select intensity with maximum expected information
        optimal_intensity = candidate_intensities[np.argmax(information_gains)]

        return optimal_intensity

    def calculate_expected_information_gain(self, intensity):
        """Calculate expected information gain for stimulus intensity."""

        # Simulate possible responses
        possible_responses = ['detected', 'not_detected']

        expected_gain = 0.0

        for response in possible_responses:
            # Probability of response
            p_response = self.participant.predict_response_probability(intensity, response)

            # Information gain if response occurs
            gain = self.calculate_information_gain(intensity, response)
            expected_gain += p_response * gain

        return expected_gain
```

## Applications

### Clinical Assessment

Psychophysics in diagnosis and treatment monitoring:

- **Visual disorders**: Contrast sensitivity, visual acuity
- **Auditory disorders**: Pure tone audiometry, speech discrimination
- **Neurological assessment**: Somatosensory thresholds, temporal discrimination

### Human Factors

Psychophysics in system design:

- **Display design**: Optimal contrast ratios, font sizes
- **Warning systems**: Detectable alarm signals
- **Interface design**: Response time requirements

### Research Methodology

Advanced psychophysical techniques:

- **Magnitude estimation**: Subjective scaling of stimulus intensity
- **Cross-modal matching**: Comparing intensities across senses
- **Psychophysical reverse correlation**: Revealing perceptual templates

## Statistical Analysis

### Bootstrap Confidence Intervals

```python
def bootstrap_psychophysical_analysis(data, n_bootstraps=1000):
    """Bootstrap analysis of psychophysical data."""

    bootstrap_results = []

    for _ in range(n_bootstraps):
        # Resample data with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)

        # Analyze resampled data
        result = analyze_psychophysical_data(bootstrap_sample)
        bootstrap_results.append(result)

    # Calculate confidence intervals
    ci_lower = np.percentile(bootstrap_results, 2.5)
    ci_upper = np.percentile(bootstrap_results, 97.5)
    mean_estimate = np.mean(bootstrap_results)

    return {
        'estimate': mean_estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_distribution': bootstrap_results
    }
```

### Model Comparison

Comparing psychophysical models:

```python
def compare_psychophysical_models(data, models):
    """Compare competing psychophysical models."""

    model_comparisons = {}

    for model_name, model_func in models.items():
        # Fit model to data
        fit_result = fit_psychophysical_model(data, model_func)

        # Calculate goodness of fit
        aic = calculate_aic(fit_result)
        bic = calculate_bic(fit_result, len(data))

        # Cross-validation performance
        cv_performance = cross_validate_model(data, model_func)

        model_comparisons[model_name] = {
            'aic': aic,
            'bic': bic,
            'cv_performance': cv_performance,
            'parameters': fit_result['parameters']
        }

    # Rank models
    ranked_models = sorted(model_comparisons.items(),
                          key=lambda x: x[1]['aic'])

    return ranked_models, model_comparisons
```

---

## Related Concepts

- [[perception_processing]] - General perceptual processes
- [[bayesian_inference]] - Probabilistic reasoning framework
- [[active_inference]] - Action-oriented perception
- [[signal_detection_theory]] - Decision-making under uncertainty
- [[psychometric_function]] - Mathematical models of perception

