---
title: Ant Colony Implementation
type: implementation
status: stable
created: 2025-01-01
updated: 2025-12-18
tags:
  - ant_colony
  - swarm_intelligence
  - implementation
  - active_inference
  - multi_agent
semantic_relations:
  - type: implements
    - [[../../knowledge_base/biology/myrmecology]]
    - [[../../knowledge_base/cognitive/social_cognition]]
    - [[AGENTS]]
---

# Ant Colony Implementation

This directory contains implementations of ant colony optimization and swarm intelligence algorithms, demonstrating collective behavior and emergent intelligence through Active Inference principles. The implementations showcase how individual agents with simple rules can create complex, adaptive systems.

## 🐜 Implementation Overview

### Core Components

#### Ant Agent Implementation
```python
class AntAgent:
    """Individual ant agent with foraging and communication capabilities."""

    def __init__(self, config, colony_id=None):
        """Initialize ant agent.

        Args:
            config: Agent configuration dictionary
            colony_id: Identifier for agent's colony (for multi-colony scenarios)
        """

        # Agent identity and state
        self.id = config.get('agent_id', np.random.randint(10000))
        self.colony_id = colony_id
        self.position = np.array(config.get('initial_position', [0.0, 0.0]))
        self.heading = config.get('initial_heading', np.random.uniform(0, 2*np.pi))

        # Behavioral state
        self.state = 'exploring'  # exploring, foraging, returning, recruiting
        self.has_food = False
        self.food_source_location = None
        self.nest_location = np.array(config.get('nest_location', [0.0, 0.0]))

        # Sensory capabilities
        self.sensory_range = config.get('sensory_range', 10.0)
        self.pheromone_sensitivity = config.get('pheromone_sensitivity', 1.0)

        # Movement parameters
        self.speed = config.get('speed', 1.0)
        self.turn_angle = config.get('turn_angle', np.pi/4)

        # Pheromone capabilities
        self.pheromone_deposit_rate = config.get('pheromone_deposit_rate', 1.0)
        self.pheromone_memory = []  # Memory of pheromone encounters

        # Decision making
        self.decision_noise = config.get('decision_noise', 0.1)
        self.memory_capacity = config.get('memory_capacity', 10)

        # Performance tracking
        self.distance_traveled = 0.0
        self.food_collected = 0
        self.encounters = 0

    def perceive_environment(self, environment):
        """Perceive local environment including pheromones, food, and other agents."""

        # Get local sensory information
        sensory_data = environment.get_local_sensory_data(
            self.position, self.sensory_range
        )

        # Process pheromone trails
        pheromone_signals = self.process_pheromone_signals(sensory_data['pheromones'])

        # Detect food sources
        food_signals = self.detect_food_sources(sensory_data['food'])

        # Detect other agents
        agent_signals = self.detect_other_agents(sensory_data['agents'])

        # Detect nest
        nest_distance = np.linalg.norm(self.position - self.nest_location)

        perception = {
            'pheromone_signals': pheromone_signals,
            'food_signals': food_signals,
            'agent_signals': agent_signals,
            'nest_distance': nest_distance,
            'obstacles': sensory_data.get('obstacles', [])
        }

        return perception

    def make_decision(self, perception):
        """Make behavioral decision based on current perception and state."""

        decision_factors = self.evaluate_decision_factors(perception)

        if self.state == 'exploring':
            decision = self.exploration_decision(perception, decision_factors)
        elif self.state == 'foraging':
            decision = self.foraging_decision(perception, decision_factors)
        elif self.state == 'returning':
            decision = self.return_decision(perception, decision_factors)
        elif self.state == 'recruiting':
            decision = self.recruitment_decision(perception, decision_factors)
        else:
            decision = self.default_decision(perception)

        # Add decision noise for realism
        decision = self.add_decision_noise(decision)

        return decision

    def evaluate_decision_factors(self, perception):
        """Evaluate factors influencing decision making."""

        factors = {}

        # Pheromone attraction/repulsion
        factors['pheromone_attraction'] = self.calculate_pheromone_attraction(
            perception['pheromone_signals']
        )

        # Food attraction
        factors['food_attraction'] = self.calculate_food_attraction(
            perception['food_signals']
        )

        # Nest attraction/repulsion
        factors['nest_attraction'] = self.calculate_nest_attraction(
            perception['nest_distance'], self.has_food
        )

        # Social factors
        factors['social_influence'] = self.calculate_social_influence(
            perception['agent_signals']
        )

        # Exploration drive
        factors['exploration_drive'] = self.calculate_exploration_drive()

        return factors

    def execute_action(self, decision, environment):
        """Execute decided action in the environment."""

        action_type = decision['action_type']

        if action_type == 'move':
            self.execute_movement(decision, environment)
        elif action_type == 'deposit_pheromone':
            self.execute_pheromone_deposit(decision, environment)
        elif action_type == 'pickup_food':
            self.execute_food_pickup(decision, environment)
        elif action_type == 'recruit':
            self.execute_recruitment(decision, environment)
        elif action_type == 'rest':
            self.execute_rest(decision, environment)

        # Update internal state
        self.update_internal_state(decision)

        # Track performance
        self.update_performance_metrics(decision)

    def execute_movement(self, decision, environment):
        """Execute movement action."""

        # Calculate new position
        movement_vector = decision['movement_vector']
        new_position = self.position + movement_vector

        # Check for obstacles and boundaries
        if environment.is_valid_position(new_position):
            # Update position and heading
            self.position = new_position
            self.heading = np.arctan2(movement_vector[1], movement_vector[0])

            # Update distance traveled
            self.distance_traveled += np.linalg.norm(movement_vector)
        else:
            # Handle invalid movement (bounce, turn, etc.)
            self.handle_invalid_movement(decision, environment)

    def execute_pheromone_deposit(self, decision, environment):
        """Execute pheromone deposition."""

        pheromone_type = decision.get('pheromone_type', 'foraging')
        amount = decision.get('amount', self.pheromone_deposit_rate)

        environment.deposit_pheromone(self.position, pheromone_type, amount, self.id)

    def exploration_decision(self, perception, factors):
        """Make decision during exploration state."""

        # Balance multiple influences
        total_attraction = (
            factors['pheromone_attraction'] * 0.4 +
            factors['food_attraction'] * 0.3 +
            factors['exploration_drive'] * 0.3
        )

        # Decide movement direction
        if np.linalg.norm(total_attraction) > 0.1:
            # Move toward attraction
            movement_direction = total_attraction / np.linalg.norm(total_attraction)
        else:
            # Random walk
            movement_direction = np.array([
                np.cos(self.heading + np.random.uniform(-self.turn_angle, self.turn_angle)),
                np.sin(self.heading + np.random.uniform(-self.turn_angle, self.turn_angle))
            ])

        # Normalize movement
        movement_vector = movement_direction * self.speed

        decision = {
            'action_type': 'move',
            'movement_vector': movement_vector,
            'state_transition': None  # Stay in exploring
        }

        return decision

    def foraging_decision(self, perception, factors):
        """Make decision during foraging state."""

        # Check for food sources
        if perception['food_signals']:
            nearest_food = min(perception['food_signals'],
                             key=lambda x: x['distance'])

            if nearest_food['distance'] < 1.0:  # Close enough to pick up
                decision = {
                    'action_type': 'pickup_food',
                    'food_source': nearest_food,
                    'state_transition': 'returning'
                }
                return decision

            # Move toward food
            movement_vector = self.calculate_movement_to_target(nearest_food['position'])
            decision = {
                'action_type': 'move',
                'movement_vector': movement_vector,
                'deposit_pheromone': True,
                'state_transition': None
            }
            return decision

        # Continue searching with pheromone guidance
        movement_vector = self.calculate_pheromone_guided_movement(
            perception['pheromone_signals']
        )

        decision = {
            'action_type': 'move',
            'movement_vector': movement_vector,
            'deposit_pheromone': True,
            'state_transition': None
        }

        return decision

    def calculate_movement_to_target(self, target_position):
        """Calculate movement vector toward target."""

        direction_vector = target_position - self.position
        distance = np.linalg.norm(direction_vector)

        if distance > 0:
            # Normalize and scale by speed
            direction = direction_vector / distance
            movement = direction * min(self.speed, distance)  # Don't overshoot
        else:
            # At target, small random movement
            angle = np.random.uniform(0, 2*np.pi)
            movement = np.array([np.cos(angle), np.sin(angle)]) * self.speed * 0.1

        return movement

    def calculate_pheromone_guided_movement(self, pheromone_signals):
        """Calculate movement influenced by pheromone trails."""

        if not pheromone_signals:
            # No pheromones, random movement
            angle = self.heading + np.random.uniform(-self.turn_angle, self.turn_angle)
            return np.array([np.cos(angle), np.sin(angle)]) * self.speed

        # Calculate pheromone gradient
        total_attraction = np.zeros(2)
        total_weight = 0

        for signal in pheromone_signals:
            direction = signal['position'] - self.position
            distance = np.linalg.norm(direction)

            if distance > 0:
                # Pheromone attraction decreases with distance
                attraction_strength = signal['intensity'] / (distance ** 2 + 1)
                attraction_direction = direction / distance

                total_attraction += attraction_direction * attraction_strength
                total_weight += attraction_strength

        if total_weight > 0:
            # Move toward pheromone source
            movement_direction = total_attraction / total_weight
            movement_direction = movement_direction / np.linalg.norm(movement_direction)
        else:
            # Random movement
            angle = np.random.uniform(0, 2*np.pi)
            movement_direction = np.array([np.cos(angle), np.sin(angle)])

        return movement_direction * self.speed

    def update_internal_state(self, decision):
        """Update internal state based on executed decision."""

        # Update behavioral state
        if 'state_transition' in decision and decision['state_transition']:
            self.state = decision['state_transition']

        # Update food status
        if decision['action_type'] == 'pickup_food':
            self.has_food = True
            self.food_source_location = decision['food_source']['position']
        elif decision['action_type'] == 'deposit_food':
            self.has_food = False
            self.food_source_location = None

        # Update memory
        self.update_memory(decision)

    def update_memory(self, decision):
        """Update agent's memory of experiences."""

        memory_item = {
            'timestamp': time.time(),
            'position': self.position.copy(),
            'action': decision['action_type'],
            'state': self.state,
            'has_food': self.has_food
        }

        self.pheromone_memory.append(memory_item)

        # Maintain memory capacity
        if len(self.pheromone_memory) > self.memory_capacity:
            self.pheromone_memory.pop(0)

    def update_performance_metrics(self, decision):
        """Update performance tracking metrics."""

        # Track encounters with other agents
        if 'agent_signals' in decision:
            self.encounters += len(decision['agent_signals'])

    def get_state_summary(self):
        """Get summary of current agent state."""

        return {
            'id': self.id,
            'position': self.position.copy(),
            'heading': self.heading,
            'state': self.state,
            'has_food': self.has_food,
            'colony_id': self.colony_id,
            'distance_traveled': self.distance_traveled,
            'food_collected': self.food_collected,
            'encounters': self.encounters
        }
```

#### Colony System Implementation
```python
class AntColony:
    """Ant colony system coordinating multiple ant agents."""

    def __init__(self, config):
        """Initialize ant colony.

        Args:
            config: Colony configuration dictionary
        """

        # Colony properties
        self.colony_id = config.get('colony_id', 0)
        self.nest_location = np.array(config.get('nest_location', [0.0, 0.0]))
        self.colony_size = config.get('colony_size', 50)

        # Agent population
        self.agents = []
        self.initialize_agents(config)

        # Environment interaction
        self.environment = config.get('environment')

        # Colony state
        self.food_stores = 0
        self.active_ants = self.colony_size
        self.found_food_sources = []

        # Performance tracking
        self.total_food_collected = 0
        self.average_distance_traveled = 0.0
        self.colony_efficiency = 0.0

        # Communication and coordination
        self.recruitment_signals = []
        self.shared_knowledge = {}

    def initialize_agents(self, config):
        """Initialize colony agent population."""

        agent_config = {
            'speed': config.get('ant_speed', 1.0),
            'sensory_range': config.get('sensory_range', 10.0),
            'pheromone_deposit_rate': config.get('pheromone_rate', 1.0),
            'memory_capacity': config.get('memory_capacity', 10),
            'nest_location': self.nest_location,
            'decision_noise': config.get('decision_noise', 0.1)
        }

        for i in range(self.colony_size):
            agent_config['agent_id'] = f"ant_{self.colony_id}_{i}"
            agent = AntAgent(agent_config, self.colony_id)
            self.agents.append(agent)

    def simulation_step(self):
        """Execute one simulation step for the entire colony."""

        # Process each agent
        for agent in self.agents:
            # Perception
            perception = agent.perceive_environment(self.environment)

            # Decision making
            decision = agent.make_decision(perception)

            # Action execution
            agent.execute_action(decision, self.environment)

            # Handle state transitions
            self.handle_agent_state_transition(agent, decision)

        # Colony-level processes
        self.update_colony_state()
        self.process_recruitment_signals()
        self.update_shared_knowledge()

        # Environment evolution
        self.environment.evolve_pheromones()

    def handle_agent_state_transition(self, agent, decision):
        """Handle agent state transitions and colony-level effects."""

        if decision.get('action_type') == 'pickup_food':
            # Agent found food
            food_source = decision['food_source']
            if food_source not in self.found_food_sources:
                self.found_food_sources.append(food_source)
                # Generate recruitment signal
                self.recruitment_signals.append({
                    'food_source': food_source,
                    'discoverer': agent.id,
                    'timestamp': time.time()
                })

        elif decision.get('action_type') == 'deposit_food':
            # Agent returned with food
            self.food_stores += 1
            self.total_food_collected += 1

    def update_colony_state(self):
        """Update overall colony state and statistics."""

        # Update active ants
        self.active_ants = sum(1 for agent in self.agents if agent.state != 'dead')

        # Update average distance
        total_distance = sum(agent.distance_traveled for agent in self.agents)
        self.average_distance_traveled = total_distance / len(self.agents)

        # Calculate colony efficiency
        if self.total_food_collected > 0:
            self.colony_efficiency = self.total_food_collected / (
                self.average_distance_traveled * len(self.agents)
            )

    def process_recruitment_signals(self):
        """Process recruitment signals to coordinate foraging."""

        current_time = time.time()

        # Remove old signals
        self.recruitment_signals = [
            signal for signal in self.recruitment_signals
            if current_time - signal['timestamp'] < 300  # 5 minutes
        ]

        # Use recruitment signals to guide exploration
        for signal in self.recruitment_signals:
            # Recruit nearby exploring ants
            self.recruit_ants_to_food_source(signal['food_source'])

    def recruit_ants_to_food_source(self, food_source):
        """Recruit ants to discovered food source."""

        # Find exploring ants near the nest
        exploring_ants = [
            agent for agent in self.agents
            if agent.state == 'exploring' and
            np.linalg.norm(agent.position - self.nest_location) < 20.0
        ]

        # Send recruitment signals to nearby ants
        for ant in exploring_ants[:5]:  # Recruit up to 5 ants
            ant.receive_recruitment_signal(food_source)

    def update_shared_knowledge(self):
        """Update colony's shared knowledge base."""

        # Aggregate knowledge from all agents
        colony_knowledge = {
            'known_food_sources': self.found_food_sources.copy(),
            'active_recruitment': len(self.recruitment_signals),
            'colony_size': self.active_ants,
            'food_stores': self.food_stores
        }

        self.shared_knowledge = colony_knowledge

    def get_colony_summary(self):
        """Get summary of colony state and performance."""

        agent_summaries = [agent.get_state_summary() for agent in self.agents]

        return {
            'colony_id': self.colony_id,
            'nest_location': self.nest_location,
            'colony_size': self.colony_size,
            'active_ants': self.active_ants,
            'food_stores': self.food_stores,
            'total_food_collected': self.total_food_collected,
            'average_distance_traveled': self.average_distance_traveled,
            'colony_efficiency': self.colony_efficiency,
            'found_food_sources': len(self.found_food_sources),
            'active_recruitment_signals': len(self.recruitment_signals),
            'agent_summaries': agent_summaries
        }
```

## 🏗️ Environment Implementation

### Ant Colony Environment
```python
class AntColonyEnvironment:
    """Environment for ant colony simulation."""

    def __init__(self, config):
        """Initialize ant colony environment."""

        # Spatial properties
        self.width = config.get('width', 100)
        self.height = config.get('height', 100)
        self.boundaries = config.get('boundaries', 'reflecting')

        # Pheromone system
        self.pheromone_layers = {}
        self.pheromone_decay_rate = config.get('pheromone_decay', 0.99)
        self.pheromone_diffusion_rate = config.get('diffusion_rate', 0.1)

        # Food sources
        self.food_sources = []
        self.initialize_food_sources(config)

        # Obstacles
        self.obstacles = []
        self.initialize_obstacles(config)

        # Environmental dynamics
        self.time_step = 0

    def initialize_food_sources(self, config):
        """Initialize food sources in environment."""

        num_food_sources = config.get('num_food_sources', 5)
        food_amount_range = config.get('food_amount_range', [50, 200])

        for i in range(num_food_sources):
            position = np.array([
                np.random.uniform(10, self.width - 10),
                np.random.uniform(10, self.height - 10)
            ])

            amount = np.random.uniform(food_amount_range[0], food_amount_range[1])

            food_source = {
                'id': i,
                'position': position,
                'amount': amount,
                'depleted': False
            }

            self.food_sources.append(food_source)

    def get_local_sensory_data(self, position, sensory_range):
        """Get sensory data available to agent at position."""

        # Pheromone signals
        pheromone_signals = self.get_pheromone_signals(position, sensory_range)

        # Food signals
        food_signals = self.get_food_signals(position, sensory_range)

        # Agent signals (would need agent list)
        agent_signals = []  # Placeholder

        # Obstacle detection
        obstacles = self.get_obstacles_in_range(position, sensory_range)

        return {
            'pheromones': pheromone_signals,
            'food': food_signals,
            'agents': agent_signals,
            'obstacles': obstacles
        }

    def get_pheromone_signals(self, position, range_radius):
        """Get pheromone signals in sensory range."""

        signals = []

        for pheromone_type, layer in self.pheromone_layers.items():
            # Get pheromone concentrations in range
            concentrations = self.sample_pheromone_layer(
                layer, position, range_radius
            )

            for concentration in concentrations:
                if concentration['intensity'] > 0.01:  # Detection threshold
                    signals.append({
                        'type': pheromone_type,
                        'position': concentration['position'],
                        'intensity': concentration['intensity']
                    })

        return signals

    def deposit_pheromone(self, position, pheromone_type, amount, agent_id):
        """Deposit pheromone at position."""

        if pheromone_type not in self.pheromone_layers:
            # Initialize pheromone layer
            self.pheromone_layers[pheromone_type] = np.zeros((self.width, self.height))

        # Deposit pheromone
        x, y = int(position[0]), int(position[1])
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pheromone_layers[pheromone_type][x, y] += amount

    def evolve_pheromones(self):
        """Evolve pheromone layers over time."""

        for pheromone_type, layer in self.pheromone_layers.items():
            # Apply decay
            layer *= self.pheromone_decay_rate

            # Apply diffusion
            layer = self.apply_diffusion(layer, self.pheromone_diffusion_rate)

            self.pheromone_layers[pheromone_type] = layer

        self.time_step += 1

    def apply_diffusion(self, layer, diffusion_rate):
        """Apply diffusion to pheromone layer."""

        # Simple diffusion using convolution
        kernel = np.array([[0.05, 0.1, 0.05],
                          [0.1,  0.6, 0.1],
                          [0.05, 0.1, 0.05]])

        diffused_layer = scipy.ndimage.convolve(layer, kernel, mode='constant')

        return diffused_layer

    def is_valid_position(self, position):
        """Check if position is valid (not obstacle, within bounds)."""

        x, y = position

        # Check boundaries
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False

        # Check obstacles
        for obstacle in self.obstacles:
            if np.linalg.norm(position - obstacle['position']) < obstacle['radius']:
                return False

        return True
```

## 🎯 Active Inference in Ant Colony Systems

### Active Inference Ant Agent
```python
class ActiveInferenceAntAgent(AntAgent):
    """Ant agent implementing Active Inference principles."""

    def __init__(self, config, colony_id=None):
        super().__init__(config, colony_id)

        # Active Inference components
        self.belief_system = AntBeliefSystem(config)
        self.policy_system = AntPolicySystem(config)
        self.free_energy_calculator = AntFreeEnergyCalculator(config)

        # Cognitive state
        self.internal_state = np.array([0.5, 0.5])  # Simple 2D internal state
        self.belief_history = [self.internal_state.copy()]

    def active_inference_decision(self, perception):
        """Make decision using Active Inference."""

        # Update beliefs about internal and external state
        self.update_active_inference_beliefs(perception)

        # Evaluate policies using expected free energy
        policies = self.generate_policies()
        policy_evaluations = []

        for policy in policies:
            efe = self.free_energy_calculator.calculate_expected_free_energy(
                policy, self.internal_state, perception
            )
            policy_evaluations.append({'policy': policy, 'efe': efe})

        # Select policy with minimum EFE
        optimal_policy = min(policy_evaluations, key=lambda x: x['efe'])
        selected_action = optimal_policy['policy'][0]  # First action

        # Convert to movement decision
        decision = self.convert_action_to_movement(selected_action, perception)

        return decision

    def update_active_inference_beliefs(self, perception):
        """Update beliefs using Active Inference."""

        # Predict next internal state
        predicted_state = self.belief_system.predict_internal_state(self.internal_state)

        # Get sensory prediction error
        prediction_error = self.calculate_prediction_error(perception, predicted_state)

        # Update internal state to minimize prediction error
        state_update = self.belief_system.update_internal_state(
            self.internal_state, prediction_error
        )

        self.internal_state = np.clip(self.internal_state + state_update, 0, 1)
        self.belief_history.append(self.internal_state.copy())

    def calculate_prediction_error(self, perception, predicted_state):
        """Calculate prediction error between perception and prediction."""

        # Simplified prediction error based on pheromone and food signals
        pheromone_intensity = np.mean([s['intensity'] for s in perception.get('pheromone_signals', [])])
        food_proximity = min([s['distance'] for s in perception.get('food_signals', [])] + [float('inf')])

        # Map to prediction error
        predicted_pheromone = predicted_state[0]
        predicted_food = 1.0 / (1.0 + predicted_state[1])  # Inverse relationship

        pheromone_error = pheromone_intensity - predicted_pheromone
        food_error = min(food_proximity / 10.0, 1.0) - predicted_food  # Normalize

        return np.array([pheromone_error, food_error])

    def generate_policies(self):
        """Generate action policies for evaluation."""

        # Simple policies: different movement directions
        actions = ['forward', 'left', 'right', 'deposit_pheromone']
        horizon = 2  # Look 2 steps ahead

        policies = list(itertools.product(actions, repeat=horizon))

        return policies

    def convert_action_to_movement(self, action, perception):
        """Convert Active Inference action to movement decision."""

        if action == 'forward':
            movement_vector = np.array([np.cos(self.heading), np.sin(self.heading)]) * self.speed
        elif action == 'left':
            new_heading = self.heading - self.turn_angle
            movement_vector = np.array([np.cos(new_heading), np.sin(new_heading)]) * self.speed
        elif action == 'right':
            new_heading = self.heading + self.turn_angle
            movement_vector = np.array([np.cos(new_heading), np.sin(new_heading)]) * self.speed
        elif action == 'deposit_pheromone':
            movement_vector = np.array([0.0, 0.0])  # Stay and deposit
        else:
            movement_vector = np.array([0.0, 0.0])  # Default: stay

        decision = {
            'action_type': 'move' if np.linalg.norm(movement_vector) > 0 else 'deposit_pheromone',
            'movement_vector': movement_vector,
            'deposit_pheromone': action == 'deposit_pheromone'
        }

        return decision
```

## 📊 Performance Analysis

### Colony Performance Metrics
```python
class AntColonyAnalyzer:
    """Analysis tools for ant colony performance."""

    def __init__(self):
        self.metrics = {}

    def analyze_colony_performance(self, colony, environment):
        """Analyze overall colony performance."""

        # Food collection efficiency
        food_efficiency = self.calculate_food_collection_efficiency(colony)

        # Exploration coverage
        exploration_coverage = self.calculate_exploration_coverage(colony, environment)

        # Pheromone trail effectiveness
        pheromone_effectiveness = self.calculate_pheromone_effectiveness(colony, environment)

        # Individual agent performance
        agent_performance = self.analyze_agent_performance(colony.agents)

        # Emergent behavior metrics
        emergent_behavior = self.analyze_emergent_behavior(colony)

        analysis = {
            'food_efficiency': food_efficiency,
            'exploration_coverage': exploration_coverage,
            'pheromone_effectiveness': pheromone_effectiveness,
            'agent_performance': agent_performance,
            'emergent_behavior': emergent_behavior,
            'overall_score': self.calculate_overall_score([
                food_efficiency, exploration_coverage, pheromone_effectiveness
            ])
        }

        return analysis

    def calculate_food_collection_efficiency(self, colony):
        """Calculate efficiency of food collection."""

        if colony.total_food_collected == 0:
            return 0.0

        # Efficiency = food collected / (distance traveled × time)
        total_distance = sum(agent.distance_traveled for agent in colony.agents)
        efficiency = colony.total_food_collected / (total_distance + 1)  # Avoid division by zero

        return min(efficiency, 1.0)  # Cap at 1.0

    def calculate_exploration_coverage(self, colony, environment):
        """Calculate exploration coverage of environment."""

        # Simple coverage calculation based on agent positions
        visited_positions = set()
        for agent in colony.agents:
            pos_tuple = (int(agent.position[0]), int(agent.position[1]))
            visited_positions.add(pos_tuple)

        total_positions = environment.width * environment.height
        coverage = len(visited_positions) / total_positions

        return coverage

    def calculate_pheromone_effectiveness(self, colony, environment):
        """Calculate effectiveness of pheromone communication."""

        total_pheromone = 0
        for layer in environment.pheromone_layers.values():
            total_pheromone += np.sum(layer)

        # Effectiveness = pheromone concentration near food sources
        effectiveness = 0.0
        for food_source in environment.food_sources:
            if not food_source['depleted']:
                pheromone_near_food = environment.get_pheromone_at_position(
                    food_source['position'], radius=5.0
                )
                effectiveness += pheromone_near_food

        effectiveness /= len(environment.food_sources) + 1  # Average

        return min(effectiveness, 1.0)

    def analyze_agent_performance(self, agents):
        """Analyze individual agent performance."""

        agent_metrics = []
        for agent in agents:
            metrics = {
                'distance_traveled': agent.distance_traveled,
                'food_collected': agent.food_collected,
                'encounters': agent.encounters,
                'efficiency': agent.food_collected / (agent.distance_traveled + 1)
            }
            agent_metrics.append(metrics)

        # Aggregate metrics
        avg_efficiency = np.mean([m['efficiency'] for m in agent_metrics])
        total_food = sum(m['food_collected'] for m in agent_metrics)

        return {
            'individual_metrics': agent_metrics,
            'average_efficiency': avg_efficiency,
            'total_colony_food': total_food
        }

    def analyze_emergent_behavior(self, colony):
        """Analyze emergent behavior in colony."""

        # Simple emergence metrics
        food_sources_found = len(colony.found_food_sources)
        recruitment_signals = len(colony.recruitment_signals)

        # Emergence score based on coordination
        emergence_score = min(food_sources_found * recruitment_signals / (len(colony.agents) + 1), 1.0)

        return {
            'food_sources_discovered': food_sources_found,
            'active_recruitment': recruitment_signals,
            'emergence_score': emergence_score
        }

    def calculate_overall_score(self, metrics):
        """Calculate overall performance score."""

        weights = [0.4, 0.3, 0.3]  # food, exploration, pheromone
        overall_score = sum(w * m for w, m in zip(weights, metrics))

        return overall_score
```

## 🚀 Usage Examples

### Basic Ant Colony Simulation
```python
# Setup environment
env_config = {
    'width': 100,
    'height': 100,
    'num_food_sources': 5,
    'pheromone_decay': 0.99
}
environment = AntColonyEnvironment(env_config)

# Setup colony
colony_config = {
    'colony_size': 30,
    'ant_speed': 1.0,
    'sensory_range': 10.0,
    'pheromone_rate': 1.0
}
colony = AntColony(colony_config)
colony.environment = environment

# Run simulation
analyzer = AntColonyAnalyzer()
for step in range(1000):
    colony.simulation_step()

    # Periodic analysis
    if step % 100 == 0:
        analysis = analyzer.analyze_colony_performance(colony, environment)
        print(f"Step {step}: Food collected = {colony.total_food_collected}, "
              f"Efficiency = {analysis['food_efficiency']:.3f}")

# Final analysis
final_analysis = analyzer.analyze_colony_performance(colony, environment)
print("Simulation complete!")
print(f"Final performance: {final_analysis['overall_score']:.3f}")
```

### Active Inference Ant Simulation
```python
# Setup Active Inference ants
ai_colony_config = {
    'colony_size': 20,
    'use_active_inference': True,
    'precision': 1.0,
    'planning_horizon': 3
}
ai_colony = ActiveInferenceAntColony(ai_colony_config, environment)

# Run Active Inference simulation
for step in range(500):
    ai_colony.active_inference_step()

    if step % 50 == 0:
        ai_analysis = analyzer.analyze_colony_performance(ai_colony, environment)
        print(f"AI Step {step}: Free Energy = {ai_colony.average_free_energy:.3f}")
```

## 📚 Related Documentation

### Theoretical Foundations
- [[../../knowledge_base/biology/myrmecology|Myrmecology]]
- [[../../knowledge_base/cognitive/social_cognition|Social Cognition]]
- [[../../knowledge_base/systems/swarm_intelligence|Swarm Intelligence]]

### Implementation Resources
- [[../../tools/src/models/active_inference/|Active Inference Models]]
- [[../../docs/examples/|Usage Examples]]
- [[../../docs/guides/|Implementation Guides]]

### Research Applications
- [[../../docs/research/ant_colony_active_inference|Ant Colony Research]]
- [[../../knowledge_base/biology/social_insect_cognition|Social Insect Cognition]]

## 🔗 Cross-References

### Core Components
- [[AGENTS|Ant Colony Agents]]
- [[../../Things/Generic_Thing/|Generic Thing Framework]]
- [[../../tools/src/visualization/|Visualization Tools]]

### Related Implementations
- [[../Generic_POMDP/|Generic POMDP]]
- [[../Simple_POMDP/|Simple POMDP]]
- [[../BioFirm/|BioFirm Implementation]]

---

> **Swarm Intelligence**: This implementation demonstrates how simple individual rules can lead to complex collective behavior, illustrating principles of emergence and self-organization.

---

> **Active Inference Integration**: The Active Inference ant agents show how cognitive principles can enhance traditional swarm intelligence algorithms.

---

> **Scalability**: The implementation supports varying colony sizes and can be extended to multi-colony scenarios with competition and cooperation.

