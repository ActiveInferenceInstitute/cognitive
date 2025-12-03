---
title: Baseball Game Implementation
type: implementation
status: active
created: 2025-01-01
updated: 2025-01-01
tags:
  - baseball
  - game
  - simulation
  - decision_making
  - implementation
semantic_relations:
  - type: implements
    - [[../../knowledge_base/cognitive/decision_making]]
    - [[../../docs/guides/application/]]
---

# Baseball Game Implementation

This directory contains implementations of baseball game simulations and decision-making systems, demonstrating cognitive modeling in sports analytics and strategic decision-making contexts. The implementations showcase how cognitive agents can model complex decision-making in dynamic, uncertain environments.

## ⚾ Implementation Overview

### Core Components

#### Baseball Environment Model
```python
class BaseballEnvironment:
    """Baseball game environment with realistic physics and rules."""

    def __init__(self, config):
        """Initialize baseball environment.

        Args:
            config: Environment configuration dictionary
        """

        # Field dimensions (standard MLB field)
        self.field_dimensions = {
            'foul_lines':  foul_lines,
            'outfield_distance': 400,  # feet
            'base_paths': 90,  # feet between bases
            'pitchers_mound': 60.5  # feet from home plate
        }

        # Game state
        self.inning = 1
        self.half_inning = 'top'  # 'top' or 'bottom'
        self.score = {'home': 0, 'away': 0}
        self.outs = 0
        self.runners_on_base = {'1st': None, '2nd': None, '3rd': None}

        # Ball and players
        self.ball_position = np.array([0.0, 0.0, 0.0])  # x, y, z coordinates
        self.ball_velocity = np.array([0.0, 0.0, 0.0])
        self.batter = None
        self.pitcher = None
        self.fielders = {}
        self.runners = {}

        # Environmental factors
        self.weather_conditions = config.get('weather', 'clear')
        self.wind_speed = config.get('wind_speed', 0.0)
        self.temperature = config.get('temperature', 70.0)  # Fahrenheit

        # Physics parameters
        self.gravity = -32.2  # ft/s²
        self.air_density = self.calculate_air_density(self.temperature)
        self.drag_coefficient = 0.3

        # Rule enforcement
        self.rules_engine = BaseballRulesEngine()

    def reset_game(self):
        """Reset game to starting state."""
        self.inning = 1
        self.half_inning = 'top'
        self.score = {'home': 0, 'away': 0}
        self.outs = 0
        self.runners_on_base = {'1st': None, '2nd': None, '3rd': None}
        self.ball_position = np.array([0.0, 0.0, 0.0])
        self.ball_velocity = np.array([0.0, 0.0, 0.0])

    def step(self, action):
        """Execute action in baseball environment.

        Args:
            action: Action to execute (pitch, swing, field, etc.)

        Returns:
            tuple: (observation, reward, done, info)
        """

        # Execute action based on type
        if action['type'] == 'pitch':
            result = self.execute_pitch(action)
        elif action['type'] == 'swing':
            result = self.execute_swing(action)
        elif action['type'] == 'field':
            result = self.execute_fielding(action)
        elif action['type'] == 'base_running':
            result = self.execute_base_running(action)

        # Update game state
        self.update_game_state(result)

        # Generate observation
        observation = self.generate_observation()

        # Calculate reward
        reward = self.calculate_reward(result)

        # Check if game/half-inning is complete
        done = self.is_game_complete()

        # Additional info
        info = {
            'result': result,
            'game_state': self.get_game_state(),
            'statistics': self.get_current_statistics()
        }

        return observation, reward, done, info

    def execute_pitch(self, action):
        """Execute pitching action."""

        pitcher = action['pitcher']
        pitch_type = action['pitch_type']
        pitch_speed = action['speed']
        pitch_location = action['location']  # x, y coordinates

        # Simulate pitch physics
        trajectory = self.simulate_pitch_trajectory(
            pitch_type, pitch_speed, pitch_location
        )

        # Determine pitch outcome
        outcome = self.determine_pitch_outcome(trajectory)

        return {
            'action_type': 'pitch',
            'trajectory': trajectory,
            'outcome': outcome,
            'pitch_type': pitch_type
        }

    def execute_swing(self, action):
        """Execute batting action."""

        batter = action['batter']
        swing_type = action['swing_type']  # contact, power, bunt, etc.
        swing_timing = action['timing']

        # Simulate swing mechanics
        swing_result = self.simulate_swing_mechanics(
            swing_type, swing_timing
        )

        # Determine contact and ball trajectory
        if swing_result['contact']:
            ball_trajectory = self.simulate_ball_trajectory_after_contact(
                swing_result, self.ball_position, self.ball_velocity
            )

            # Determine hit outcome
            hit_outcome = self.determine_hit_outcome(ball_trajectory)
        else:
            hit_outcome = {'type': 'miss'}

        return {
            'action_type': 'swing',
            'swing_result': swing_result,
            'hit_outcome': hit_outcome,
            'ball_trajectory': ball_trajectory if swing_result['contact'] else None
        }

    def simulate_pitch_trajectory(self, pitch_type, speed, location):
        """Simulate pitch trajectory with physics."""

        # Initial conditions
        initial_position = np.array([0.0, 60.5, 6.0])  # pitcher's mound height
        initial_velocity = self.calculate_initial_pitch_velocity(pitch_type, speed, location)

        # Simulate trajectory
        trajectory = []
        position = initial_position.copy()
        velocity = initial_velocity.copy()
        dt = 0.01  # time step

        while position[1] > 0:  # until ball reaches home plate
            # Update physics
            acceleration = self.calculate_acceleration(velocity)
            velocity += acceleration * dt
            position += velocity * dt

            trajectory.append(position.copy())

            if len(trajectory) > 1000:  # Prevent infinite loops
                break

        return np.array(trajectory)

    def calculate_acceleration(self, velocity):
        """Calculate acceleration including gravity and air resistance."""

        # Gravity
        acceleration = np.array([0.0, 0.0, self.gravity])

        # Air resistance (drag)
        speed = np.linalg.norm(velocity)
        if speed > 0:
            drag_force = -0.5 * self.air_density * self.drag_coefficient * speed**2
            drag_direction = -velocity / speed
            acceleration += drag_force * drag_direction / 0.33  # baseball mass

        # Magnus effect (simplified)
        if hasattr(self, 'spin_rate'):
            magnus_acceleration = self.calculate_magnus_effect(velocity)
            acceleration += magnus_acceleration

        return acceleration

    def determine_pitch_outcome(self, trajectory):
        """Determine outcome of pitch (ball, strike, hit, etc.)."""

        # Get final ball position
        final_position = trajectory[-1]

        # Check if ball crosses home plate
        if abs(final_position[0]) > 17/24:  # 17 inches from center in feet
            return {'type': 'ball'}
        elif final_position[2] < 1.5 or final_position[2] > 3.5:  # strike zone height
            return {'type': 'ball'}
        else:
            return {'type': 'strike'}

    def update_game_state(self, result):
        """Update game state based on action result."""

        if result['action_type'] == 'pitch':
            if result['outcome']['type'] == 'strike':
                self.outs += 1
            # Additional pitch result handling

        elif result['action_type'] == 'swing':
            if result['hit_outcome']['type'] == 'hit':
                self.handle_hit(result['hit_outcome'])
            elif result['hit_outcome']['type'] == 'out':
                self.outs += 1

        # Check for inning changes
        if self.outs >= 3:
            self.advance_half_inning()

    def handle_hit(self, hit_outcome):
        """Handle successful hit outcomes."""

        hit_type = hit_outcome['type']

        if hit_type == 'single':
            self.advance_runners(1)
        elif hit_type == 'double':
            self.advance_runners(2)
        elif hit_type == 'triple':
            self.advance_runners(3)
        elif hit_type == 'home_run':
            self.advance_runners(4)  # All runners score
            self.score[self.half_inning] += 1  # Batter scores

        # Reset outs for new batter
        # (This is simplified - actual baseball has more complex rules)

    def advance_half_inning(self):
        """Advance to next half-inning."""

        if self.half_inning == 'top':
            self.half_inning = 'bottom'
        else:
            self.half_inning = 'top'
            self.inning += 1

        self.outs = 0
        self.runners_on_base = {'1st': None, '2nd': None, '3rd': None}

    def generate_observation(self):
        """Generate observation for agent."""

        return {
            'game_state': self.get_game_state(),
            'ball_position': self.ball_position,
            'ball_velocity': self.ball_velocity,
            'fielders': self.fielders,
            'runners': self.runners_on_base,
            'environmental_factors': {
                'weather': self.weather_conditions,
                'wind': self.wind_speed,
                'temperature': self.temperature
            }
        }

    def get_game_state(self):
        """Get current game state summary."""

        return {
            'inning': self.inning,
            'half_inning': self.half_inning,
            'score': self.score,
            'outs': self.outs,
            'runners_on_base': self.runners_on_base
        }
```

#### Cognitive Baseball Agent
```python
class CognitiveBaseballAgent:
    """Cognitive agent for baseball decision making."""

    def __init__(self, config, role='batter'):
        """Initialize cognitive baseball agent.

        Args:
            config: Agent configuration
            role: Agent role ('batter', 'pitcher', 'fielder', 'manager')
        """

        self.role = role
        self.config = config

        # Cognitive components
        self.perception_system = BaseballPerceptionSystem(config)
        self.belief_system = BaseballBeliefSystem(config)
        self.decision_system = BaseballDecisionSystem(config, role)
        self.learning_system = BaseballLearningSystem(config)

        # Performance tracking
        self.game_history = []
        self.decision_history = []
        self.performance_metrics = {
            'games_played': 0,
            'decisions_made': 0,
            'success_rate': 0.0,
            'average_reward': 0.0
        }

    def make_decision(self, game_state, observation):
        """Make decision based on game state and observation."""

        # Perceive situation
        perceived_situation = self.perception_system.perceive_situation(
            game_state, observation
        )

        # Update beliefs
        self.belief_system.update_beliefs(perceived_situation)

        # Generate decision options
        decision_options = self.decision_system.generate_options(
            perceived_situation, self.belief_system.get_current_beliefs()
        )

        # Evaluate options
        evaluated_options = []
        for option in decision_options:
            evaluation = self.evaluate_decision_option(option, perceived_situation)
            evaluated_options.append((option, evaluation))

        # Select best option
        best_option = max(evaluated_options, key=lambda x: x[1])[0]

        # Record decision
        self.record_decision(best_option, perceived_situation)

        return best_option

    def evaluate_decision_option(self, option, situation):
        """Evaluate a decision option."""

        # Simplified evaluation based on expected outcomes
        if self.role == 'batter':
            return self.evaluate_batting_option(option, situation)
        elif self.role == 'pitcher':
            return self.evaluate_pitching_option(option, situation)
        elif self.role == 'fielder':
            return self.evaluate_fielding_option(option, situation)

        return 0.5  # Default neutral evaluation

    def evaluate_batting_option(self, option, situation):
        """Evaluate batting decision option."""

        swing_type = option.get('swing_type', 'no_swing')

        if swing_type == 'no_swing':
            # Decision not to swing - risk of strike
            strike_probability = self.estimate_strike_probability(situation)
            return 1.0 - strike_probability  # Higher value for avoiding bad pitches

        else:
            # Decision to swing - evaluate hit probability
            hit_probability = self.estimate_hit_probability(option, situation)
            expected_bases = self.estimate_expected_bases(option, situation)

            # Combine hit probability and expected advancement
            return hit_probability * (expected_bases + 1)

    def estimate_strike_probability(self, situation):
        """Estimate probability that pitch will be a strike."""

        ball_position = situation.get('ball_position', np.array([0.0, 0.0, 0.0]))
        ball_velocity = situation.get('ball_velocity', np.array([0.0, 0.0, 0.0]))

        # Simple strike zone model
        x_pos = ball_position[0]  # horizontal position
        z_pos = ball_position[2]  # vertical position

        # Strike zone boundaries (approximate)
        if -0.7 < x_pos < 0.7 and 1.5 < z_pos < 3.5:
            return 0.8  # Likely strike
        else:
            return 0.2  # Likely ball

    def estimate_hit_probability(self, option, situation):
        """Estimate probability of successful hit."""

        swing_type = option.get('swing_type', 'contact')
        timing = option.get('timing', 0.0)  # -1 to 1 (early to late)

        # Base hit probability
        base_probability = {
            'contact': 0.25,
            'power': 0.15,
            'bunt': 0.35
        }.get(swing_type, 0.20)

        # Adjust for timing (optimal timing = 0.0)
        timing_penalty = abs(timing) * 0.5
        adjusted_probability = base_probability * (1.0 - timing_penalty)

        # Adjust for pitch location and speed
        location_factor = self.calculate_location_factor(situation)
        speed_factor = self.calculate_speed_factor(situation)

        final_probability = adjusted_probability * location_factor * speed_factor

        return np.clip(final_probability, 0.0, 1.0)

    def calculate_location_factor(self, situation):
        """Calculate hit probability factor based on pitch location."""

        ball_pos = situation.get('ball_position', np.array([0.0, 0.0, 0.0]))

        # Distance from sweet spot (center of strike zone)
        distance_from_center = np.sqrt(ball_pos[0]**2 + (ball_pos[2] - 2.5)**2)

        # Closer to center = higher probability
        location_factor = max(0.5, 1.0 - distance_from_center * 0.3)

        return location_factor

    def calculate_speed_factor(self, situation):
        """Calculate hit probability factor based on pitch speed."""

        velocity = situation.get('ball_velocity', np.array([0.0, 0.0, 0.0]))
        speed = np.linalg.norm(velocity)

        # Optimal speed range (80-95 mph ≈ 117-139 ft/s)
        if 117 < speed < 139:
            speed_factor = 1.0
        elif speed < 117:
            speed_factor = 0.8  # Slower pitches harder to time
        else:
            speed_factor = 0.6  # Faster pitches harder to react to

        return speed_factor

    def estimate_expected_bases(self, option, situation):
        """Estimate expected bases advanced on hit."""

        swing_type = option.get('swing_type', 'contact')

        # Expected bases by swing type
        expected_bases = {
            'contact': 1.2,  # Singles and some doubles
            'power': 2.5,    # Extra bases, home runs
            'bunt': 0.8      # Sacrifice, short hits
        }.get(swing_type, 1.0)

        # Adjust for fielding and situation
        fielding_factor = self.estimate_fielding_factor(situation)
        situation_factor = self.estimate_situation_factor(situation)

        adjusted_bases = expected_bases * fielding_factor * situation_factor

        return adjusted_bases

    def learn_from_experience(self, decision, outcome, reward):
        """Learn from decision outcomes."""

        # Store experience
        experience = {
            'decision': decision,
            'outcome': outcome,
            'reward': reward,
            'timestamp': time.time()
        }

        self.game_history.append(experience)

        # Update learning system
        self.learning_system.update_from_experience(experience)

        # Update performance metrics
        self.update_performance_metrics(reward)

    def update_performance_metrics(self, reward):
        """Update performance tracking metrics."""

        self.performance_metrics['decisions_made'] += 1
        self.performance_metrics['games_played'] += 1  # Simplified

        # Rolling average reward
        alpha = 0.1  # Learning rate for average
        current_avg = self.performance_metrics['average_reward']
        self.performance_metrics['average_reward'] = (
            alpha * reward + (1 - alpha) * current_avg
        )

        # Success rate (reward > 0)
        success = 1.0 if reward > 0 else 0.0
        current_success = self.performance_metrics['success_rate']
        self.performance_metrics['success_rate'] = (
            alpha * success + (1 - alpha) * current_success
        )

    def get_performance_summary(self):
        """Get performance summary."""

        return {
            'metrics': self.performance_metrics.copy(),
            'recent_decisions': self.decision_history[-10:],
            'learning_progress': self.learning_system.get_learning_summary()
        }
```

## 📊 Analysis and Visualization

### Baseball Analytics System
```python
class BaseballAnalytics:
    """Analytics and visualization for baseball simulations."""

    def __init__(self):
        self.game_data = []
        self.player_stats = {}
        self.team_stats = {}

    def analyze_game(self, game_data):
        """Analyze baseball game data."""

        analysis = {
            'scoring_summary': self.analyze_scoring(game_data),
            'player_performance': self.analyze_player_performance(game_data),
            'strategic_decisions': self.analyze_strategic_decisions(game_data),
            'game_flow': self.analyze_game_flow(game_data)
        }

        return analysis

    def analyze_scoring(self, game_data):
        """Analyze scoring patterns."""

        innings = {}
        for event in game_data:
            if event['type'] == 'run_scored':
                inning = event['inning']
                if inning not in innings:
                    innings[inning] = 0
                innings[inning] += 1

        return {
            'runs_by_inning': innings,
            'total_runs': sum(innings.values()),
            'scoring_efficiency': self.calculate_scoring_efficiency(innings)
        }

    def analyze_player_performance(self, game_data):
        """Analyze individual player performance."""

        player_stats = {}

        for event in game_data:
            player = event.get('player')
            if player:
                if player not in player_stats:
                    player_stats[player] = {
                        'at_bats': 0, 'hits': 0, 'runs': 0,
                        'rbis': 0, 'outs': 0, 'errors': 0
                    }

                if event['type'] == 'hit':
                    player_stats[player]['hits'] += 1
                    player_stats[player]['at_bats'] += 1
                elif event['type'] == 'out':
                    player_stats[player]['outs'] += 1
                    player_stats[player]['at_bats'] += 1
                elif event['type'] == 'run_scored':
                    player_stats[player]['runs'] += 1
                elif event['type'] == 'rbi':
                    player_stats[player]['rbis'] += event['rbis']

        # Calculate advanced stats
        for player, stats in player_stats.items():
            stats['batting_average'] = stats['hits'] / max(stats['at_bats'], 1)
            stats['slugging_percentage'] = self.calculate_slugging(stats)

        return player_stats

    def calculate_slugging(self, stats):
        """Calculate slugging percentage (simplified)."""

        # This would be more complex with actual hit types
        return stats['hits'] / max(stats['at_bats'], 1)

    def visualize_pitch_trajectory(self, trajectory, outcome):
        """Visualize pitch trajectory."""

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', linewidth=2)

        # Plot strike zone
        self.plot_strike_zone(ax)

        # Mark outcome
        final_pos = trajectory[-1]
        if outcome['type'] == 'strike':
            color = 'red'
        else:
            color = 'green'
        ax.scatter(final_pos[0], final_pos[1], final_pos[2], c=color, s=100)

        ax.set_xlabel('X Position (ft)')
        ax.set_ylabel('Distance from Home Plate (ft)')
        ax.set_zlabel('Height (ft)')
        ax.set_title('Pitch Trajectory Visualization')

        return fig

    def plot_strike_zone(self, ax):
        """Plot strike zone boundaries."""

        # Strike zone rectangle
        x = [-0.708, 0.708, 0.708, -0.708, -0.708]
        y = [0, 0, 0, 0, 0]  # At home plate
        z = [1.5, 1.5, 3.5, 3.5, 1.5]

        ax.plot(x, y, z, 'k--', alpha=0.5)

        # Fill strike zone
        verts = [(x[i], y[i], z[i]) for i in range(len(x))]
        ax.add_collection3d(Poly3DCollection([verts], alpha=0.1, color='gray'))
```

## 🎯 Usage Examples

### Running a Baseball Simulation
```python
# Setup environment and agents
environment = BaseballEnvironment()
batter_agent = CognitiveBaseballAgent({'role': 'batter'})
pitcher_agent = CognitiveBaseballAgent({'role': 'pitcher'})

# Run simulation
analytics = BaseballAnalytics()
game_data = []

for inning in range(9):
    for half in ['top', 'bottom']:
        outs = 0
        while outs < 3:
            # Pitcher decision
            game_state = environment.get_game_state()
            pitch_decision = pitcher_agent.make_decision(game_state, environment.generate_observation())

            # Execute pitch
            observation, reward, done, info = environment.step(pitch_decision)
            game_data.append(info['result'])

            # Batter decision
            swing_decision = batter_agent.make_decision(game_state, observation)

            # Execute swing
            observation, reward, done, info = environment.step(swing_decision)
            game_data.append(info['result'])

            outs = info['game_state']['outs']

# Analyze results
analysis = analytics.analyze_game(game_data)
print(f"Final score: {environment.score}")
print(f"Game analysis: {analysis}")
```

## 📚 Related Documentation

### Theoretical Foundations
- [[../../knowledge_base/cognitive/decision_making|Decision Making]]
- [[../../docs/guides/application/|Application Guides]]

### Implementation Resources
- [[../../tools/src/models/|Model Implementations]]
- [[../../docs/examples/|Usage Examples]]

## 🔗 Cross-References

### Core Components
- [[../../Things/Generic_Thing/|Generic Thing Framework]]
- [[../../tools/src/visualization/|Visualization Tools]]

### Related Applications
- [[../../docs/guides/application/|Application Guides]]
- [[../../docs/research/|Research Documentation]]

---

> **Sports Analytics**: This implementation demonstrates cognitive modeling in sports, showing how decision-making under uncertainty applies to baseball strategy.

---

> **Realistic Simulation**: The physics-based simulation provides a realistic testbed for cognitive agents in complex, rule-based environments.

---

> **Multi-Agent Coordination**: The implementation supports multiple agents (batters, pitchers, fielders) working together in a coordinated system.
