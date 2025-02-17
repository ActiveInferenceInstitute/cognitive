import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.signal import hilbert
import pytest
from pathlib import Path

class TestComplexDynamics:
    """Test suite for complex dynamical behaviors."""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup test fixtures."""
        plt.style.use('seaborn')
        sns.set_palette("husl")
        self.output_dir = tmp_path / "figures"
        self.output_dir.mkdir(exist_ok=True)
        
        # Test parameters
        self.dt = 0.01
        self.t_max = 10.0
        self.num_steps = int(self.t_max / self.dt)
        self.time = np.linspace(0, self.t_max, self.num_steps)
        
        # System parameters
        self.omega = 2.0  # Natural frequency
        self.damping = 0.1
        self.amplitude = 1.0
        self.frequency = 2.5
        
        yield
        
        # Cleanup
        plt.close('all')
    
    def simulate_harmonic_motion(self):
        """Simulate harmonic oscillator dynamics."""
        states = np.zeros((self.num_steps, 2))
        states[0] = [1.0, 0.0]  # Initial conditions
        
        for i in range(1, self.num_steps):
            # Simple harmonic motion with damping
            states[i, 0] = states[i-1, 0] + self.dt * states[i-1, 1]
            states[i, 1] = states[i-1, 1] - self.dt * (
                self.omega**2 * states[i-1, 0] + 
                2 * self.damping * states[i-1, 1]
            )
        
        return states
    
    def test_harmonic_motion(self):
        """Test visualization of harmonic motion dynamics."""
        states = self.simulate_harmonic_motion()
        free_energy = self.compute_free_energy(states)
        
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=fig)
        
        # Phase space trajectory
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(states[:, 0], states[:, 1], 'b-', label='Phase trajectory')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Velocity')
        ax1.set_title('Phase Space')
        ax1.grid(True)
        ax1.legend()
        
        # Time series
        ax2 = fig.add_subplot(gs[0, 1:])
        ax2.plot(self.time, states[:, 0], 'b-', label='Position')
        ax2.plot(self.time, states[:, 1], 'r--', label='Velocity')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('State Variables')
        ax2.set_title('Time Evolution')
        ax2.grid(True)
        ax2.legend()
        
        # Energy plot
        ax3 = fig.add_subplot(gs[1, 0])
        kinetic = 0.5 * states[:, 1]**2
        potential = 0.5 * self.omega**2 * states[:, 0]**2
        total = kinetic + potential
        ax3.plot(self.time, kinetic, 'g-', label='Kinetic')
        ax3.plot(self.time, potential, 'r-', label='Potential')
        ax3.plot(self.time, total, 'k--', label='Total')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Energy')
        ax3.set_title('Energy Components')
        ax3.grid(True)
        ax3.legend()
        
        # Free energy evolution
        ax4 = fig.add_subplot(gs[1, 1:])
        if len(free_energy) > 0:  # Only plot if free energy is computed
            ax4.plot(self.time, free_energy, 'b-', label='Free Energy')
            ax4.legend()
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Free Energy')
        ax4.set_title('Free Energy Evolution')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'harmonic_motion_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Assertions to verify the simulation
        assert np.all(np.isfinite(states)), "States contain invalid values"
        assert np.all(np.abs(total - total[0]) < 1e-2), "Energy is not conserved"
    
    def test_driven_oscillator(self):
        """Test visualization of driven oscillator dynamics."""
        states = self.simulate_driven_oscillator()
        
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 2, figure=fig)
        
        # Phase space with driving force
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(states[:, 0], states[:, 1], 'b-', label='Phase trajectory')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Velocity')
        ax1.set_title('Phase Space')
        ax1.grid(True)
        ax1.legend()
        
        # Time series with driving force
        ax2 = fig.add_subplot(gs[0, 1])
        driving_force = self.amplitude * np.sin(self.frequency * self.time)
        ax2.plot(self.time, states[:, 0], 'b-', label='Position')
        ax2.plot(self.time, states[:, 1], 'r--', label='Velocity')
        ax2.plot(self.time, driving_force, 'g:', label='Driving Force')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('State Variables')
        ax2.set_title('Time Evolution')
        ax2.grid(True)
        ax2.legend()
        
        # Power spectrum
        ax3 = fig.add_subplot(gs[1, :])
        frequencies = np.fft.fftfreq(len(states), self.dt)
        spectrum = np.abs(np.fft.fft(states[:, 0]))
        mask = frequencies > 0  # Only show positive frequencies
        ax3.plot(frequencies[mask], spectrum[mask], 'b-', label='Position Spectrum')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Amplitude')
        ax3.set_title('Power Spectrum')
        ax3.grid(True)
        ax3.legend()
        
        # Phase difference analysis
        ax4 = fig.add_subplot(gs[2, 0])
        phase_diff = np.angle(hilbert(states[:, 0])) - np.angle(hilbert(driving_force))
        ax4.plot(self.time, np.unwrap(phase_diff), 'r-', label='Phase Difference')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Phase Difference (rad)')
        ax4.set_title('Phase Relationship')
        ax4.grid(True)
        ax4.legend()
        
        # Response amplitude vs time
        ax5 = fig.add_subplot(gs[2, 1])
        envelope = np.abs(hilbert(states[:, 0]))
        ax5.plot(self.time, envelope, 'r-', label='Response Amplitude')
        ax5.plot(self.time, np.abs(driving_force), 'b--', label='Driving Amplitude')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Amplitude')
        ax5.set_title('Response Amplitude')
        ax5.grid(True)
        ax5.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'driven_oscillator_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Assertions to verify the simulation
        assert np.all(np.isfinite(states)), "States contain invalid values"
        assert np.max(np.abs(states[:, 0])) > 0, "No oscillation detected"
    
    def simulate_driven_oscillator(self):
        """Simulate driven oscillator dynamics."""
        states = np.zeros((self.num_steps, 2))
        states[0] = [0.0, 0.0]  # Initial conditions
        
        for i in range(1, self.num_steps):
            driving_force = self.amplitude * np.sin(self.frequency * self.time[i])
            states[i, 0] = states[i-1, 0] + self.dt * states[i-1, 1]
            states[i, 1] = states[i-1, 1] - self.dt * (
                self.omega**2 * states[i-1, 0] + 
                2 * self.damping * states[i-1, 1] - 
                driving_force
            )
        
        return states
    
    def compute_free_energy(self, states):
        """Compute free energy for the system."""
        # Placeholder - implement actual free energy computation
        return np.zeros_like(self.time)

class TestGeneralizedCoordinates:
    """Test suite for generalized coordinates."""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup test fixtures."""
        plt.style.use('seaborn')
        sns.set_palette("husl")
        self.output_dir = tmp_path / "figures"
        self.output_dir.mkdir(exist_ok=True)
        
        yield
        
        plt.close('all')
    
    def test_generalized_coordinates_consistency(self):
        """Test consistency of generalized coordinates predictions."""
        # ... existing test code ...
        
        # Plotting
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot position predictions
        lines1 = ax1.plot(time_points, positions, 'b-', label='Actual')
        ax1.plot(time_points, predicted_positions, 'r--', label='Predicted')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position')
        ax1.set_title('Position Prediction')
        if lines1:  # Only add legend if there are plotted lines
            ax1.legend()
        ax1.grid(True)
        
        # Plot velocity predictions
        lines2 = ax2.plot(time_points, velocities, 'b-', label='Actual')
        ax2.plot(time_points, predicted_velocities, 'r--', label='Predicted')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity')
        ax2.set_title('Velocity Prediction')
        if lines2:  # Only add legend if there are plotted lines
            ax2.legend()
        ax2.grid(True)
        
        # Plot acceleration predictions
        lines3 = ax3.plot(time_points, accelerations, 'b-', label='Actual')
        ax3.plot(time_points, predicted_accelerations, 'r--', label='Predicted')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Acceleration')
        ax3.set_title('Acceleration Prediction')
        if lines3:  # Only add legend if there are plotted lines
            ax3.legend()
        ax3.grid(True)
        
        # Plot prediction errors
        lines4 = []
        if len(position_errors) > 0:
            lines4.extend(ax4.plot(time_points[1:], position_errors, 'r-', label='Position Error'))
        if len(velocity_errors) > 0:
            lines4.extend(ax4.plot(time_points[1:], velocity_errors, 'b--', label='Velocity Error'))
        if len(acceleration_errors) > 0:
            lines4.extend(ax4.plot(time_points[1:], acceleration_errors, 'g:', label='Acceleration Error'))
        
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Prediction Error')
        ax4.set_title('Prediction Errors')
        if lines4:  # Only add legend if there are plotted lines
            ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'generalized_coordinates_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Assertions
        assert np.all(np.isfinite(positions)), "Invalid position values"
        assert np.all(np.isfinite(velocities)), "Invalid velocity values"
        assert np.all(np.isfinite(accelerations)), "Invalid acceleration values" 