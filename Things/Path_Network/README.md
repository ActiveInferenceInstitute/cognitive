# Path Network Simulation

A simulation of Active Inference agents in a dynamic network topology, where agents adapt their heights in response to changing water levels and network influences.

## Features

- **Active Inference Agents**: Each node implements continuous-time active inference
- **Dynamic Network Topology**: Network connections evolve based on agent performance
- **Environmental Dynamics**: Complex water level patterns through nested sinusoidal waves
- **Real-time Visualization**: Interactive visualization of network state and agent behavior
- **Comprehensive Analysis**: Detailed metrics and visualizations of simulation results

## Requirements

### System Dependencies

- Python 3.8+
- Qt5 or Tk (for visualization)
- Required system libraries (installed automatically by setup script)

### Python Dependencies

All Python dependencies are listed in `requirements.txt` and will be installed automatically by the setup script.

## Quick Start

The easiest way to run the simulation is using the provided setup script:

```bash
./run_simulation.sh
```

This script will:
1. Create a Python virtual environment
2. Install required system dependencies
3. Install Python dependencies
4. Run the simulation
5. Save results to the `output` directory

## Manual Setup

If you prefer to set up manually:

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the simulation:
   ```bash
   python example.py
   ```

## Output

The simulation creates a timestamped output directory containing:

- `simulation.log`: Detailed log of the simulation run
- `initial_state.png`: Network state visualization after initial period
- `final_state.png`: Network state visualization after perturbation
- `water_level_distribution.png`: Distribution of water levels
- `agent_height_variations.png`: Time series of agent heights
- `water_levels.npy`: Raw water level data
- `agent_heights.npy`: Raw agent height data

## Visualization

The simulation provides real-time visualization of:
- Network topology with node colors representing heights
- Current agent heights compared to water level
- Historical evolution of heights and water levels

## Configuration

Key parameters can be adjusted in `example.py`:

- Network parameters (number of nodes, connectivity, etc.)
- Wave components (frequencies, amplitudes, phases)
- Simulation duration and visualization intervals
- Perturbation parameters

## Troubleshooting

### Visualization Issues

If you encounter visualization problems:

1. The script will automatically try different backends (Qt5 → Tk → Agg)
2. Check if Qt5 or Tk is properly installed
3. Look for error messages in the simulation log

### System Dependencies

If system dependencies fail to install:

1. Check your system's package manager
2. Install required packages manually:
   - For Ubuntu/Debian: `sudo apt-get install python3-tk python3-qt5 libxcb-cursor0`
   - For Fedora: `sudo dnf install python3-tkinter python3-qt5 xcb-util-cursor`
   - For Arch: `sudo pacman -S python-tk qt5-base xcb-util-cursor`

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 