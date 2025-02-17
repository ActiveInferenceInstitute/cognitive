#!/bin/bash

# Exit on error
set -e

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to setup Python virtual environment
setup_venv() {
    echo "Setting up Python virtual environment..."
    
    # Check if python3-venv is installed
    if command_exists apt-get; then
        if ! dpkg -l | grep -q python3-venv; then
            echo "Installing python3-venv..."
            sudo apt-get update
            sudo apt-get install -y python3-venv
        fi
    fi
    
    # Create and activate virtual environment
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    
    # Upgrade pip
    python3 -m pip install --upgrade pip
}

# Function to install system dependencies
install_system_deps() {
    echo "Checking system dependencies..."
    
    if command_exists apt-get; then
        echo "Debian/Ubuntu system detected"
        sudo apt-get update
        sudo apt-get install -y \
            python3-tk \
            python3-qt5 \
            libxcb-cursor0 \
            python3-dev \
            build-essential \
            ffmpeg \
            python3-vtk7 \
            libvtk7-dev \
            mayavi2 \
            python3-mayavi \
            imagemagick
    elif command_exists dnf; then
        echo "Fedora system detected"
        sudo dnf install -y \
            python3-tkinter \
            python3-qt5 \
            xcb-util-cursor \
            python3-devel \
            gcc \
            ffmpeg \
            vtk \
            vtk-devel \
            mayavi \
            ImageMagick
    elif command_exists pacman; then
        echo "Arch system detected"
        sudo pacman -S --noconfirm \
            python-tk \
            qt5-base \
            xcb-util-cursor \
            python-dev \
            base-devel \
            ffmpeg \
            vtk \
            mayavi \
            imagemagick
    else
        echo "Warning: Unknown package manager. Please install dependencies manually if needed."
    fi
    
    # Configure ImageMagick to allow PDF operations if needed
    if command_exists convert; then
        sudo sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml 2>/dev/null || true
        sudo sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-7/policy.xml 2>/dev/null || true
    fi
}

# Function to install Python dependencies
install_python_deps() {
    echo "Installing Python dependencies..."
    
    # First install numpy and other core dependencies
    pip install numpy wheel setuptools
    
    # Install VTK separately if needed
    if ! python3 -c "import vtk" 2>/dev/null; then
        pip install vtk
    fi
    
    # Then install other dependencies
    pip install -r requirements.txt
}

# Function to check installation
check_installation() {
    echo "Checking installation..."
    
    # Try importing required packages
    python3 -c "
import numpy
import torch
import matplotlib
import networkx
import seaborn
import plotly
import imageio
import vtk
import mayavi.mlab
print('All required packages are installed correctly!')
"
}

# Function to check output directory
check_output_dir() {
    echo "Checking output directory..."
    if [ ! -d "output" ]; then
        mkdir output
    fi
    
    # Test write permissions
    if ! touch output/.test 2>/dev/null; then
        echo "Error: Cannot write to output directory"
        exit 1
    fi
    rm output/.test
}

# Function to run the simulation
run_simulation() {
    echo "Running simulation..."
    python example.py
}

# Main execution
echo "=== Path Network Simulation Setup ==="

# Setup virtual environment
setup_venv

# Install dependencies
install_system_deps
install_python_deps

# Check installation
check_installation

# Check output directory
check_output_dir

# Run simulation
echo "=== Starting Simulation ==="
run_simulation

echo "=== Simulation Complete ==="
echo "Check the 'output' directory for results and visualizations." 