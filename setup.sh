#!/bin/bash
# Setup script for neuromorphic drone project on a fresh GPU instance (Ubuntu 22.04, RTX GPU)
# Run once after deployment: bash setup.sh
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HOME/isaaclab-env"
ISAACLAB_DIR="$HOME/IsaacLab"

# Accept NVIDIA Omniverse EULA non-interactively
export OMNI_ACCEPT_EULA=Y
export NVIDIA_ACCEPT_EULA=Y

echo "[1/5] Installing system dependencies..."
sudo apt-get update -q
sudo apt-get install -y cmake git python3-venv python3-pip

echo "[2/5] Creating Python virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[3/5] Installing Isaac Sim..."
pip install --upgrade pip
pip install "isaacsim[all]==4.5.0" --extra-index-url https://pypi.nvidia.com

echo "[4/5] Installing Isaac Lab..."
if [ ! -d "$ISAACLAB_DIR" ]; then
    git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi
cd "$ISAACLAB_DIR"
./isaaclab.sh --install

echo "[5/5] Installing project dependencies..."
PROJECT_DIR="$HOME/snn-evs-drone"
if [ ! -d "$PROJECT_DIR" ]; then
    git clone https://github.com/cyber-run/snn-evs-drone.git "$PROJECT_DIR"
fi
pip install -r "$PROJECT_DIR/requirements.txt"

# Persist venv activation in .bashrc
if ! grep -q "isaaclab-env" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Isaac Lab environment" >> ~/.bashrc
    echo "source $VENV_DIR/bin/activate" >> ~/.bashrc
    echo "export ISAACLAB_PATH=$ISAACLAB_DIR" >> ~/.bashrc
fi

echo ""
echo "Setup complete. Run 'source ~/.bashrc' or open a new shell to activate."
echo "Verify with: python -c \"import isaaclab; print('Isaac Lab OK')\""
