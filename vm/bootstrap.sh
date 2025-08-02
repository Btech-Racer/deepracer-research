#!/bin/bash

set -e

trap ctrl_c INT

function ctrl_c() {
    echo "Requested to stop."
    exit 1
}

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
export NEEDRESTART_SUSPEND=1
export DEBCONF_NONINTERACTIVE_SEEN=true
export UCF_FORCE_CONFFOLD=YES

echo "Detecting system distribution..."
distribution=$(
    . /etc/os-release
    echo $ID$VERSION_ID | sed 's/\.//'
)
echo "Detected distribution: $distribution"

IS_WSL2=""
if grep -qi Microsoft /proc/version && grep -q "WSL2" /proc/version; then
    IS_WSL2="yes"
    echo "Detected WSL2 environment"
fi

echo "Detecting cloud environment..."
if [[ -f /var/run/cloud-init/instance-data.json ]]; then
    CLOUD_NAME=$(jq -r '.v1."cloud-name"' /var/run/cloud-init/instance-data.json)
    if [[ "${CLOUD_NAME}" == "azure" ]]; then
        export CLOUD_NAME
        export CLOUD_INSTANCETYPE=$(jq -r '.ds."meta_data".imds.compute."vmSize"' /var/run/cloud-init/instance-data.json)
    elif [[ "${CLOUD_NAME}" == "aws" ]]; then
        export CLOUD_NAME
        export CLOUD_INSTANCETYPE=$(jq -r '.ds."meta-data"."instance-type"' /var/run/cloud-init/instance-data.json)
    else
        export CLOUD_NAME=local
    fi
else
    export CLOUD_NAME=local
fi
echo "Detected cloud type: ${CLOUD_NAME}"

echo "Configuring needrestart for non-interactive mode..."
sudo mkdir -p /etc/needrestart/conf.d/
echo '$nrconf{restart} = "a";' | sudo tee /etc/needrestart/conf.d/50-auto.conf > /dev/null

echo "Configuring apt for non-interactive mode..."
echo 'APT::Get::Assume-Yes "true";' | sudo tee /etc/apt/apt.conf.d/90-assumeyes > /dev/null
echo 'DPkg::Options "--force-confdef";' | sudo tee -a /etc/apt/apt.conf.d/90-assumeyes > /dev/null
echo 'DPkg::Options "--force-confold";' | sudo tee -a /etc/apt/apt.conf.d/90-assumeyes > /dev/null
echo 'DPkg::Options "--force-confnew";' | sudo tee -a /etc/apt/apt.conf.d/90-assumeyes > /dev/null

echo "Starting DeepRacer for Cloud setup..."

echo "Disabling cloud-init to prevent configuration conflicts..."
sudo touch /etc/cloud/cloud-init.disabled || true

echo "Fixing any broken packages..."
yes N | sudo DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true UCF_FORCE_CONFFOLD=YES dpkg --configure -a --force-confold --force-confdef || true

echo "Removing needrestart in Ubuntu 22.04 (if not WSL2)..."
if [[ "${distribution}" == "ubuntu2204" && -z "${IS_WSL2}" ]]; then
    sudo apt remove -y needrestart || true
fi

echo "Updating package list..."
DEBIAN_FRONTEND=noninteractive sudo apt-get update -qq

echo "Fixing cloud-init configuration conflicts..."
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --reinstall cloud-init \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" || true

echo "Upgrading packages..."
DEBIAN_FRONTEND=noninteractive sudo apt-get upgrade -y -qq

echo "Installing core packages..."
DEBIAN_FRONTEND=noninteractive sudo apt-get install -y -qq --no-install-recommends jq awscli python3-boto3 screen

echo "Detecting GPU capabilities..."
GPUS=0
ARCH="cpu"
if [[ -z "${IS_WSL2}" ]]; then
    GPUS=$(lspci | awk '/NVIDIA/ && ( /VGA/ || /3D controller/ ) ' | wc -l)
else
    if [[ -f /usr/lib/wsl/lib/nvidia-smi ]]; then
        GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    fi
fi

if [ $? -ne 0 ] || [ $GPUS -eq 0 ]; then
    ARCH="cpu"
    echo "No NVIDIA GPU detected. Will not install GPU drivers."
else
    ARCH="gpu"
    echo "NVIDIA GPU detected. GPU count: $GPUS"
fi

echo "Installing Python build dependencies..."
DEBIAN_FRONTEND=noninteractive sudo apt-get install -y -qq build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python3-openssl sed jq awscli xserver-xorg-dev xutils-dev

echo "Installing X.Org packages for headless GPU acceleration..."
DEBIAN_FRONTEND=noninteractive sudo apt-get install -y -qq xinit xserver-xorg-legacy x11-xserver-utils x11-utils \
    menu mesa-utils xterm mwm x11vnc pkg-config screen

echo "Setting up environment variables..."
export HOME="${HOME:-/home/$(whoami)}"
export PYENV_ROOT="$HOME/.pyenv"

echo "Installing pyenv..."
curl https://pyenv.run | bash

echo "Configuring pyenv..."
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

echo "Installing Python 3.12..."
pyenv install 3.12.0

echo "Setting Python 3.12 as global version..."
pyenv global 3.12.0

python --version

echo "Installing Python packages..."
python -m pip install pyyaml boto3

echo "Stopping Docker service..."
sudo service docker stop

echo "Setting Docker socket permissions..."
sudo chmod 666 /var/run/docker.sock

echo "Starting Docker service..."
sudo service docker start

echo "Configuring Docker daemon..."
sudo mkdir -p /etc/docker

echo "Adding user to docker group..."
sudo usermod -a -G docker $(id -un)

echo "Creating X authority file..."
touch ~/.Xauthority

echo "Cloning DeepRacer for Cloud repository..."
git clone https://github.com/aws-deepracer-community/deepracer-for-cloud.git

echo "Configuring X.Org for headless GPU acceleration..."
sudo sed -i -e "s/console/anybody/" /etc/X11/Xwrapper.config

echo "Bootstrap setup completed!"
echo "Detected configuration:"
echo "  Distribution: $distribution"
echo "  Cloud: $CLOUD_NAME"
echo "  Architecture: $ARCH"
echo "  WSL2: ${IS_WSL2:-no}"

CLOUD_INIT=$(pstree -s $BASHPID | awk /cloud-init/ | wc -l)

if [[ "${CLOUD_INIT}" -ne 0 ]]; then
    echo "Running in cloud-init environment. Rebooting in 5 seconds to complete driver installation."
    sleep 5s
    sudo shutdown -r +1
elif [[ -n "${IS_WSL2}" || "${ARCH}" == "cpu" ]]; then
    echo "Setup complete. Log out and log back in to ensure all changes take effect."
    echo "If using GPU features, you may need to restart your WSL2 instance or reboot your system."
else
    echo "Setup complete. Rebooting to load NVIDIA drivers and complete installation."
    sudo reboot
fi

# echo "Initializing DeepRacer for Cloud..."
# cd deepracer-for-cloud
# bin/init.sh -a gpu -c local

# dr-update && dr-update-env && dr-upload-custom-files
# ./utils/start-xorg.sh
