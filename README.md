# DeepRacer Research Library

## 🎓 Research Project

**Author:** Bartlomiej Mlynarkiewicz
**Student ID:** 17241782
**Supervisor:** Dr. Tony Scanlan
**Course:** Artificial Intelligence - MSc
**Date:** May 28, 2025

## 📋 Project Overview

A Python library for exploring deep learning architectures and performance optimisation techniques specifically designed for AWS DeepRacer autonomous racing environments.

### 🎯 Research Objectives

- **Advanced Neural Network Architectures**: CNNs with attention mechanisms, Soft Actor-Critic (SAC), multi-modal sensor fusion
- **Performance Optimization Techniques**: Bayesian hyperparameter optimization, transfer learning, curriculum learning, model compression
- **Advanced Training Methodologies**: Meta-learning, multi-task learning, domain randomization, continual learning

## 🏗️ Library Structure

```
deepracer_research/
├── __init__.py                     # Main package exports
├── cli.py                          # Command-line interface
├── architectures/                  # Neural network architectures
│   ├── attention_modules.py        # Attention mechanisms & transformers
│   ├── residual_blocks.py          # ResNet components
│   ├── advanced_architectures.py   # EfficientNet, SAC, Vision Transformers
│   └── factory.py                  # Architecture factory
├── config/                         # Configuration management
│   ├── aws/                        # AWS DeepRacer configurations
│   │   ├── action_space_config.py  # Action space definitions
│   │   ├── scenario_action_spaces.py # Scenario-optimized action spaces
│   │   ├── sensor_config.py        # Sensor configurations
│   │   └── types/                  # Configuration enums
│   │       ├── action_space_type.py # Continuous/Discrete action spaces
│   │       └── sensor_type.py      # Camera, LIDAR, stereo sensors
│   ├── network/                    # Neural network configurations
│   │   ├── architecture_type.py    # Network architecture types
│   │   ├── neural_network_type.py  # 3-layer vs 5-layer CNNs
│   │   └── racing_configs.py       # Racing scenario configurations
│   ├── training/                   # Training configurations
│   │   ├── training_algorithm.py   # PPO, Clipped PPO, SAC, TD3
│   │   ├── loss_type.py            # MSE, Huber, Categorical loss
│   │   └── exploration_strategy.py # Categorical, Epsilon-greedy
│   └── track/                      # Track configurations
│       └── track_type.py           # Official, Championship, Custom tracks
├── deployment/                     # Multi-platform deployment
│   ├── aws_ec2/                    # AWS EC2 deployment
│   │   ├── config/                 # EC2 instance configurations
│   │   ├── management/             # Instance lifecycle management
│   │   └── cli.py                  # EC2 deployment CLI
│   ├── nvidia_brev/                # NVIDIA Brev deployment
│   │   ├── config/                 # Brev instance configurations
│   │   ├── management/             # Deployment automation
│   │   └── cli.py                  # Brev deployment CLI
│   ├── thunder_compute/            # Thunder Compute deployment
│   │   ├── config/                 # Thunder instance configurations
│   │   ├── management/             # Training job management
│   │   └── cli.py                  # Thunder deployment CLI
│   ├── deepracer/                  # AWS DeepRacer console integration
│   │   ├── config/                 # DeepRacer configurations
│   │   └── deployment_manager.py   # Console deployment automation
│   └── templates/                  # Environment file templates
│       ├── template-system.env.j2  # System environment template
│       ├── template-run.env.j2     # Run environment template
│       └── template-worker.env.j2  # Worker environment template
├── experiments/                    # Experimental design framework
│   ├── config/                     # Experiment configurations
│   ├── enums/                      # Experimental scenarios
│   └── evaluation/                 # Performance analysis
├── models/                         # Model management
│   └── build/                      # Model builders and converters
├── optimization/                   # Performance optimization
│   ├── gpu_utils.py                # GPU optimization utilities
│   └── model_optimizer.py          # Model compression & optimization
├── rewards/                        # Reward function framework
│   ├── templates/                  # Optimized reward function templates
│   │   ├── centerline_following.yaml    # Beginner-friendly template
│   │   ├── speed_optimization.yaml      # High-speed racing template
│   │   ├── object_avoidance.yaml        # Obstacle navigation template
│   │   ├── head_to_head.yaml           # Competitive racing template
│   │   └── time_trial.yaml             # Time optimization template
│   ├── parameters/                 # Reward function parameters
│   └── builder.py                  # Reward function builder
├── training/                       # Training pipeline management
│   ├── management/                 # Training job orchestration
│   └── monitoring/                 # Training progress monitoring
└── utils/                          # Utility functions
    ├── logger.py                   # Advanced logging
    ├── aws_config.py               # AWS configuration helpers
    └── s3_utils.py                 # S3 bucket management
```

## 🛠️ Environment Setup

This project uses **Poetry** for dependency management and **Taskfile** for task automation, providing superior package resolution and reproducible environments.

### Prerequisites

- Python 3.12+
- Git
- Task (optional, for task automation)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Btech-Racer/notebooks.git
   cd notebooks
   ```

2. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install Task** (optional, for task automation):
   ```bash
   # macOS
   brew install go-task

   # Or download from https://taskfile.dev/installation/
   ```

4. **Install dependencies:**
   ```bash
   poetry install --with dev,research,aws
   ```

5. **Install pre-commit hooks:**
   ```bash
   poetry run pre-commit install
   ```

6. **Activate the environment:**
   ```bash
   poetry shell
   ```

## 🚀 Quick Start

### Using the CLI

The library provides a modern CLI built with Typer for comprehensive DeepRacer workflow management:

```bash
# Show system information
deepracer-research info

# List available configurations
deepracer-research list-configs

# Create reward functions from templates
deepracer-research rewards create-from-racing-scenario centerline_following
deepracer-research rewards create-from-racing-scenario head_to_head

# List available tracks and algorithms
deepracer-research deploy list-tracks
deepracer-research deploy list-algorithms

# Deploy to different platforms
deepracer-research deploy deepracer my_model \
  --reward-function rewards/speed_optimization/reward_function.py \
  --algorithm clipped_ppo \
  --sensors "FRONT_FACING_CAMERA"

# Multi-platform deployment options
deepracer-research deploy aws my_model --sensors "SECTOR_LIDAR,FRONT_FACING_CAMERA"
```

### Sensor Configuration

The framework supports advanced sensor configurations with validation:

```bash
# Single sensor (recommended for beginners)
--sensors "FRONT_FACING_CAMERA"

# Multi-sensor for advanced scenarios
--sensors "SECTOR_LIDAR,FRONT_FACING_CAMERA"

# Invalid combinations (will be rejected)
--sensors "LIDAR,SECTOR_LIDAR"  # ❌ Conflicting LIDAR types
--sensors "STEREO_CAMERAS,FRONT_FACING_CAMERA"  # ❌ Conflicting camera types
```

**Supported Sensor Types:**
- `FRONT_FACING_CAMERA` - Recommended for time trials (fastest training, 1.0x multiplier)
- `SECTOR_LIDAR` - Efficient object detection with balanced performance (1.7x multiplier)
- `LIDAR` - Full 360° sensing for complex environments (2.0x multiplier)
- `STEREO_CAMERAS` - Depth perception for advanced scenarios (1.5x multiplier)

**Sensor Validation Rules:**
- ✅ `FRONT_FACING_CAMERA` (single sensor, most common)
- ✅ `SECTOR_LIDAR,FRONT_FACING_CAMERA` (multi-modal for advanced scenarios)
- ❌ `LIDAR,SECTOR_LIDAR` (conflicting LIDAR types)
- ❌ `STEREO_CAMERAS,FRONT_FACING_CAMERA` (conflicting camera types)
- ❌ `FRONT_FACING_CAMERA_LIDAR,LIDAR` (complete config conflicts with individual sensors)

### Complete Example Workflow

```bash
# 1. Setup environment
poetry install --with dev,research,aws
poetry shell

# 2. Create reward function for competitive racing
poetry run deepracer-research rewards create-from-racing-scenario head_to_head

# 3. Validate sensor configuration
poetry run python -c "
from deepracer_research.config.aws.types.sensor_type import SensorType
sensors = SensorType.parse_sensor_list('SECTOR_LIDAR,FRONT_FACING_CAMERA')
print('✅ Valid sensors:', [s.value for s in sensors])
"

# 4. Generate complete model files
poetry run deepracer-research deploy deepracer competitive_model \
  --reward-function rewards/head_to_head/reward_function.py \
  --algorithm clipped_ppo \
  --sensors "SECTOR_LIDAR,FRONT_FACING_CAMERA" \
  --files-only

# 5. Check generated files
ls -la models/competitive_model/
# Output:
# hyperparameters.json    - Optimized training parameters
# model_metadata.json     - Model configuration with sensor setup
# reward_function.py      - Generated reward function
# system.env             - System environment variables
# run.env                - Runtime configuration
# worker.env             - Worker configuration

# 6. Deploy to Thunder Compute for training
poetry run thunder-compute deploy-training competitive_model \
  --reward-function rewards/head_to_head/reward_function.py \
  --sensors "SECTOR_LIDAR,FRONT_FACING_CAMERA" \
  --race-type HEAD_TO_HEAD \
  --gpu-type a100-xl
```

### Using Tasks

If you have Task installed, use the convenient task commands:

```bash
# Development setup
task init-project

# Quick deployment workflow
task quick-test

# Create reward functions
task reward-centerline
task reward-speed
task reward-head-to-head

# Deployment to different platforms
task deploy-files MODEL_NAME=my_model SENSORS=FRONT_FACING_CAMERA
task deploy-aws MODEL_NAME=production_model SENSORS=SECTOR_LIDAR
task deploy-thunder MODEL_NAME=test_model SENSORS=FRONT_FACING_CAMERA

# Sensor validation and information
task validate-sensors SENSORS=SECTOR_LIDAR,FRONT_FACING_CAMERA
task sensor-info

# Development tools
task test
task format
task cli-info
```

### Using Python API

```python
from deepracer_research.architectures import ArchitectureFactory
from deepracer_research.config import RACING_CONFIGS

# Create a high-speed racing model
config = RACING_CONFIGS['high_speed_racing']
model = ArchitectureFactory.create_model(config)

# Compile and summary
model.compile(optimizer='adam', loss='mse')
model.summary()
```

## 📦 Dependency Groups

The project organizes dependencies into logical groups:

#### Core Dependencies
- **Deep Learning**: TensorFlow 2.17+, Keras
- **Scientific Computing**: NumPy, Pandas, SciPy, Matplotlib
- **Computer Vision**: OpenCV, PIL, Albumentations
- **CLI Framework**: Typer, Rich (for beautiful CLI output)
- **Utilities**: Click, Pydantic

#### Development Dependencies (`--with dev`)
- **Testing**: pytest, pytest-cov, hypothesis
- **Code Quality**: black, isort, flake8, mypy, bandit
- **Documentation**: Sphinx, sphinx-rtd-theme

#### Research Dependencies (`--with research`)
- **Advanced ML**: Transformers, Stable-Baselines3, Ray[RLLib]
- **Optimization**: Optuna, Hyperopt, Scikit-Optimize
- **Distributed Computing**: Dask, Distributed

#### AWS Dependencies (`--with aws`)
- **AWS Services**: Boto3, SageMaker SDK, AWS CLI
- **Development Tools**: AWS SAM CLI
- **Testing**: Moto (AWS service mocking)

## � Documentation

### CLI Documentation
See [CLI_README.md](CLI_README.md) for detailed CLI usage and examples.

### Available Tasks
Run `task --list` to see all available tasks:

```bash
# Development
task setup              # Setup development environment
task test               # Run tests
task format             # Format code
task lint               # Run linting
task type-check         # Run type checking

# Library Usage
task sample-model       # Create a sample model
task cli-info          # Show CLI information
task cli-benchmark     # Benchmark model creation
task validate-config   # Validate configurations

# Notebooks
task notebook          # Start Jupyter Lab
task notebook-clean    # Clean notebook outputs
```

## 🏗️ Architecture

### Neural Network Architectures

The library provides 8 sophisticated architectures:

1. **Attention CNN** - For high-speed racing scenarios
2. **Multi-Scale CNN** - For obstacle avoidance
3. **SAC Policy** - For continuous control
4. **EfficientNet** - For time trials
5. **ResNet** - For head-to-head racing
6. **Temporal CNN** - For motion-aware racing
7. **Vision Transformer** - For global understanding
8. **Multi-Modal** - For multi-sensor fusion

### Usage Examples

```python
from deepracer_research.architectures import ArchitectureFactory
from deepracer_research.config import RACING_CONFIGS

# High-speed racing with attention
config = RACING_CONFIGS['high_speed_racing']
model = ArchitectureFactory.create_model(config)

# Multi-sensor fusion
config = RACING_CONFIGS['multi_modal_perception']
model = ArchitectureFactory.create_model(config)
```

## 🔧 Development

### Essential Commands

```bash
# Using Poetry directly
poetry shell                    # Activate virtual environment
poetry install --with dev       # Install with dev dependencies
poetry run pre-commit install   # Setup pre-commit hooks
poetry run deepracer-research   # Use CLI
poetry run pytest              # Run tests

# Using Task (recommended)
task setup                      # Complete setup
task test                       # Run tests
task sample-model              # Create sample model
task cli-info                  # CLI information
```

## 🚀 Multi-Platform Deployment

The framework supports deployment to multiple cloud and local platforms with unified configuration.

### Available Deployment Targets

| Platform | Type | Best For | Cost | Setup Complexity |
|----------|------|----------|------|------------------|
| **AWS DeepRacer Console** | Managed Service | Production training | 💰💰💰 | ⭐ |
| **AWS EC2** | Cloud Infrastructure | Scalable training | 💰💰 | ⭐⭐ |
| **NVIDIA Brev** | GPU Cloud | Cost-effective GPUs | 💰 | ⭐⭐ |
| **Thunder Compute** | Cloud GPUs | High-performance training | 💰💰 | ⭐⭐ |
| **Local (Docker)** | Local Development | Development & testing | Free | ⭐⭐⭐ |

### AWS DeepRacer Console Deployment

Direct integration with AWS DeepRacer console for production training:

```bash
# Deploy to AWS DeepRacer console
poetry run deepracer-research deploy deepracer my_model \
  --reward-function rewards/speed_optimization/reward_function.py \
  --algorithm clipped_ppo \
  --sensors "FRONT_FACING_CAMERA" \
  --max-speed 4.0 \
  --training-duration 7200

# Files-only mode (prepare for manual upload)
poetry run deepracer-research deploy deepracer my_model \
  --reward-function rewards/head_to_head/reward_function.py \
  --algorithm clipped_ppo \
  --sensors "SECTOR_LIDAR,FRONT_FACING_CAMERA" \
  --files-only
```

### AWS EC2 Deployment

Full control with AWS EC2 instances:

```bash
# Deploy to AWS EC2
poetry run deepracer-research deploy aws my_model \
  --track reInvent2019_track \
  --algorithm clipped_ppo \
  --sensors "FRONT_FACING_CAMERA" \
  --training-time 30 \
  --instance ml.c5.2xlarge

# Generate files only
poetry run deepracer-research deploy aws my_model \
  --sensors "SECTOR_LIDAR" \
  --files-only
```

### NVIDIA Brev Deployment

Cost-effective GPU cloud with NVIDIA optimization:

```bash
# Deploy to NVIDIA Brev
poetry run nvidia-brev deploy my_model \
  --gpu-type a100 \
  --sensors "FRONT_FACING_CAMERA" \
  --race-type TIME_TRIAL

# Files-only for manual deployment
poetry run nvidia-brev deploy my_model \
  --sensors "LIDAR" \
  --files-only
```

### Thunder Compute Deployment

High-performance cloud training:

```bash
# Deploy to Thunder Compute
poetry run thunder-compute deploy-training my_model \
  --gpu-type a100-xl \
  --sensors "SECTOR_LIDAR,FRONT_FACING_CAMERA" \
  --race-type HEAD_TO_HEAD \
  --workers 4

# Files-only mode
poetry run thunder-compute deploy-training my_model \
  --reward-function rewards/object_avoidance/reward_function.py \
  --sensors "LIDAR" \
  --files-only
```

### File Generation

All platforms support `--files-only` mode to generate complete model configurations:

**Generated Files:**
- `models/{model_name}/hyperparameters.json` - Training hyperparameters
- `models/{model_name}/model_metadata.json` - Model metadata with sensor config
- `models/{model_name}/reward_function.py` - Reward function code
- `models/{model_name}/system.env` - System environment variables
- `models/{model_name}/run.env` - Runtime environment variables
- `models/{model_name}/worker.env` - Worker environment variables

**S3 Bucket Integration:**
- Automatically creates optimized S3 bucket for model storage
- Displays bucket name and URI for manual use
- Configures proper CORS and encryption settings

## ☁️ AWS Integration

### AWS Infrastructure Deployment

The project includes comprehensive AWS integration for training and deployment.

#### Prerequisites

1. **AWS CLI Configuration**:
   ```bash
   aws configure
   # Enter your AWS Access Key ID, Secret Access Key, region, and output format
   ```

2. **Install AWS Dependencies**:
   ```bash
   poetry install --with aws
   ```

#### Deployment Commands

```bash
# Activate Poetry environment
poetry shell

# Deploy infrastructure (with default settings)
./infrastructure/deploy.py deploy

# Deploy with custom parameters
./infrastructure/deploy.py deploy \
    --project my-deepracer-project \
    --env development \
    --suffix unique-suffix \
    --region us-west-2

# Check deployment status
./infrastructure/deploy.py status my-deepracer-project-infrastructure

# Validate CloudFormation template
./infrastructure/deploy.py validate

# Delete infrastructure
./infrastructure/deploy.py delete my-deepracer-project-infrastructure
```

#### Infrastructure Components

The CloudFormation template creates:

| Component | Purpose |
|-----------|---------|
| **IAM Roles** | DeepRacer service and SageMaker execution roles |
| **S3 Buckets** | Model storage, training data, and logs (encrypted) |
| **VPC** | Research environment with public/private subnets |
| **Security Groups** | Network access control for DeepRacer instances |
| **CloudWatch** | Monitoring dashboard and log groups |

#### AWS Resources Created

- **S3 Buckets**:
  - `{project}-models-{suffix}-{account-id}`: Trained model storage
  - `{project}-training-data-{suffix}-{account-id}`: Training datasets
  - `{project}-logs-{suffix}-{account-id}`: Training and execution logs

- **IAM Roles**:
  - `{project}-deepracer-service-role`: DeepRacer service permissions
  - `{project}-sagemaker-execution-role`: SageMaker training permissions

- **VPC Infrastructure**:
  - Research VPC with CIDR `10.0.0.0/16`
  - Public subnet: `10.0.1.0/24`
  - Private subnet: `10.0.2.0/24`
  - Internet Gateway and route tables

#### Deployment Script Features

- ✅ **Rich CLI**: Beautiful command-line interface with progress indicators
- ✅ **Validation**: AWS credentials and template validation
- ✅ **Smart Updates**: Detects and applies only necessary changes
- ✅ **Output Display**: Formatted table of stack outputs
- ✅ **Error Handling**: Comprehensive error reporting and rollback

### Custom Project Scripts
poetry run train-model          # Train DeepRacer model
poetry run evaluate-model       # Evaluate model performance
poetry run analyze-logs         # Analyze training logs
```

### Notebook Structure

1. **Environment Setup** - Poetry configuration and dependency verification
2. **Advanced Neural Architectures** - Attention mechanisms, ResNet blocks, multi-modal fusion
3. **Performance Optimization** - Bayesian optimization, curriculum learning, model compression
4. **Advanced Training Strategies** - Meta-learning, multi-task learning, domain randomization
5. **Reward Function Engineering** - Multi-objective optimization, adaptive reward shaping
6. **Experimental Design** - Systematic evaluation across tracks and scenarios
7. **Model Import/Export** - Comprehensive model sharing and backup system
8. **AWS Integration** - Cloud training, evaluation, and deployment
9. **Results Analysis** - Advanced visualization and performance metrics

## 🔬 Research Features

### Sophisticated Deep Learning Models
- **Attention-Enhanced CNNs** with spatial focus mechanisms
- **Temporal Models** for sequential decision making
- **SAC Policy Networks** for continuous action spaces
- **Multi-Modal Fusion** for camera and LiDAR integration

### Performance Optimization Methods
- **Bayesian Hyperparameter Optimization** using Optuna
- **Curriculum Learning** with progressive difficulty
- **Model Compression** through quantization and pruning
- **Transfer Learning** from pre-trained models

### Advanced Training Methodologies
- **Model-Agnostic Meta-Learning (MAML)** for rapid adaptation
- **Multi-Task Learning** for weather/surface variations
- **Domain Randomization** for robustness
- **Continual Learning** with catastrophic forgetting prevention

## 📊 Experimental Design

The research platform supports systematic evaluation across:

- **Multiple Track Types**: 15+ different racing circuits
- **Sensor Modalities**: Camera, LiDAR, and multi-modal configurations
- **Network Architectures**: CNN, LSTM, Attention, and hybrid models
- **Training Scenarios**: Time trials, head-to-head, object avoidance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `poetry install --with dev`
4. Install pre-commit hooks: `poetry run pre-commit install`
5. Run tests: `poetry run pytest`
6. Format code: `poetry run black .`
7. Submit a pull request

## 📚 References

The project builds upon cutting-edge research in:
- Deep Reinforcement Learning (Schulman et al., 2017; Haarnoja et al., 2018)
- Attention Mechanisms (Vaswani et al., 2017; Hu et al., 2018)
- Meta-Learning (Finn et al., 2017; Nichol et al., 2018)
- Computer Vision (He et al., 2016; Tan & Le, 2019)

## 📄 License

MIT License - see LICENSE file for details.

## 📧 Contact

**Bartlomiej Mlynarkiewicz**
Email: bartlomiej.mlynarkiewicz@student.ie
GitHub: [@Btech-Racer](https://github.com/Btech-Racer)

---
