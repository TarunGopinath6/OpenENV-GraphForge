---
title: OpenENV GraphForge
emoji: 🌐
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# GraphForge OpenENV

A graph-based optimization and learning environment built on the OpenENV framework.

## Overview

GraphForge is an OpenENV-compliant reinforcement learning environment for graph-based optimization problems. It provides a structured interface for developing and testing RL agents on graph manipulation, traversal, and optimization tasks.

## Features

- **OpenENV Compliant**: Full compatibility with the OpenENV standard interface
- **FastAPI Server**: REST API for environment interaction
- **Web Interface**: Built-in web dashboard for monitoring
- **Docker Ready**: Pre-configured Dockerfile for easy deployment
- **Modular Design**: Extensible task and baseline implementations

## Project Structure

```
.
├── server/                  # FastAPI server implementation
│   ├── __init__.py
│   ├── environment.py       # Main environment class (GraphForgeEnvironment)
│   └── app.py              # FastAPI application
├── tasks/                   # Task definitions
│   ├── __init__.py
│   └── task1.py            # Example task
├── baseline/               # Baseline policy implementations
│   ├── __init__.py
│   └── baseline.py         # Random baseline policy
├── examples/               # Example usage scripts
│   ├── __init__.py
│   └── quickstart.py       # Quickstart example
├── tests/                  # Unit tests
│   ├── __init__.py
│   └── test_environment.py
├── frontend/               # Web interface
│   ├── index.html
│   └── .env.example
├── scripts/                # Utility scripts
├── data/                   # Data directory
├── models.py               # Pydantic models (observation/action spaces)
├── openenv.yaml            # OpenENV configuration
├── pyproject.toml          # Python project configuration
├── requirements.txt        # Dependencies
├── Dockerfile              # Docker configuration
├── .env                    # Environment variables
└── .gitignore              # Git ignore rules
```

## Quick Start

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/OpenENV-GraphForge.git
cd OpenENV-GraphForge
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Server

```bash
python -m server.app
```

The server will start at `http://localhost:7860`

API documentation: `http://localhost:7860/docs`

### Running an Example

```bash
python examples/quickstart.py
```

### Running Tests

```bash
pytest tests/
```

## Environment Interface

### Reset

Reset the environment and get initial observation:

```python
from server.environment import GraphForgeEnvironment

env = GraphForgeEnvironment()
observation = env.reset(seed=42, task="basic_task")
```

### Step

Execute an action in the environment:

```python
from models import GraphAction

action = GraphAction(action_type="modify_node", parameters={})
result = env.step(action)

# Returns StepResult with:
# - observation: GraphObservation
# - reward: float
# - terminated: bool
# - truncated: bool
# - info: dict
```

### State

Get current episode state:

```python
state = env.state()
```

## API Endpoints

### Health Check

- `GET /health` - Returns server health status

### Environment Control

- `POST /reset` - Reset environment
- `POST /step` - Execute action
- `GET /state` - Get current state

### Web Interface

- `GET /` - Main web interface
- `GET /docs` - Swagger UI documentation
- `GET /redoc` - ReDoc documentation

## Configuration

Environment settings are defined in `openenv.yaml`:

- **Environment Class**: `server.environment.GraphForgeEnvironment`
- **Action Space**: `GraphAction`
- **Observation Space**: `GraphObservation`
- **Port**: 7860 (default)

## Deployment

### Docker

Build the Docker image:

```bash
docker build -t graphforge:latest .
```

Run the container:

```bash
docker run -p 7860:7860 graphforge:latest
```

## Development

### Adding New Tasks

Create a new task file in `tasks/`:

```python
class MyTask:
    def __init__(self):
        self.name = "my_task"
        self.difficulty = "medium"

    def sample(self):
        """Sample a task instance"""
        return {}

    def evaluate(self, trajectory):
        """Evaluate a trajectory"""
        return 0.0
```

### Adding Baseline Policies

Extend `baseline.baseline.RandomBaseline` with your policy:

```python
class MyPolicy(RandomBaseline):
    def select_action(self, observation):
        # TODO: Implement your policy
        return GraphAction(...)
```

## Requirements

- Python >= 3.11
- FastAPI >= 0.104.0
- Pydantic >= 2.0.0
- openenv-core >= 0.2.0

See [requirements.txt](requirements.txt) for all dependencies.

## License

BSD-3-Clause

## Contributing

Contributions are welcome! Please follow the project structure and ensure tests pass.

## References

- [OpenENV Specification](https://openenv.org)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
