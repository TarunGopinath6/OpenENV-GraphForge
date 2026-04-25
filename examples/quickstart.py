"""Quickstart example for GraphForge environment."""

from server.environment import GraphForgeEnvironment
from models import GraphAction


def main():
    """Run a simple example."""
    # Create environment
    env = GraphForgeEnvironment()
    
    # Reset environment
    observation = env.reset()
    print(f"Initial observation: {observation}")
    
    # Take a step
    action = GraphAction(
        action_type="default",
        parameters={}
    )
    result = env.step(action)
    print(f"Step result: {result}")
    
    # Get state
    state = env.state()
    print(f"Episode state: {state}")


if __name__ == "__main__":
    main()
