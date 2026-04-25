"""GraphForge OpenENV server — uses openenv-core create_app factory."""

from __future__ import annotations

from openenv.core import create_app

from models import GraphAction, GraphObservation
from server.environment import GraphForgeEnvironment

app = create_app(
    GraphForgeEnvironment,
    GraphAction,
    GraphObservation,
    env_name="graphforge",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
