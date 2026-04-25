"""GraphForge AST-query OpenENV server — uses openenv-core create_app factory."""

from __future__ import annotations

from openenv.core import create_app

from models import ASTAction, ASTObservation
from server.ast_environment import ASTEnvironment

app = create_app(
    ASTEnvironment,
    ASTAction,
    ASTObservation,
    env_name="graphforge-ast",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
