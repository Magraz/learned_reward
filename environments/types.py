from enum import StrEnum
from dataclasses import dataclass


@dataclass
class EnvironmentParams:
    environment: str = None
    n_envs: int = 1
    n_agents: int = 1
    n_objects: int = 3
    reward_mode: str = "dense"
    env_variant: str = None


class EnvironmentEnum(StrEnum):
    ROVER = "rover"
