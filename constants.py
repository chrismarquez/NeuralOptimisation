import os
from typing import Literal

Environment = Literal["DEV", "PROD"]


def get_env() -> Environment:
    env = os.environ.get("APP_ENV")
    if env == "PROD":
        return "PROD"
    else:
        return "DEV"


def get_config(env: Environment) -> str:
    if env == "PROD":
        return "prod.ini"
    elif env == "DEV":
        return "dev.ini"


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
