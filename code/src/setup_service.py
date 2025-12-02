# # Copyright (C) KonaAI - All Rights Reserved
"""KonaAI Intelligence Server
This script is the entry point for the KonaAI Intelligence Server.
It allows you to run either the web server or the worker process.
"""
import argparse
import os

from src import external_dependencies
from src.scheduler import main as scheduler_main
from src.utils.conf import ENVIRONMENT_VAR_NAME
from src.webserver import main as webserver_main
from src.worker import main as worker_main


def main():
    """
    Parses command line arguments, validates required environment variable, sets up dependencies,
    and runs either the worker or web server based on the selected mode.
    - Checks for the presence and validity of the environment variable specified by ENVIRONMENT_VAR_NAME.
    - Ensures the environment variable points to a valid directory, creating it if necessary.
    - Sets up external dependencies required for the service.
    - Parses command line arguments to determine whether to run in 'worker' or 'web' mode.
    - If 'web' mode is selected, allows specifying the port (default: 80).
    """
    # check if environment variable is set
    env_var = os.getenv(ENVIRONMENT_VAR_NAME) or os.getenv(
        str(ENVIRONMENT_VAR_NAME).lower()
    )
    if env_var is None:
        print(
            f"Environment variable {ENVIRONMENT_VAR_NAME} is not set. "
            "Please set first and then run the server."
        )
        return

    # environment variable provides a valid path, check if its a directory
    env_var = env_var.replace('"', "").replace("'", "").strip()
    print(f"Environment variable {ENVIRONMENT_VAR_NAME} is set to '{env_var}'. ")
    os.makedirs(env_var, exist_ok=True)
    if not os.path.isdir(env_var):
        print(f"Environment variable {ENVIRONMENT_VAR_NAME} is not a valid directory. ")
        return

    # setup external dependencies
    external_dependencies.setup_dependencies()

    parser = argparse.ArgumentParser(description="KonaAI Intelligence Server")
    parser.add_argument(
        "mode",
        choices=["worker", "web", "scheduler"],
        help="Run the worker or web server or scheduler",
    )
    parser.add_argument(
        "--port", type=int, default=80, help="Port for the web server (default: 80)"
    )
    args = parser.parse_args()

    if args.mode == "worker":
        worker_main()
    elif args.mode == "web":
        webserver_main(port=args.port)
    elif args.mode == "scheduler":
        scheduler_main()


if __name__ == "__main__":
    main()
