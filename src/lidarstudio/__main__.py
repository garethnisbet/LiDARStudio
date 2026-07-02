"""Interface for ``python -m lidarstudio`` — starts the LidarStudio server."""

from collections.abc import Sequence

from lidarstudio import server

__all__ = ["main"]


def main(args: Sequence[str] | None = None) -> None:
    """Run the LidarStudio server CLI."""
    server.main(args)


if __name__ == "__main__":
    main()
