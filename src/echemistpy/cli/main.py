"""Compatibility entry point for echemistpy CLI."""

from echemistpy.cli.app import app, main

__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
