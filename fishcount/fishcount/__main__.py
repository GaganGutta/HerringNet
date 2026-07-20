"""Allow `python -m fishcount` as an alternative to the `fishcount` console script."""

from fishcount.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
