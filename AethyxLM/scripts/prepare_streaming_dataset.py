"""Generic entry point for the storage-bounded dataset preparer.

The original filename is retained for backwards compatibility; new sources
should use this entry point.
"""

from prepare_fineweb import main


if __name__ == "__main__":
    main()
