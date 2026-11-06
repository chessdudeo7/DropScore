from .cli import main

# Guarded: the eval scores clips in worker processes, and on Windows each one
# imports this module again; unguarded, every worker would start the CLI over.
if __name__ == "__main__":
    raise SystemExit(main())
