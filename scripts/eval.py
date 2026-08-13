import sys

from gcw.cli import main

if __name__ == "__main__":
    main(["eval", *sys.argv[1:]])
