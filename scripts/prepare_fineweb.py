import sys

from gcw.cli import main


if __name__ == "__main__":
    main(["prepare", "fineweb", *sys.argv[1:]])
