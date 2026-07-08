import sys

from gcw.cli import main


if __name__ == "__main__":
    main(["convert", "hf", *sys.argv[1:]])
