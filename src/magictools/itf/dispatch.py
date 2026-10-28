"""Dispatch the shipped ITF candidate-linking commands without heavy imports."""
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "campaign":
        from .campaign import main as command
        return command(argv[1:])
    if argv and argv[0] == "recover":
        from .longarc import main as command
        return command(argv[1:])
    if argv and argv[0] == "replay-growth":
        from .replay_growth import main as command
        return command(argv[1:])
    if argv in (["--help"], ["-h"]):
        print("Additional commands: campaign, recover, replay-growth.\n"
              "Use solve_itf COMMAND --help for each command's options.\n")
    from .cli import main as command
    return command(argv)


if __name__ == "__main__":
    raise SystemExit(main())
