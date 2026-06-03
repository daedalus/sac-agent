from __future__ import annotations

import sys

from sac.cli import interactive, main


def _main() -> int:
    if len(sys.argv) > 1:
        main()
    else:
        interactive()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
