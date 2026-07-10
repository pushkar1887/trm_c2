import sys
import warnings

warnings.warn(
    "check_extraction.py has been refactored into parse.py, solve.py, and oracle_eval.py.\n"
    "Please update your imports. For evaluation tasks, run oracle_eval.py instead.",
    DeprecationWarning,
    stacklevel=2
)

if __name__ == "__main__":
    print("ERROR: check_extraction.py is deprecated. Use oracle_eval.py, parse.py, or solve.py.")
    sys.exit(1)
