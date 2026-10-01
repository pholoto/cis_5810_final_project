"""Log in interactively, storing the Hugging Face token only inside this checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_detector.config import configure_environment


def main():
    configure_environment()
    from huggingface_hub import login

    login(add_to_git_credential=False)


if __name__ == "__main__":
    main()
