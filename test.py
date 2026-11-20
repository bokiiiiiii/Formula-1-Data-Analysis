import os
from pathlib import Path

from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv(Path(__file__).with_name(".env"))
    for variable in ("INSTAGRAM_USERNAME", "INSTAGRAM_PASSWORD"):
        status = "configured" if os.environ.get(variable) else "missing"
        print(f"{variable}: {status}")
