from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_RAW: Path = ROOT / "src" / "data" / "raw" / "clients.csv"
    ARTIFACTS: Path = ROOT / "src" / "data" / "processed"

@dataclass(frozen=True)
class Columns:
    TARGET: str = "default"
    SENSITIVE: str = "sex"

@dataclass(frozen=True)
class Split:
    TEST_SIZE: float = 0.3
    RANDOM_STATE: int = 42