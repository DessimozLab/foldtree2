from pathlib import Path
from typing import Optional


def resolve_aapropcsv_path(aapropcsv: Optional[str] = None) -> str:
    """Resolve amino-acid properties CSV path in source and installed layouts."""
    default_csv = Path(__file__).resolve().parents[1] / "config" / "aaindex1.csv"

    if aapropcsv:
        user_path = Path(aapropcsv).expanduser()
        if user_path.exists():
            return str(user_path)

        if user_path.name == "aaindex1.csv" and default_csv.exists():
            return str(default_csv)

        raise FileNotFoundError(
            f"Could not find amino acid properties CSV at '{aapropcsv}'."
        )

    if default_csv.exists():
        return str(default_csv)

    raise FileNotFoundError(
        "Could not find default amino acid properties CSV at "
        f"'{default_csv}'."
    )
