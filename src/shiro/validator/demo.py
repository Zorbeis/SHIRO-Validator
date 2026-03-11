from __future__ import annotations

from dataclasses import asdict

from cdm_from_stresslab import generate_default_synthetic_cdm
from cdm_visualizer import visualize_cdm


def main() -> None:
    event = generate_default_synthetic_cdm(seed=42)
    visualize_cdm(event)

    print("\nCDMEvent fields")
    print(asdict(event))


if __name__ == "__main__":
    main()
