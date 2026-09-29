import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icon_solver.challenges import HUMAN, ChallengeStore
from icon_solver.collection.browser import HumanSolver, browser_session, save_attempt
from icon_solver.paths import RAW_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--out", type=Path, default=RAW_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    store, human, harvested = ChallengeStore(args.out), HumanSolver(), 0
    with browser_session() as session:
        while harvested < args.count:
            print(f"Resolve el captcha en la ventana del navegador ({harvested}/{args.count} etiquetados)...")
            attempt = human.attempt(session)
            if attempt is not None:
                save_attempt(attempt, HUMAN, store)
                harvested += 1
                print(f"[{harvested}/{args.count}] etiquetado")
    print(f"\nTotal etiquetados: {harvested}")


if __name__ == "__main__":
    main()
