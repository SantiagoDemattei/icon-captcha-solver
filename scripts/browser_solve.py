import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icon_solver.challenges import MODEL, ChallengeStore, decode_bgr, decode_bgra
from icon_solver.collection.browser import ModelSolver, browser_session, log_attempt, save_attempt
from icon_solver.paths import LIVE_BROWSER_DIR, RAW_DIR
from icon_solver.solver import IconSolver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--delay", type=float, default=3.0, help="segundos entre challenges")
    parser.add_argument("--min-think", type=float, default=2.5)
    parser.add_argument("--max-think", type=float, default=5.0)
    parser.add_argument("--raw", type=Path, default=RAW_DIR, help="donde van los aciertos verificados")
    parser.add_argument("--out", type=Path, default=LIVE_BROWSER_DIR, help="donde van los rechazados")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    solver = IconSolver()
    model = ModelSolver(lambda bg, lg: solver.solve(decode_bgr(bg), decode_bgra(lg)), (args.min_think, args.max_think))
    labeled, rejected = ChallengeStore(args.raw), ChallengeStore(args.out)
    verified = done = 0
    with browser_session() as session:
        for i in range(args.count):
            try:
                attempt = model.attempt(session)
                status = "sin challenge" if attempt is None else ("verificado" if attempt.verdict.verified else "rechazado")
            except Exception as exc:
                attempt, status = None, f"error: {exc}"
            if attempt is not None:
                challenge = save_attempt(attempt, MODEL, labeled, rejected)
                log_attempt(args.out, challenge.id, attempt.verdict.verified)
                done += 1
                verified += attempt.verdict.verified
            print(f"[{i + 1}/{args.count}] {status} (acumulado {verified}/{done})", flush=True)
            if i + 1 < args.count:
                time.sleep(args.delay)
    print(f"\nverificados: {verified}/{done}" + (f" = {verified / done:.1%}" if done else ""))


if __name__ == "__main__":
    main()
