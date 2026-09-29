import argparse
import secrets
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icon_solver.challenges import MODEL, Challenge, ChallengeStore
from icon_solver.collection.http import get_icon_challenge, mint_flow_token, new_session
from icon_solver.collection.labeler import label_challenge
from icon_solver.collection.targets import DEPORTICK
from icon_solver.paths import RAW_DIR


def collect_one(session, target, out_dir: Path) -> bool:
    flow_token = mint_flow_token(session, target)
    fingerprint = secrets.token_hex(16)
    challenge = get_icon_challenge(session, flow_token, fingerprint, target)

    labeled = label_challenge(session, challenge, flow_token, fingerprint, target)
    if labeled is None:
        return False
    points, verdict = labeled

    # Los puntos los eligio el ranking automatico, no un humano: como los del modelo, van siempre a
    # entrenamiento y nunca a validacion.
    labeled_challenge = Challenge(uuid.uuid4().hex, challenge.legend_bytes, challenge.background_bytes, points, MODEL)
    ChallengeStore(out_dir).save(labeled_challenge, verdict.record())
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    session = new_session()
    hits = 0
    for i in range(args.count):
        verified = collect_one(session, DEPORTICK, args.out)
        hits += verified
        status = "etiquetado" if verified else "descartado (top-1 incorrecto)"
        print(f"[{i + 1}/{args.count}] {status}")
        if i + 1 < args.count:
            time.sleep(args.delay)

    print(f"\n{hits}/{args.count} etiquetados ({hits / args.count:.1%} yield)")


if __name__ == "__main__":
    main()
