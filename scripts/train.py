import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icon_solver.solver import DEFAULT_MODEL
from icon_solver.training import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="negativos (candidatos que no son ningun icono pedido) por positivo en cada epoch; 0 los desactiva",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL, help="donde guardar el modelo entrenado")
    parser.add_argument("--seed", type=int, default=None, help="fija las semillas para una corrida reproducible")
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        negative_ratio=args.negative_ratio,
        out_path=args.out,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
