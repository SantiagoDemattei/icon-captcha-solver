import json

import compute
import pytest


def test_solver_points(baseline, solver):
    points, _ = compute.solver_points(solver)
    assert points == baseline["solver_points"]


def test_legend_classes(baseline, icon_model):
    assert compute.legend_classes(icon_model) == baseline["legend_classes"]


def test_val_ids(baseline):
    assert compute.val_ids() == baseline["val_ids"]


def test_candidates_and_truth_regions(baseline):
    assert sorted(compute.segmentation_sample_ids()) == sorted(baseline["segmentation"])
    actual = compute.segmentation(list(baseline["segmentation"]))
    for example_id, expected in baseline["segmentation"].items():
        assert actual[example_id]["candidates"] == expected["candidates"], example_id
        assert actual[example_id]["truth"] == expected["truth"], example_id


def test_encoding(baseline, icon_model):
    assert compute.encoding(icon_model, list(baseline["encoding"])) == baseline["encoding"]


@pytest.mark.slow
def test_build(baseline, built):
    expected = baseline["build"]
    assert built["stdout"] == expected["stdout"]
    assert built["num_entries"] == expected["num_entries"]
    assert built["crops_sha256_by_example"] == expected["crops_sha256_by_example"]
    assert built["manifest_sha256"] == expected["manifest_sha256"]
    assert built["meta"] == expected["meta"]
    assert built["legend_templates"] == expected["legend_templates"]
    assert built["canonical_templates"] == expected["canonical_templates"]


@pytest.mark.slow
def test_derived_labels_match_build(baseline):
    expected = baseline["build"]
    actual = compute.derived_labels()
    assert actual["crops_sha256_by_example"] == expected["crops_sha256_by_example"]
    assert actual["manifest_sha256"] == expected["manifest_sha256"]
    assert actual["num_classes"] == expected["meta"]["num_classes"]
    assert f"({actual['skipped']} descartados" in expected["stdout"]


@pytest.mark.slow
def test_split(baseline, built):
    assert built["split"] == baseline["build"]["split"]


@pytest.mark.slow
def test_solver_metrics(baseline, solver):
    actual = compute.solver_metrics(solver)
    assert actual["result"] == baseline["metrics"]["evaluate_solver"]
    assert actual["stdout"] == baseline["metrics"]["evaluate_solver_stdout"]


@pytest.mark.slow
def test_classifier_metrics(baseline):
    assert compute.classifier_metrics()["stdout"] == baseline["metrics"]["evaluate_stdout"]


def test_solver_needs_only_the_model_artifact(baseline, tmp_path, monkeypatch):
    import shutil

    from icon_solver import paths, solver

    shutil.copy(compute.ICON_MODEL_PATH, tmp_path / "icon_model.pt")
    monkeypatch.setattr(paths, "PROCESSED_DIR", tmp_path / "no-existe")
    isolated = solver.IconSolver(tmp_path / "icon_model.pt")
    key, challenge = next(iter(compute.challenges().items()))
    assert isolated.solve(challenge.background, challenge.legend) == baseline["solver_points"][key]


def test_solve_icon_signature(baseline):
    import inspect

    from icon_solver.solver import solve_icon

    assert list(inspect.signature(solve_icon).parameters) == ["background_bytes", "legend_bytes"]
    key, challenge = next(iter(compute.challenges().items()))
    points = solve_icon(challenge.background_bytes, challenge.legend_bytes)
    assert points == baseline["solver_points"][key]


def _json_shape(value):
    if isinstance(value, dict):
        return {k: _json_shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_shape(v) for v in value[:1]]
    return type(value).__name__


def test_saved_attempts_match_existing_examples(tmp_path):
    from icon_solver.challenges import HUMAN, MODEL, ChallengeStore
    from icon_solver.collection.browser import Attempt, save_attempt
    from icon_solver.collection.http import parse_verdict

    existing = {}
    for challenge in compute.raw_store():
        existing.setdefault(challenge.labeler, compute.RAW_DIR / challenge.id)
    record = json.loads((existing[MODEL] / "model_label.json").read_text())
    verdict = parse_verdict(record["verify_status"], record["verify_request"], record["verify_response"])
    rejected_verdict = parse_verdict(400, record["verify_request"], '{"error": "I024"}')
    labeled, rejected = ChallengeStore(tmp_path / "raw"), ChallengeStore(tmp_path / "live")

    for kind, attempt_verdict in ((HUMAN, verdict), (MODEL, verdict), (MODEL, rejected_verdict)):
        source = existing[kind]
        attempt = Attempt(
            (source / "legend.png").read_bytes(),
            (source / "background.webp").read_bytes(),
            json.loads((source / "points.json").read_text()),
            attempt_verdict,
        )
        saved = save_attempt(attempt, kind, labeled, rejected)
        directory = (labeled if attempt_verdict.verified else rejected).root / saved.id
        names = sorted(p.name for p in directory.iterdir())
        if not attempt_verdict.verified:
            assert names == ["background.webp", "legend.png", "points.json", "result.json"]
            assert _json_shape(json.loads((directory / "result.json").read_text())) == _json_shape(record)
            continue
        assert names == sorted(p.name for p in source.iterdir())
        for name in names:
            if name.endswith(".json"):
                saved_json = json.loads((directory / name).read_text())
                assert _json_shape(saved_json) == _json_shape(json.loads((source / name).read_text())), name
            else:
                assert (directory / name).read_bytes() == (source / name).read_bytes()


def test_recorded_verdicts_parse_as_stored():
    from icon_solver.collection.http import parse_verdict, submitted_points
    from icon_solver.paths import LIVE_BROWSER_DIR

    paths = sorted(LIVE_BROWSER_DIR.glob("*/result.json")) + sorted(compute.RAW_DIR.glob("*/model_label.json"))
    if not paths:
        pytest.skip("sin veredictos grabados en data/live_browser ni data/raw")
    for path in paths:
        record = json.loads(path.read_text())
        verdict = parse_verdict(record["verify_status"], record["verify_request"], record["verify_response"])
        assert verdict.verified == record["verified"], path
        if (path.parent / "points.json").exists() and record["verified"]:
            assert submitted_points(record["verify_request"]) == json.loads((path.parent / "points.json").read_text())
