import json

import pytest

from icon_solver.collection.http import parse_verdict, submitted_points

REQUEST = json.dumps({"solution": [{"x": 10, "y": 20}, {"x": 30, "y": 40}], "challengeToken": "t"})


def test_status_200_with_token_is_verified():
    verdict = parse_verdict(200, REQUEST, '{"token": "abc"}')
    assert verdict.verified
    assert (verdict.status, verdict.request) == (200, REQUEST)


@pytest.mark.parametrize(
    "status, response",
    [
        (400, '{"error": "I024 ..."}'),
        (200, '{"error": "I024 ..."}'),
        (200, "no es json"),
        (200, None),
        (None, None),
        (502, '{"token": "abc"}'),
    ],
)
def test_anything_else_is_rejected(status, response):
    assert not parse_verdict(status, REQUEST, response).verified


def test_submitted_points_come_from_the_verify_request():
    assert submitted_points(REQUEST) == [{"x": 10, "y": 20}, {"x": 30, "y": 40}]
    assert submitted_points(None) is None
    assert submitted_points('{"sin": "solution"}') is None

