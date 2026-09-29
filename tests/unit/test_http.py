import json

from icon_solver.collection.http import IconChallenge, submitted_points, verify_icon
from icon_solver.collection.targets import DEPORTICK

POINTS = [{"x": 1, "y": 2}]


class _Response:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text


class _Session:
    def __init__(self, response: _Response):
        self.response = response
        self.posted: dict = {}

    def post(self, url, json, headers, timeout):
        self.posted = {"url": url, "json": json}
        return self.response


def _verify(response: _Response):
    session = _Session(response)
    challenge = IconChallenge(challenge_token="tok", legend_bytes=b"", background_bytes=b"")
    verdict = verify_icon(session, challenge, "flow", DEPORTICK, POINTS, "fp", solve_time_ms=1500)  # type: ignore[arg-type]
    return session, verdict


def test_verify_returns_the_full_verdict():
    session, verdict = _verify(_Response(200, '{"token": "abc"}'))

    assert verdict.verified
    assert verdict.status == 200
    assert json.loads(verdict.request or "") == session.posted["json"]
    assert submitted_points(verdict.request) == POINTS
    assert set(verdict.record()) == {"verified", "verify_status", "verify_request", "verify_response", "timestamp"}


def test_verify_rejection():
    _, verdict = _verify(_Response(400, '{"error": "I024 ..."}'))

    assert not verdict.verified
    assert verdict.response == '{"error": "I024 ..."}'
