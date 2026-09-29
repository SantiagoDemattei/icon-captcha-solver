import json
import re
import time
from dataclasses import dataclass

from curl_cffi import requests

from .jwt import decode_payload
from .pow import build_graph, solve_challenges

DUAL_BASE_URL = "https://dual.challenge.queue-it.net"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36 Edg/154.0.0.0"
)
SEC_CH_UA = '"Chromium";v="154", "Microsoft Edge";v="154", "Not A(Brand";v="99"'


@dataclass(frozen=True)
class Verdict:
    verified: bool
    status: int | None
    request: str | None
    response: str | None

    def record(self) -> dict:
        return {
            "verified": self.verified,
            "verify_status": self.status,
            "verify_request": self.request,
            "verify_response": self.response,
            "timestamp": time.time(),
        }


def parse_verdict(status: int | None, request: str | None, response: str | None) -> Verdict:
    # Confirmado en vivo: exito es 200 {"token": ...}, fallo es 400 {"error": "I0XX ..."}. No existe
    # un campo isVerified, y cualquier otra cosa (sin respuesta, cuerpo invalido) es un rechazo.
    verified = False
    if status == 200 and response:
        try:
            verified = "token" in json.loads(response)
        except (TypeError, ValueError):
            verified = False
    return Verdict(verified, status, request, response)


def submitted_points(request: str | None) -> list[dict] | None:
    try:
        return json.loads(request or "")["solution"]
    except (TypeError, ValueError, KeyError):
        return None


@dataclass(frozen=True)
class BotDeflectorTarget:
    org_domain: str
    account_id: str
    site_key: str

    @property
    def origin(self) -> str:
        return f"https://{self.org_domain}"

    @property
    def page_url(self) -> str:
        return f"{self.origin}/?c={self.account_id}&e={self.site_key}"


@dataclass
class IconChallenge:
    challenge_token: str
    legend_bytes: bytes
    background_bytes: bytes


def _dual_headers(target: BotDeflectorTarget) -> dict:
    origin = target.origin
    return {
        "accept": "*/*",
        "accept-language": "es-419,es;q=0.9",
        "content-type": "application/json",
        "dnt": "1",
        "origin": origin,
        "referer": origin + "/",
        "sec-ch-ua": SEC_CH_UA,
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": USER_AGENT,
    }


def _dual_image_headers(target: BotDeflectorTarget) -> dict:
    return {
        "accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "accept-language": "es-419,es;q=0.9",
        "dnt": "1",
        "referer": target.origin + "/",
        "sec-ch-ua": SEC_CH_UA,
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "image",
        "sec-fetch-mode": "no-cors",
        "sec-fetch-site": "same-site",
        "user-agent": USER_AGENT,
    }


def new_session() -> requests.Session:
    return requests.Session(impersonate="chrome124")


def mint_flow_token(session: requests.Session, target: BotDeflectorTarget) -> str:
    # GET anonimo a la challenge page de Queue-it -- sin login, sin enqueuetoken, sin
    # proxy, confirmado en vivo contra deportick/avb24092026 (2026-09-27). Cada llamada
    # devuelve un flow_token nuevo con su propio sessionId.
    response = session.get(target.page_url, headers={"user-agent": USER_AGENT}, timeout=(15, 30))
    match = re.search(r"botDeflectorJwtToken: '([^']*)'", response.text)
    if not match:
        raise RuntimeError("No se encontro botDeflectorJwtToken en la challenge page")
    return match.group(1)


def _solve_pow_step(session: requests.Session, flow_token: str, fingerprint: str, headers: dict) -> None:
    # El challengeFlow es [inv, icon, inv]: el servidor rastrea por sessionId si el primer
    # 'inv' (proof-of-work) ya se resolvio, y devuelve un 502 opaco en /icon/get si no.
    begin = time.time()
    get_response = session.get(
        f"{DUAL_BASE_URL}/pow/get", headers={**headers, "x-tt-flow-token": flow_token}, timeout=(15, 30),
    )
    challenge_token = get_response.json()["challengeToken"]
    payload = decode_payload(challenge_token)

    params = dict(payload["params"])
    params["GRAPH_SEED"] = payload["sessionId"]
    solution = solve_challenges(payload["challenges"], params, graph=build_graph(params))
    duration = time.time() - begin

    verify_body = {
        "solution": solution,
        "params": params,
        "solution_fingerprint": fingerprint,
        "time": f"{duration:.3f}",
        "challengeToken": challenge_token,
        "challengeFlowToken": flow_token,
    }
    session.post(f"{DUAL_BASE_URL}/pow/verify", json=verify_body, headers=headers, timeout=(15, 30))


def get_icon_challenge(
    session: requests.Session, flow_token: str, fingerprint: str, target: BotDeflectorTarget,
) -> IconChallenge:
    headers = _dual_headers(target)
    _solve_pow_step(session, flow_token, fingerprint, headers)
    get_response = session.get(
        f"{DUAL_BASE_URL}/icon/get", headers={**headers, "x-tt-flow-token": flow_token}, timeout=(15, 30),
    )
    challenge_token = get_response.json()["challengeToken"]
    payload = decode_payload(challenge_token)
    lib, ico, img = payload["lib"], payload["ico"], payload["img"]

    image_headers = _dual_image_headers(target)
    legend = session.get(f"{DUAL_BASE_URL}/icons/{lib}/ico/{ico}", headers=image_headers, timeout=(15, 30)).content
    background = session.get(f"{DUAL_BASE_URL}/icons/{lib}/bgnd/{img}", headers=image_headers, timeout=(15, 30)).content
    return IconChallenge(challenge_token=challenge_token, legend_bytes=legend, background_bytes=background)


def verify_icon(
    session: requests.Session,
    challenge: IconChallenge,
    flow_token: str,
    target: BotDeflectorTarget,
    points: list[dict],
    fingerprint: str,
    solve_time_ms: int,
) -> Verdict:
    # Un fallo consume el challengeToken: no se puede reintentar sobre la misma imagen (ver labeler.py).
    headers = _dual_headers(target)
    body = {
        "solution": points,
        "solution_fingerprint": fingerprint,
        "time": solve_time_ms,
        "challengeToken": challenge.challenge_token,
        "challengeFlowToken": flow_token,
    }
    response = session.post(f"{DUAL_BASE_URL}/icon/verify", json=body, headers=headers, timeout=(15, 30))
    return parse_verdict(response.status_code, json.dumps(body), response.text)
