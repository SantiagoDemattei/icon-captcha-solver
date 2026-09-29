import json
import random
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from playwright.sync_api import FloatRect, Page, Request, Response, sync_playwright

from icon_solver.challenges import HUMAN, Challenge, ChallengeStore

from .http import DUAL_BASE_URL, BotDeflectorTarget, Verdict, parse_verdict, submitted_points
from .targets import DEPORTICK

VERIFY_URL = f"{DUAL_BASE_URL}/icon/verify"
ATTEMPTS_LOG = "attempts.jsonl"


@dataclass(frozen=True)
class VerifyEvent:
    verdict: Verdict
    legend: bytes | None
    background: bytes | None


@dataclass(frozen=True)
class Attempt:
    legend: bytes
    background: bytes
    points: list[dict]
    verdict: Verdict


class ChallengeSession:
    # Toma la leyenda y el fondo del trafico que descarga el propio browser y asocia cada
    # /icon/verify con las imagenes del ultimo challenge recibido antes: si un intento falla, el
    # widget pide otro challenge dentro de la misma pagina y las imagenes cambian.
    def __init__(self, page: Page, target: BotDeflectorTarget = DEPORTICK):
        self.page = page
        self.target = target
        self._legend: bytes | None = None
        self._background: bytes | None = None
        self._events: list[VerifyEvent] = []
        page.on("response", self._on_response)
        page.on("requestfinished", self._on_request_finished)

    def _on_response(self, response: Response) -> None:
        url = response.url.split("?")[0]
        if response.status != 200 or "/icons/" not in url:
            return
        if "/ico/" in url:
            self._legend = response.body()
        elif "/bgnd/" in url:
            self._background = response.body()

    def _on_request_finished(self, request: Request) -> None:
        if request.method != "POST" or request.url.split("?")[0] != VERIFY_URL:
            return
        response = request.response()
        verdict = parse_verdict(
            response.status if response else None, request.post_data, response.text() if response else None
        )
        self._events.append(VerifyEvent(verdict, self._legend, self._background))

    def open(self, wait_until: str, timeout_ms: int) -> None:
        self._legend = self._background = None
        self._events.clear()
        self.page.goto(self.target.page_url, wait_until=wait_until, timeout=timeout_ms)  # type: ignore[arg-type]

    def _wait(self, done: Callable[[], bool], timeout_s: float, poll_ms: int) -> bool:
        deadline = time.time() + timeout_s
        while not done() and time.time() < deadline:
            self.page.wait_for_timeout(poll_ms)
        return done()

    def wait_for_images(self, timeout_s: float) -> tuple[bytes, bytes] | None:
        if not self._wait(lambda: self._legend is not None and self._background is not None, timeout_s, 200):
            return None
        assert self._legend is not None and self._background is not None
        return self._legend, self._background

    def next_verify(self, timeout_s: float, poll_ms: int = 200) -> VerifyEvent | None:
        if not self._wait(lambda: bool(self._events), timeout_s, poll_ms):
            return None
        return self._events.pop(0)


class ChallengeSolver(Protocol):
    def attempt(self, session: ChallengeSession) -> Attempt | None: ...


class HumanSolver:
    # Los puntos son los del `solution` que el widget mando en el verify aceptado, y las imagenes
    # las del challenge que efectivamente se verifico (no el primero de la pagina).
    def __init__(self, timeout_s: float = 180):
        self.timeout_s = timeout_s

    def attempt(self, session: ChallengeSession) -> Attempt | None:
        session.open(wait_until="networkidle", timeout_ms=30000)
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            event = session.next_verify(deadline - time.time(), poll_ms=1000)
            if event is None:
                return None
            points = submitted_points(event.verdict.request)
            if event.verdict.verified and points is not None and event.legend and event.background:
                return Attempt(event.legend, event.background, points, event.verdict)
        return None


class ModelSolver:
    def __init__(self, solve: Callable[[bytes, bytes], list[dict]], think_s: tuple[float, float]):
        self.solve = solve
        self.think_s = think_s

    def attempt(self, session: ChallengeSession) -> Attempt | None:
        session.open(wait_until="domcontentloaded", timeout_ms=60000)
        canvas = session.page.locator("canvas#canvas")
        canvas.wait_for(state="visible", timeout=90000)
        images = session.wait_for_images(30)
        if images is None:
            return None
        legend, background = images
        shown_at = time.time()

        points = self.solve(background, legend)
        session.page.wait_for_timeout(int(max(0.0, random.uniform(*self.think_s) - (time.time() - shown_at)) * 1000))
        box = canvas.bounding_box()
        if box is None:
            return None
        for point in points:
            _human_click(session.page, box, point)
            session.page.wait_for_timeout(random.randint(400, 900))

        # Sin boton de confirmar: el widget manda /icon/verify solo tras el ultimo click.
        event = session.next_verify(20)
        verdict = event.verdict if event else parse_verdict(None, None, None)
        return Attempt(legend, background, points, verdict)


def _human_click(page: Page, canvas_box: FloatRect, point: dict) -> None:
    # El canvas del widget se muestra al mismo tamaño que la imagen de fondo (300x200), asi que
    # un punto de la imagen es el mismo offset dentro del canvas.
    x, y = canvas_box["x"] + point["x"], canvas_box["y"] + point["y"]
    page.mouse.move(x + random.uniform(-40, 40), y + random.uniform(-40, 40), steps=random.randint(8, 15))
    page.mouse.move(x, y, steps=random.randint(10, 20))
    page.wait_for_timeout(random.randint(120, 300))
    page.mouse.click(x, y, delay=random.randint(60, 140))


@contextmanager
def browser_session(target: BotDeflectorTarget = DEPORTICK) -> Iterator[ChallengeSession]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        try:
            yield ChallengeSession(browser.new_context().new_page(), target)
        finally:
            browser.close()


def log_attempt(out_dir: Path, challenge_id: str, verified: bool) -> None:
    # Registro de todos los intentos: la tasa en vivo sin sesgo de seleccion sale de aca, no de
    # contar lo que quedo en data/raw.
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / ATTEMPTS_LOG).open("a") as log:
        log.write(json.dumps({"example": challenge_id, "verified": verified, "timestamp": time.time()}) + "\n")


def save_attempt(attempt: Attempt, labeler: str, labeled: ChallengeStore, rejected: ChallengeStore | None = None) -> Challenge:
    # Un acierto verificado por el server es un challenge etiquetado (por el humano o por el modelo);
    # los rechazados quedan aparte y sin etiqueta: son los casos dificiles, para etiquetar a mano.
    verified = attempt.verdict.verified
    challenge = Challenge(uuid.uuid4().hex, attempt.legend, attempt.background, attempt.points, labeler if verified else None)
    store = labeled if verified else rejected
    if store is None:
        raise ValueError("intento rechazado sin almacen para rechazados")
    store.save(challenge, None if challenge.labeler == HUMAN else attempt.verdict.record())
    return challenge
