from .http import BotDeflectorTarget

# Organizador demo de Queue-it, sin login-gating ni enqueuetoken -- confirmado en vivo
# el 2026-09-27 que mintea flow tokens con el paso 'icon' presente.
DEPORTICK = BotDeflectorTarget(
    org_domain="deportick.queue-it.net",
    account_id="deportick",
    site_key="avb24092026",
)
