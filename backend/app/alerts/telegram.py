"""Telegram bot: sending alerts (text, photos, albums) and answering /estado.

The bot token is kept in the system keychain (macOS Keychain / Windows Credential Locker,
via ``keyring``), never in a file; the chat id goes to ALERTS_DIR/telegram.json. Only
messages from that chat are answered. Errors never include the request URL, which
contains the token.
"""
from __future__ import annotations

import datetime as dt
import json

import httpx

from ..config import ALERTS_DIR

SERVICE = "forecast-workstation"
ACCOUNT = "telegram-bot"
CONF_FILE = ALERTS_DIR / "telegram.json"
API = "https://api.telegram.org/bot{token}/{method}"


class TelegramError(Exception):
    pass


def token() -> str | None:
    import keyring
    return keyring.get_password(SERVICE, ACCOUNT)


def set_token(tok: str) -> None:
    import keyring
    keyring.set_password(SERVICE, ACCOUNT, tok)


def conf() -> dict:
    try:
        return json.loads(CONF_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_conf(c: dict) -> None:
    CONF_FILE.write_text(json.dumps(c, ensure_ascii=False, indent=1))


def configured() -> bool:
    return bool(conf().get("chat_id")) and bool(token())


async def call(method: str, data: dict | None = None, files: dict | None = None,
               tok: str | None = None, timeout: float = 60.0) -> dict:
    tok = tok or token()
    if not tok:
        raise TelegramError("bot do Telegram por configurar (python -m app.alerts configurar)")
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            if files:
                r = await c.post(API.format(token=tok, method=method), data=data or {}, files=files)
            else:
                r = await c.post(API.format(token=tok, method=method), json=data or {})
    except httpx.HTTPError as e:            # the message would carry the URL (and the token)
        raise TelegramError(f"{method}: falha de rede ({e.__class__.__name__})") from None
    try:
        js = r.json()
    except ValueError:
        raise TelegramError(f"{method}: resposta inválida (HTTP {r.status_code})") from None
    if not js.get("ok"):
        raise TelegramError(f"{method}: {js.get('description', r.status_code)}")
    return js["result"]


def _silent(urgent: bool) -> bool:
    """Quiet hours (Lisbon time): the message arrives without sound unless urgent."""
    from zoneinfo import ZoneInfo

    from .events import load_config
    if urgent:
        return False
    a, b = load_config().get("quiet_hours", [23, 8])
    h = dt.datetime.now(ZoneInfo("Europe/Lisbon")).hour
    return (h >= a or h < b) if a > b else (a <= h < b)


async def send_text(text: str, urgent: bool = False, silent: bool | None = None) -> dict:
    return await call("sendMessage", {
        "chat_id": conf()["chat_id"], "text": text[:4096], "parse_mode": "HTML",
        "link_preview_options": {"is_disabled": True},
        "disable_notification": _silent(urgent) if silent is None else silent})


async def send_photos(photos: list[tuple[str, bytes, str]], silent: bool = True) -> None:
    """[(name, png, caption)] — one photo, or albums of up to 10."""
    chat = conf()["chat_id"]
    if not photos:
        return
    if len(photos) == 1:
        name, png, cap = photos[0]
        await call("sendPhoto", {"chat_id": str(chat), "caption": cap[:1024], "parse_mode": "HTML",
                                 "disable_notification": str(silent).lower()},
                   files={"photo": (f"{name}.png", png, "image/png")})
        return
    for i in range(0, len(photos), 10):
        chunk = photos[i:i + 10]
        media, files = [], {}
        for j, (name, png, cap) in enumerate(chunk):
            media.append({"type": "photo", "media": f"attach://f{j}", "caption": cap[:1024],
                          "parse_mode": "HTML"})
            files[f"f{j}"] = (f"{name}.png", png, "image/png")
        await call("sendMediaGroup", {"chat_id": str(chat), "media": json.dumps(media),
                                      "disable_notification": str(silent).lower()}, files=files,
                   timeout=120)


async def updates(offset: int) -> list[dict]:
    return await call("getUpdates", {"offset": offset, "timeout": 0,
                                     "allowed_updates": ["message"]}, timeout=30)


# ---------------------------------------------------------------- set-up (interactive)
async def configure() -> None:
    """`python -m app.alerts configurar`: asks for the token, finds the chat, sends a test."""
    import asyncio
    import getpass
    print("Configuração do bot do Telegram para os avisos da Forecast Workstation.\n"
          "1) No Telegram, fala com @BotFather, envia /newbot e escolhe um nome.\n"
          "2) Cola aqui o token que ele te der (não aparece ao escrever).\n")
    tok = getpass.getpass("Token do bot: ").strip()
    if not tok:
        print("Sem token: nada feito.")
        return
    me = await call("getMe", tok=tok)
    print(f"\nBot encontrado: @{me['username']} ({me['first_name']}).")
    set_token(tok)
    print("Token guardado no porta-chaves do sistema.\n")
    print(f"3) Agora abre o Telegram, procura @{me['username']} e envia-lhe /start "
          "(fico à espera até 5 minutos)…")
    offset, chat = 0, None
    for _ in range(100):
        ups = await call("getUpdates", {"offset": offset, "timeout": 3}, tok=tok, timeout=20)
        for u in ups:
            offset = u["update_id"] + 1
            m = u.get("message") or {}
            if m.get("chat", {}).get("type") == "private":
                chat = m["chat"]
        if chat:
            break
        await asyncio.sleep(1)
    if not chat:
        print("Não recebi nenhuma mensagem. Volta a correr este comando e envia /start ao bot.")
        return
    await call("getUpdates", {"offset": offset, "timeout": 0}, tok=tok)   # mark as read
    save_conf({"chat_id": chat["id"], "name": chat.get("first_name", ""), "bot": me["username"],
               "configured": dt.datetime.now().isoformat(timespec="seconds")})
    await send_text("✅ <b>Forecast Workstation</b>\nOs avisos de tempo extremo vão chegar a esta "
                    "conversa. Envia /estado para ver os eventos em curso.", silent=False)
    print(f"Pronto: mensagem de teste enviada para {chat.get('first_name', 'ti')}.")
