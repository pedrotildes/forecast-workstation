"""Alert service of the Forecast Workstation.

    python -m app.alerts configurar      # Telegram bot (token in the keychain) — once
    python -m app.alerts servico         # continuous service (what the login agent runs)
    python -m app.alerts agora [--sem-envio]   # one cycle now (and print what would be sent)
    python -m app.alerts estado          # events in progress
    python -m app.alerts verificar       # verify the past days now (prints, sends nothing)
    python -m app.alerts teste           # test message to the phone
    python -m app.alerts instalar | remover    # login agent (macOS launchd)

Every 5 minutes: is there a new 00/12Z ECMWF ENS run? -> analyse it, update the events and
send what changed; new official IPMA orange/red warnings -> relay them. Every morning the
previous day is verified against the IPMA warnings and the observations (verify.py), and a
verification message follows each event that ended. A second task answers /estado on Telegram.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import shutil
import sys
import time
from pathlib import Path

from ..config import ALERTS_DIR
from . import ens, events, figures, ipma, messages, telegram, verify
from .detect import RUNS_DIR, analyse

H = dt.timedelta(hours=1)
PID_FILE = ALERTS_DIR / "service.pid"
LOG_FILE = ALERTS_DIR / "service.log"
OFFSET_FILE = ALERTS_DIR / "telegram_offset.txt"
LABEL = "local.forecast-workstation.alerts"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
CYCLE_S = 300


def log(msg: str) -> None:
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    try:
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > 3_000_000:
            LOG_FILE.replace(LOG_FILE.with_suffix(".log.1"))
        with open(LOG_FILE, "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def load_signals(run: dt.datetime) -> dict | None:
    p = RUNS_DIR / run.strftime("%Y%m%d%H") / "signals.json"
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------- notifications
async def notify(st: dict, note: dict, send: bool) -> None:
    ev = events.find(st, note["event"])
    if ev is None:
        return
    kind = note["kind"]
    text = messages.event_text(ev, kind, st["ipma"].get("active", []), note.get("why"))
    figs = []
    if kind in ("alerta", "atualizacao"):
        figs = await figures.render_all(ev, log)
    urgent = ev.get("level") == "vermelho" and kind in ("alerta", "atualizacao")
    if not send:
        log(f"[sem envio] {kind} {ev['id']} ({len(figs)} figuras)\n{text}")
        return
    if not telegram.configured():
        log(f"{kind} {ev['id']}: Telegram por configurar — fica para quando estiver")
        return
    try:
        await telegram.send_text(text, urgent=urgent, silent=(kind == "fim") or None)
        if figs:
            await telegram.send_photos(figs, silent=True)
    except telegram.TelegramError as e:
        log(f"{kind} {ev['id']}: envio falhou ({e})")
        return
    events.mark_notified(ev, kind, text, dt.datetime.now(dt.timezone.utc).replace(tzinfo=None))
    log(f"enviado: {kind} {ev['id']} ({len(figs)} figuras)")


async def ens_step(st: dict, cfg: dict, send: bool) -> None:
    run = await ens.latest_complete_run()
    if run is None or run.isoformat() in st["runs"]:
        return
    prev = run - 12 * H
    prev_an = load_signals(prev)
    if prev_an is None:
        # first start (or a gap): the previous run is needed for the consistency rule
        log(f"run anterior {prev:%Y-%m-%d %HZ} (para a consistência)")
        try:
            prev_an = await analyse(prev, log)
            events.update(st, prev_an, load_signals(prev - 12 * H), cfg)   # history only
        except Exception as e:  # noqa: BLE001
            log(f"  run anterior indisponível: {e}")
            prev_an = None
    log(f"run ENS {run:%Y-%m-%d %HZ}: análise")
    t = time.time()
    an = await analyse(run, log)
    notes = events.update(st, an, prev_an, cfg)
    events.save_state(st)
    log(f"run ENS {run:%Y-%m-%d %HZ}: {len(notes)} notificação(ões) [{time.time() - t:.0f} s]")
    for n in notes:
        await notify(st, n, send)
    events.save_state(st)


async def ipma_step(st: dict, cfg: dict, send: bool) -> None:
    ws = await ipma.fetch()
    st["ipma"]["active"] = ws
    try:
        verify.archive_warnings(ws)               # every level, for the verification
    except OSError as e:
        log(f"verificação: arquivo dos avisos falhou ({e})")
    st["ipma"]["checked"] = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    for grp in ipma.new_groups(st, ws, cfg.get("ipma_levels", ["orange", "red"])):
        text = messages.official_text(grp)
        if not send:
            log(f"[sem envio] IPMA\n{text}")
            continue
        if not telegram.configured():
            continue           # relayed only while the bot is set up (no backlog)
        try:
            await telegram.send_text(text, urgent=grp[0]["level"] == "red")
        except telegram.TelegramError as e:
            log(f"IPMA: envio falhou ({e})")
            continue
        ipma.mark_seen(st, grp)
        log(f"enviado: IPMA {grp[0]['level']} {grp[0]['type']} ({len(grp)} áreas)")
    if send and not telegram.configured():
        for grp in ipma.new_groups(st, ws, cfg.get("ipma_levels", ["orange", "red"])):
            ipma.mark_seen(st, grp)


async def verify_step(st: dict, send: bool) -> None:
    """Observations archive, the days due for verification, and their messages (silent)."""
    texts = await verify.step(st, log=log)
    events.save_state(st)
    for txt in texts:
        if not send or not telegram.configured():
            log(f"[sem envio] verificação\n{txt}")
            continue
        try:
            await telegram.send_text(txt, silent=True)
            log("enviado: verificação")
        except telegram.TelegramError as e:
            log(f"verificação: envio falhou ({e})")


async def catch_up(st: dict, send: bool) -> None:
    """Confirmed events never sent (bot configured later, or a failed send)."""
    if not send or not telegram.configured():
        return
    for ev in st["events"]:
        if ev["status"] == "confirmado" and not ev.get("notified"):
            await notify(st, {"event": ev["id"], "kind": "alerta", "why": []}, send)


def housekeeping() -> None:
    from ..models.base import cleanup_cache
    cleanup_cache(3)
    ens.cleanup(6)
    from .rain1h import cleanup as rain1h_cleanup
    rain1h_cleanup(4)
    from .waves import WAVE_CACHE
    for base, keep in ((RUNS_DIR, 40), (WAVE_CACHE / "ifs", 6), (WAVE_CACHE / "gfs", 8)):
        if base.exists():
            for d in sorted(p for p in base.iterdir() if p.is_dir())[:-keep]:
                shutil.rmtree(d, ignore_errors=True)
    if figures.FIG_DIR.exists():
        cutoff = time.time() - 60 * 86400
        for d in figures.FIG_DIR.iterdir():
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)


async def cycle(send: bool = True) -> None:
    st = events.load_state()
    cfg = events.load_config()
    # IPMA first: the alert texts say which official warnings are already in force
    for name, step in (("IPMA", lambda: ipma_step(st, cfg, send)), ("ENS", lambda: ens_step(st, cfg, send))):
        try:
            await step()
        except Exception as e:  # noqa: BLE001 — keep the service alive
            log(f"{name}: erro {e.__class__.__name__}: {e}")
        events.save_state(st)
    for n in events.tick(st):
        await notify(st, n, send)
    await catch_up(st, send)
    try:
        await verify_step(st, send)
    except Exception as e:  # noqa: BLE001
        log(f"verificação: erro {e.__class__.__name__}: {e}")
    st["heartbeat"] = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    events.save_state(st)


# ---------------------------------------------------------------- Telegram commands
HELP = ("<b>Avisos da Forecast Workstation</b>\n"
        "/estado — eventos confirmados e em vigilância, avisos do IPMA em vigor\n"
        "/ajuda — esta mensagem\n\n"
        "Os alertas chegam sozinhos quando um evento laranja ou vermelho fica confirmado "
        "(ensemble do ECMWF consistente em duas runs e apoiado pelos modelos determinísticos).")


async def listen() -> None:
    try:
        offset = int(OFFSET_FILE.read_text())
    except (OSError, ValueError):
        offset = 0
    while True:
        if not telegram.configured():
            await asyncio.sleep(60)
            continue
        try:
            ups = await telegram.call("getUpdates", {"offset": offset, "timeout": 50,
                                                     "allowed_updates": ["message"]}, timeout=70)
        except telegram.TelegramError as e:
            log(f"telegram: {e}")
            await asyncio.sleep(30)
            continue
        chat = telegram.conf().get("chat_id")
        for u in ups:
            offset = u["update_id"] + 1
            m = u.get("message") or {}
            if m.get("chat", {}).get("id") != chat:
                continue                     # only the configured chat is answered
            cmd = (m.get("text") or "").strip().split()[0].split("@")[0].lower() if m.get("text") else ""
            try:
                if cmd in ("/estado", "estado"):
                    await telegram.send_text(messages.status_text(events.load_state()), silent=False)
                elif cmd in ("/ajuda", "/start", "/help", "ajuda"):
                    await telegram.send_text(HELP, silent=False)
            except telegram.TelegramError as e:
                log(f"telegram: {e}")
        OFFSET_FILE.write_text(str(offset))


# ---------------------------------------------------------------- service
def other_running() -> bool:
    if os.name != "posix":          # the service is macOS-only; os.kill(pid, 0) kills on Windows
        return False
    try:
        pid = int(PID_FILE.read_text())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    if pid == os.getpid():
        return False
    import subprocess
    try:
        cmd = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True,
                             timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return True
    return "app.alerts" in cmd


async def service() -> None:
    if other_running():
        print("serviço de avisos já em execução — nada a fazer", flush=True)
        return
    PID_FILE.write_text(str(os.getpid()))
    log("serviço de avisos iniciado")
    asyncio.create_task(listen())
    last_hk = 0.0
    while True:
        await cycle(send=True)
        if time.time() - last_hk > 6 * 3600:
            try:
                await asyncio.to_thread(housekeeping)
            except Exception as e:  # noqa: BLE001
                log(f"limpeza: {e}")
            last_hk = time.time()
        await asyncio.sleep(CYCLE_S)


def install_agent() -> None:
    import subprocess
    py = Path(sys.executable)
    backend = Path(__file__).resolve().parents[2]
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<!-- Forecast Workstation: alert service (python -m app.alerts servico), always running.
	     Remove: python -m app.alerts remover (in backend/) -->
	<key>Label</key>
	<string>{LABEL}</string>
	<key>ProgramArguments</key>
	<array>
		<string>{py}</string>
		<string>-m</string>
		<string>app.alerts</string>
		<string>servico</string>
	</array>
	<key>WorkingDirectory</key>
	<string>{backend}</string>
	<key>EnvironmentVariables</key>
	<dict>
		<key>PATH</key>
		<string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
	</dict>
	<key>RunAtLoad</key>
	<true/>
	<key>KeepAlive</key>
	<true/>
	<key>ThrottleInterval</key>
	<integer>120</integer>
	<key>ProcessType</key>
	<string>Standard</string>
	<key>StandardOutPath</key>
	<string>/dev/null</string>
	<key>StandardErrorPath</key>
	<string>{ALERTS_DIR / "service.err.log"}</string>
</dict>
</plist>
"""
    PLIST.write_text(plist)
    uid = os.getuid()
    subprocess.run(["launchctl", "bootout", f"gui/{uid}/{LABEL}"], capture_output=True)
    r = subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(PLIST)], capture_output=True, text=True)
    print("agente instalado e iniciado" if r.returncode == 0 else f"launchctl: {r.stderr.strip()}")


def remove_agent() -> None:
    import subprocess
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"], capture_output=True)
    PLIST.unlink(missing_ok=True)
    print("agente removido")


def main(argv: list[str]) -> None:
    cmd = argv[0] if argv else "servico"
    if cmd == "configurar":
        asyncio.run(telegram.configure())
    elif cmd == "servico":
        asyncio.run(service())
    elif cmd == "agora":
        asyncio.run(cycle(send="--sem-envio" not in argv))
    elif cmd == "verificar":
        import re

        async def run_verify():
            st = events.load_state()
            for txt in await verify.step(st, log=print):
                print(re.sub(r"<[^>]+>", "", txt), "\n")
            events.save_state(st)
            for typ, t in verify.summary()["types"].items():
                n = t["acertos"] + t["alarmes_falsos"] + t["falhas"]
                if n:
                    print(f"{typ}: {t['acertos']} acertos, {t['alarmes_falsos']} alarmes falsos, "
                          f"{t['falhas']} falhas (POD {t['pod']}, FAR {t['far']})")
        asyncio.run(run_verify())
    elif cmd == "estado":
        import re
        print(re.sub(r"<[^>]+>", "", messages.status_text(events.load_state())))
    elif cmd == "teste":
        asyncio.run(telegram.send_text("🧪 Teste dos avisos da Forecast Workstation.", silent=False))
        print("enviado")
    elif cmd == "instalar":
        install_agent()
    elif cmd == "remover":
        remove_agent()
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
