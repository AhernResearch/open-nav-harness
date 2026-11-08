"""Service operations use trusted templates; external services are read-only."""

import asyncio
import shlex
import json
import time
from pathlib import Path
import httpx
from sqlalchemy import text
from nav_harness.core.config import PlatformSettings


class ServiceManager:
    def __init__(self, db, settings: PlatformSettings):
        self.db, self.settings = db, settings
        self.lock = asyncio.Lock()

    async def ssh(self, host, argv):
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        if self.settings.ssh_config:
            command += ["-F", self.settings.ssh_config]
        command += [host, shlex.join(argv)]
        proc = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError("SSH timeout; remote operation must be reconciled before retry")
        if proc.returncode:
            raise RuntimeError(stderr.decode()[-2000:])
        return stdout.decode()

    async def health(self, sid):
        spec = self.db.service(sid)
        if spec.kind == "planner" and spec.planner.runtime != "openai_compatible":
            return {"ok": None, "status": "SDK login and model access are checked during preflight"}
        suffix = {"planner": "/models", "environment": "/health", "vla": "/info", "execution": "/info"}[
            spec.kind
        ]
        try:
            async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                r = await client.get(spec.url.rstrip("/") + suffix)
                r.raise_for_status()
            return {"ok": True, "details": r.json()}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def owned(self, sid):
        spec = self.db.service(sid)
        if not spec.managed:
            raise ValueError("External services cannot be started or stopped by the platform")
        template = self.settings.service_templates.get(spec.template)
        if not template:
            raise ValueError("Unknown administrator-defined service template")
        return spec, template, "navh-" + sid

    def set_status(self, sid, status, **details):
        with self.db.transaction() as c:
            c.execute(text("UPDATE services SET status=:s WHERE id=:id"), {"s": status, "id": sid})
            self.db.emit(c, "service_" + status, sid, details)

    async def unit_state(self, sid):
        _, template, unit = self.owned(sid)
        raw = await self.ssh(
            template.host,
            [
                "systemctl",
                "--user",
                "show",
                unit,
                "-p",
                "Environment",
                "-p",
                "ActiveState",
                "-p",
                "LoadState",
            ],
        )
        state = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        if state.get("LoadState") != "not-found" and "NAV_HARNESS_OWNER=" + sid not in shlex.split(
            state.get("Environment", "")
        ):
            raise ValueError("Remote unit ownership could not be verified")
        return state

    async def reconcile_attempt(self, aid):
        # Validate motion services are stopped before touching shared actor state.
        attempt = self.db.release_reconciled(aid, check_only=True)
        if attempt["binding"]:
            spec = json.loads(attempt["spec"])
            vla = self.db.service(spec["vla_service"])
            status = self.db.rows("SELECT status FROM services WHERE id=:id", id=spec["vla_service"])[0][
                "status"
            ]
            if not (vla.managed and status == "stopped"):
                receipts = list(Path(attempt["output"]).rglob("execution_session.json"))
                if len(receipts) > 1:
                    raise ValueError("Ambiguous execution session receipts")
                if receipts:
                    ident = json.loads(receipts[0].read_text())["session_id"]
                    if len(ident) != 32 or any(c not in "0123456789abcdef" for c in ident):
                        raise ValueError("Invalid session receipt")
                    async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
                        url = vla.url.rstrip("/") + "/sessions/" + ident
                        response = await client.get(url)
                        if response.status_code != 404:
                            response.raise_for_status()
                            (await client.post(url + "/cancel")).raise_for_status()
                            # A 409 means CUDA work is still in flight; retain all leases.
                            response = await client.post(url + "/close")
                            if response.status_code == 409:
                                raise ValueError(
                                    "Inference is still running; retain the lease until it drains"
                                )
                            response.raise_for_status()
        self.db.release_reconciled(aid, inference_reconciled=True)

    async def operate(self, sid, action):
        if action not in ("start", "stop"):
            raise ValueError("Unknown service action")
        async with self.lock:
            spec, template, unit = self.owned(sid)
            # Fence the scheduler in the same database transaction as the lease check.
            with self.db.transaction() as c:
                leases = (
                    c.execute(
                        text(
                            "SELECT a.status FROM leases l JOIN attempts a ON a.id=l.attempt_id WHERE l.resource=:r OR l.resource LIKE :p"
                        ),
                        {"r": sid, "p": sid + ":%"},
                    )
                    .scalars()
                    .all()
                )
                if leases and (action != "stop" or any(x != "blocked" for x in leases)):
                    raise ValueError("Service is leased by a running attempt")
                current = c.execute(
                    text("SELECT status FROM services WHERE id=:id"), {"id": sid}
                ).scalar_one()
                if current == "stopping" or (
                    action == "start" and current in ("starting", "ready", "unknown")
                ):
                    raise ValueError("Service operation already active or state requires stopping first")
                c.execute(
                    text("UPDATE services SET status=:s WHERE id=:id"),
                    {"s": "starting" if action == "start" else "stopping", "id": sid},
                )
                self.db.emit(c, "service_" + action + "_requested", sid, {})
            try:
                state = await self.unit_state(sid)
                if action == "start":
                    if (
                        state.get("ActiveState") not in ("inactive", "failed", None)
                        or (await self.health(sid))["ok"]
                    ):
                        raise ValueError("Service already running or endpoint occupied; refusing to adopt it")
                    gpu_csv = await self.ssh(
                        template.host,
                        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                    )
                    free = {int(row.split(",")[0]): int(row.split(",")[1]) for row in gpu_csv.splitlines()}
                    pending: dict[int, int] = {}
                    for row in self.db.rows(
                        "SELECT id,spec FROM services WHERE status='starting' AND id!=:id", id=sid
                    ):
                        other = json.loads(row["spec"])
                        other_template = self.settings.service_templates.get(other["template"])
                        if other_template and other_template.host == template.host:
                            for gpu in other["gpu_ids"]:
                                pending[gpu] = pending.get(gpu, 0) + other["reserve_mib"]
                    for gpu in spec.gpu_ids:
                        if free.get(gpu, 0) - pending.get(gpu, 0) < spec.reserve_mib + 1024:
                            raise ValueError(
                                "Insufficient unreserved GPU memory; no services will be evicted"
                            )
                    argv = [
                        "systemd-run",
                        "--user",
                        "--collect",
                        "--unit=" + unit,
                        "--property=KillMode=control-group",
                        "--property=TimeoutStopSec=20",
                        "--working-directory=" + template.working_directory,
                    ]
                    env = {
                        **template.environment,
                        "CUDA_VISIBLE_DEVICES": ",".join(map(str, spec.gpu_ids)),
                        "NAV_HARNESS_OWNER": sid,
                    }
                    argv += ["--setenv=" + k + "=" + v for k, v in env.items()]
                    argv += ["--", *template.argv]
                    await self.ssh(template.host, argv)
                    self.set_status(sid, "starting")
                else:
                    if state.get("LoadState") != "not-found":
                        await self.ssh(template.host, ["systemctl", "--user", "stop", unit])
                        state = await self.unit_state(sid)
                    if state.get("ActiveState") not in ("inactive", "failed", None):
                        raise RuntimeError("Remote unit has not stopped")
                    if (await self.health(sid))["ok"]:
                        raise RuntimeError(
                            "Endpoint still answers after owned unit stopped; inspect endpoint ownership"
                        )
                    self.set_status(sid, "stopped")
            except Exception as exc:
                self.set_status(sid, "unknown", error=str(exc))
                raise
            return {"status": "starting" if action == "start" else "stopped"}

    async def reconcile_services(self):
        """Reconcile startup across API restarts; never restart a model implicitly."""
        for row in self.db.rows("SELECT id,status FROM services WHERE status IN ('starting','stopping')"):
            sid = row["id"]
            async with self.lock:
                # A request may have finished while this monitor waited for the lock.
                current = self.db.rows("SELECT status FROM services WHERE id=:id", id=sid)[0]["status"]
                if current not in ("starting", "stopping"):
                    continue
                try:
                    _, template, _ = self.owned(sid)
                    state = await self.unit_state(sid)
                    healthy = (await self.health(sid))["ok"]
                    if current == "stopping":
                        if state.get("ActiveState") in ("inactive", "failed", None) and not healthy:
                            self.set_status(sid, "stopped")
                        else:
                            self.set_status(sid, "unknown", error="Interrupted stop; verify before retry")
                    elif state.get("ActiveState") == "active" and healthy:
                        self.set_status(sid, "ready")
                    else:
                        started = self.db.rows(
                            "SELECT time FROM events WHERE entity=:id AND kind='service_start_requested' ORDER BY seq DESC LIMIT 1",
                            id=sid,
                        )
                        if (
                            not started
                            or time.time() - started[0]["time"] > template.startup_timeout_s
                            or (
                                time.time() - started[0]["time"] > 10
                                and state.get("ActiveState") in ("failed", "inactive", None)
                            )
                        ):
                            self.set_status(
                                sid, "unknown", error="Startup deadline exceeded; inspect service logs"
                            )
                except (ValueError, RuntimeError) as exc:
                    self.set_status(sid, "unknown", error=str(exc))

    async def logs(self, sid):
        _, template, unit = self.owned(sid)
        return await self.ssh(template.host, ["journalctl", "--user", "-u", unit, "-n", "100", "--no-pager"])
