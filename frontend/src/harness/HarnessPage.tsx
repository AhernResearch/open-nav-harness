import { useEffect, useState } from "react";
import { api, type Row } from "../api";
import { Composer, usePendingJob } from "./Composer";
import { Icon } from "./icons";
import { MapView } from "./MapPanel";
import { PlanCard, Thread } from "./Thread";
import { useTrace } from "./useTrace";

const statusText: Record<string, string> = {
  running: "运行中",
  completed: "已结束",
  failed: "失败",
  interrupted: "已中断",
  blocked: "待处理",
  cancelled: "已取消",
  queued: "排队中",
  paused: "已暂停",
};

function useTheme(): [string, () => void] {
  const initial = localStorage.getItem("theme") ?? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const [theme, setTheme] = useState(initial);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);
  }, [theme]);
  return [theme, () => setTheme(theme === "dark" ? "light" : "dark")];
}

export function HarnessPage() {
  const [theme, toggleTheme] = useTheme();
  const [attempts, setAttempts] = useState<Row[]>([]);
  const [attempt, setAttempt] = useState<string | null>(new URLSearchParams(location.search).get("attempt"));
  const [job, setJob] = useState<string | null>(new URLSearchParams(location.search).get("job"));
  const [cursor, setCursor] = useState<number | null>(null);
  const trace = useTrace(attempt);
  const refresh = () => api<Row[]>("/harness/attempts").then(setAttempts).catch(() => {});
  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, []);
  const open = (id: string | null) => {
    setAttempt(id);
    setJob(null);
    setCursor(null);
    history.replaceState(null, "", id ? "/harness?attempt=" + encodeURIComponent(id) : "/harness");
  };
  const pending = usePendingJob(job, (id) => {
    open(id);
    refresh();
  });
  const enqueue = (id: string) => {
    setJob(id);
    history.replaceState(null, "", "/harness?job=" + encodeURIComponent(id));
  };
  const waitingText: Record<string, string> = {
    needs_reconciliation: "上次运行已中断，资源仍被锁定，需核验后释放。",
    services_unavailable: "等待环境、VLA 或执行器服务就绪。",
    resources_busy: "其他运行正在使用环境和执行器，释放后会自动开始。",
    queued: "已加入队列，等待环境和 VLA 资源…",
  };
  const pendingText = pending?.connection_error ? "暂时无法读取运行状态，正在重新连接…"
    : pending && !["queued", "running"].includes(pending.status) ? `运行${statusText[pending.status] ?? pending.status}。${pending.error ?? ""}`
    : pending?.status === "running" ? "已开始调度，正在打开运行记录…"
    : waitingText[pending?.waiting_reason ?? "queued"] ?? waitingText.queued;
  const last = trace.entries.length - 1;
  const upto = cursor ?? last;
  const s = trace.status;
  const frame = trace.latestMotion(upto)?.frames.at(-1);
  const running = s?.status === "running";
  const control = async (action: string) => {
    if (action === "cancel" && !confirm("停止会终止正在执行的 episode，中断后需要在 Workbench 中核验并释放资源。确定停止？")) return;
    await api(`/jobs/${s!.job_id}/action`, { action });
  };

  return (
    <div className="hx">
      <div className="hx-side" role="navigation" aria-label="会话">
        <div className="hx-brand">
          <span className="logo-dot">N</span>Nav Harness
        </div>
        <button className="new-run" onClick={() => open(null)}>
          <Icon name="plus" /> 新的运行
        </button>
        <div className="sessions">
          {attempts.map((a) => (
            <button key={a.id} className={"session " + (a.id === attempt ? "active" : "")} onClick={() => open(a.id)}>
              <span className={"dot " + a.status} />
              <span className="session-text">
                <span className="title">{a.meta?.instruction ?? a.name}</span>
                <span className="sub">
                  {String(a.meta?.scene_id ?? "").split("/").at(-1)?.replace(".glb", "") || `ep ${a.episode}`} · {statusText[a.status] ?? a.status}
                </span>
              </span>
            </button>
          ))}
        </div>
        <div className="side-foot">
          <button className="ghost" onClick={toggleTheme} aria-label="切换主题">
            <Icon name={theme === "dark" ? "sun" : "moon"} />
          </button>
          <a className="ghost-link" href={attempt ? `/workbench?attempt=${encodeURIComponent(attempt)}` : "/workbench"}>
            <Icon name="grid" /> Workbench
          </a>
        </div>
      </div>

      <main className="hx-main">
        {!attempt ? (
          job ? (
            <div className="hero">
              <p role="status" className="thinking">{pendingText}</p>
              <a className="ghost-link" href={pending?.blocked_attempt_id ? `/workbench?attempt=${encodeURIComponent(pending.blocked_attempt_id)}` : "/workbench"}>
                {pending?.blocked_attempt_id ? "查看中断运行" : "查看工作台"}
              </a>
            </div>
          ) : (
            <Composer onStarted={enqueue} />
          )
        ) : (
          <>
            <header className="run-head">
              <span className={"status-pill " + (s?.status ?? "")}>
                <i />
                {statusText[s?.status] ?? "连接中"}
              </span>
              <span className="spacer" />
              {cursor !== null && (
                <button className="pill-button" onClick={() => setCursor(null)}>
                  回到实时
                </button>
              )}
              {running && (
                <>
                  <button className="pill-button" onClick={() => control("pause")}>
                    <Icon name="pause" size={14} /> 完成当前条后暂停
                  </button>
                  <button className="pill-button danger" onClick={() => control("cancel")}>
                    <Icon name="stop" size={14} /> 停止
                  </button>
                </>
              )}
            </header>
            <div className="plan-dock">
              <PlanCard tasks={trace.plan(upto)} />
            </div>
            <Thread trace={trace} upto={upto} live={cursor === null} select={setCursor} attempt={attempt} />
            {last > 0 && (
              <div className="scrubber">
                <input
                  aria-label="回放位置"
                  type="range"
                  min={0}
                  max={last}
                  value={upto}
                  onChange={(e) => setCursor(Number(e.target.value) === last ? null : Number(e.target.value))}
                />
              </div>
            )}
          </>
        )}
      </main>

      {attempt && (
        <aside className="hx-panel" aria-label="环境">
          <section className="card">
            <h3>
              <Icon name="map" /> 地图
            </h3>
            <MapView trace={trace} upto={upto} attempt={attempt} />
          </section>
          <section className="card">
            <h3>
              <Icon name="eye" /> 机器人视角
            </h3>
            {frame ? (
              <div className="camera">
                {Object.entries(frame).map(([name, path]) => path && <img key={name} alt={`${name} 视角`} src={`/api/attempts/${attempt}/files/${path}`} />)}
              </div>
            ) : (
              <div className="map-empty">开始运动后显示</div>
            )}
          </section>
        </aside>
      )}
    </div>
  );
}
