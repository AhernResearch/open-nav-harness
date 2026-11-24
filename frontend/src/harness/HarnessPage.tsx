import { useEffect, useState } from "react";
import { api, type Row } from "../api";
import { Composer, ConversationComposer, usePendingJob } from "./Composer";
import { Logo } from "../Logo";
import { Icon } from "./icons";
import { MapView } from "./MapPanel";
import { PlanCard, Thread } from "./Thread";
import { useConversation } from "./useConversation";
import { toolImages } from "./trace";
import { useTrace } from "./useTrace";
import { CameraGallery } from "./CameraGallery";
import { RunComposer } from "./RunComposer";

const statusText: Record<string, string> = {
  idle: "就绪",
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

const attemptBase = (id: string) => `/api/attempts/${encodeURIComponent(id)}`;
const runBase = (key: string) => {
  const [run, episode] = key.split("/").map(encodeURIComponent);
  return `/api/runs/${run}/episodes/${episode}`;
};
/** The selected episode's API base from the address bar: legacy attempts or kernel run episodes. */
function initialSource(): string | null {
  const q = new URLSearchParams(location.search);
  const attempt = q.get("attempt"), run = q.get("run");
  return attempt ? attemptBase(attempt) : run ? runBase(run) : null;
}

export function HarnessPage() {
  const [theme, toggleTheme] = useTheme();
  const [attempts, setAttempts] = useState<Row[]>([]);
  const [source, setSource] = useState<string | null>(initialSource);
  const attempt = source?.startsWith("/api/attempts/") ? decodeURIComponent(source.split("/").at(-1)!) : null;
  const [job, setJob] = useState<string | null>(new URLSearchParams(location.search).get("job"));
  const [cursor, setCursor] = useState<number | null>(null);
  const conversation = useConversation();
  const [datasetMode, setDatasetMode] = useState(false);
  const navigationTrace = useTrace(source);
  const trace = source ? navigationTrace : conversation.trace;
  const refresh = () => api<Row[]>("/harness/attempts").then(setAttempts).catch(() => {});
  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, []);
  /** Select an episode row (`run_id` marks a kernel run episode) or clear the selection. */
  const open = (row: Row | null) => {
    conversation.open(null);
    setDatasetMode(false);
    setSource(row ? row.base : null);
    setJob(null);
    setCursor(null);
    const query = !row ? "" : row.run_id ? "?run=" + encodeURIComponent(row.id) : "?attempt=" + encodeURIComponent(row.id);
    history.replaceState(null, "", "/harness" + query);
  };
  const openConversation = (id: string) => {
    setSource(null); setJob(null); setCursor(null); setDatasetMode(false); conversation.open(id);
  };
  const pending = usePendingJob(job, (id) => {
    open({ id, base: attemptBase(id) });
    refresh();
  });
  /** A submitted kernel run opens on its first episode; it joins the sidebar once it starts. */
  const started = (run: string, episode: string) => {
    const key = `${run}/${episode}`;
    open({ id: key, base: runBase(key), run_id: run });
    refresh();
  };
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
  const camera = source ? trace.camera(upto, source) : [];
  const running = s?.status === "running";
  const recentItems = [...trace.items.values()].reverse();
  const spatialImage = recentItems.find(i => i.name.endsWith("__view_map") && i.images?.length);
  const objectImage = recentItems.find(i => !i.name.endsWith("__view_map") && i.images?.length);
  const panelImages = source ? camera : objectImage ? toolImages(objectImage, "", "物体图片") : [];
  const control = async (action: string) => {
    if (s?.run_id) {
      if (confirm("停止整个运行？排队中的 episode 会取消，正在执行的会中断并记录。")) await api(`/runs/${s.run_id}/cancel`, {});
      return;
    }
    if (action === "cancel" && !confirm("停止会终止正在执行的 episode，中断后需要在 Workbench 中核验并释放资源。确定停止？")) return;
    await api(`/jobs/${s!.job_id}/action`, { action });
  };

  return (
    <div className="hx">
      <div className="hx-side" role="navigation" aria-label="会话">
        <div className="hx-brand">
          <Logo />
          open nav harness
        </div>
        <button className="new-run" onClick={() => open(null)}>
          <Icon name="plus" /> 新的运行
        </button>
        <button className="ghost-link" onClick={() => { open(null); setDatasetMode(true); }}><Icon name="list" size={14}/> 从数据集运行</button>
        <div className="sessions">
          {conversation.sessions.map(a => <button key={"conversation-" + a.id} className={"session " + (a.id === conversation.id ? "active" : "")} onClick={() => openConversation(a.id)}>
            <span className={"dot " + a.status}/><span className="session-text"><span className="title">{a.title || "新的会话"}</span><span className="sub">{a.spec.model} · {statusText[a.status] ?? a.status}</span></span>
          </button>)}
          {attempts.map((a) => (
            <button key={a.base} className={"session " + (a.base === source ? "active" : "")} onClick={() => open(a)}>
              <span className={"dot " + a.status} />
              <span className="session-text">
                <span className="title">{a.meta?.instruction ?? a.name}</span>
                <span className="sub">
                  {String(a.meta?.scene_id ?? "").split("/").at(-1)?.replace(".glb", "") || `ep ${a.episode}`} · {statusText[a.status] ?? a.status}
                  {a.created && <> · {new Date(a.created * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</>}
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
        {!source ? (
          job ? (
            <div className="hero">
              <p role="status" className="thinking">{pendingText}</p>
              <a className="ghost-link" href={pending?.blocked_attempt_id ? `/workbench?attempt=${encodeURIComponent(pending.blocked_attempt_id)}` : "/workbench"}>
                {pending?.blocked_attempt_id ? "查看中断运行" : "查看工作台"}
              </a>
            </div>
          ) : (
            datasetMode ? <RunComposer onStarted={started} /> : (conversation.ready && !conversation.options.models.length) ? <Composer onStarted={enqueue} /> : conversation.id ? <>
              <header className="run-head"><span className={"status-pill " + conversation.status}><i/>{statusText[conversation.status] ?? conversation.status}</span><span className="spacer"/><span className="muted">{conversation.model}</span></header>
              <Thread trace={trace} upto={last} live={true} select={() => {}} base=""/>
              <ConversationComposer conversation={conversation}/>
            </> : <ConversationComposer conversation={conversation}/>
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
                  {s?.job_id && <button className="pill-button" onClick={() => control("pause")}>
                    <Icon name="pause" size={14} /> 完成当前条后暂停
                  </button>}
                  <button className="pill-button danger" onClick={() => control("cancel")}>
                    <Icon name="stop" size={14} /> 停止
                  </button>
                </>
              )}
            </header>
            <div className="plan-dock">
              <PlanCard tasks={trace.plan(upto)} />
            </div>
            <Thread trace={trace} upto={upto} live={cursor === null} select={setCursor} base={source} />
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

      {(source || conversation.id) && (
        <aside className="hx-panel" aria-label="环境">
          <section className="card">
            <h3>
              <Icon name="map" /> 地图
            </h3>
            {source ? <MapView trace={trace} upto={upto} base={source} /> : spatialImage ? <CameraGallery images={toolImages(spatialImage, "", "空间图")}/> : <div className="map-empty">查询空间图后显示</div>}
          </section>
          <section className="card">
            <h3>
              <Icon name="eye" /> {source ? "机器人视角" : "物体图片"}
            </h3>
            {panelImages.length ? (
              <CameraGallery images={panelImages} />
            ) : (
              <div className="map-empty">观察或运动后显示</div>
            )}
          </section>
        </aside>
      )}
    </div>
  );
}
