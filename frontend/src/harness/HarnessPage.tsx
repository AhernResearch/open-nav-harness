import { useEffect, useState } from "react";
import { api, type Row } from "../api";
import { ConversationComposer } from "./Composer";
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

const runBase = (key: string) => {
  const [run, episode] = key.split("/").map(encodeURIComponent);
  return `/api/runs/${run}/episodes/${episode}`;
};
/** The selected run episode's API base from the address bar. */
function initialSource(): string | null {
  const run = new URLSearchParams(location.search).get("run");
  return run ? runBase(run) : null;
}

export function HarnessPage() {
  const [theme, toggleTheme] = useTheme();
  const [episodes, setEpisodes] = useState<Row[]>([]);
  const [source, setSource] = useState<string | null>(initialSource);
  const [cursor, setCursor] = useState<number | null>(null);
  const conversation = useConversation();
  const [datasetMode, setDatasetMode] = useState(false);
  const navigationTrace = useTrace(source);
  const trace = source ? navigationTrace : conversation.trace;
  const refresh = () => api<Row[]>("/runs/episodes/recent").then(setEpisodes).catch(() => {});
  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, []);
  /** Select a run episode or clear the selection. */
  const open = (row: Row | null) => {
    conversation.open(null);
    setDatasetMode(false);
    setSource(row ? row.base : null);
    setCursor(null);
    history.replaceState(null, "", "/harness" + (row ? "?run=" + encodeURIComponent(row.id) : ""));
  };
  const openConversation = (id: string) => {
    setSource(null); setCursor(null); setDatasetMode(false); conversation.open(id);
  };
  /** A submitted kernel run opens on its first episode; it joins the sidebar once it starts. */
  const started = (run: string, episode: string) => {
    const key = `${run}/${episode}`;
    open({ id: key, base: runBase(key) });
    refresh();
  };
  const last = trace.entries.length - 1;
  const upto = cursor ?? last;
  const s = trace.status;
  const camera = source ? trace.camera(upto, source) : [];
  const running = s?.status === "running";
  const recentItems = [...trace.items.values()].reverse();
  const spatialImage = recentItems.find(i => i.name.endsWith("__view_map") && i.images?.length);
  const objectImage = recentItems.find(i => !i.name.endsWith("__view_map") && i.images?.length);
  const panelImages = source ? camera : objectImage ? toolImages(objectImage, "", "物体图片") : [];
  const cancel = async () => {
    if (s?.run_id && confirm("停止整个运行？排队中的 episode 会取消，正在执行的会中断并记录。")) await api(`/runs/${s.run_id}/cancel`, {});
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
          {episodes.map((a) => (
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
          <a className="ghost-link" href="/workbench">
            <Icon name="grid" /> Workbench
          </a>
        </div>
      </div>

      <main className="hx-main">
        {!source ? (
          datasetMode ? <RunComposer onStarted={started} /> : conversation.id ? <>
            <header className="run-head"><span className={"status-pill " + conversation.status}><i/>{statusText[conversation.status] ?? conversation.status}</span><span className="spacer"/><span className="muted">{conversation.model}</span></header>
            <Thread trace={trace} upto={last} live={true} select={() => {}} base=""/>
            <ConversationComposer conversation={conversation}/>
          </> : <ConversationComposer conversation={conversation}/>
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
                <button className="pill-button danger" onClick={cancel}>
                  <Icon name="stop" size={14} /> 停止
                </button>
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
