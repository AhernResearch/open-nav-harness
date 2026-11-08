import { useEffect, useState } from "react";
import { api, type Row } from "./api";
import { QueuePage } from "./pages/QueuePage";
import { ServicesPage } from "./pages/ServicesPage";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { ReplayPage } from "./pages/ReplayPage";
import { ComparePage } from "./pages/ComparePage";
export function Workbench() {
  const [page, setPage] = useState("queue");
  const [services, setServices] = useState<Row[]>([]),
    [jobs, setJobs] = useState<Row[]>([]),
    [experiments, setExperiments] = useState<Row[]>([]),
    [templates, setTemplates] = useState<Row[]>([]);
  const [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [detail, setDetail] = useState<Row | null>(null);
  const perform = async (fn: () => Promise<unknown>) => {
    setError("");
    setNotice("");
    try {
      await fn();
    } catch (e) {
      setError(String(e));
    }
  };
  const refresh = async () => {
    const [s, j, e, t] = await Promise.all([
      api<Row[]>("/services"),
      api<Row[]>("/jobs"),
      api<Row[]>("/experiments"),
      api<Row[]>("/templates"),
    ]);
    setServices(s);
    setJobs(j);
    setExperiments(e);
    setTemplates(t);
  };
  useEffect(() => {
    refresh().catch((e) => setError(String(e)));
    const requested = new URLSearchParams(window.location.search).get(
      "attempt",
    );
    if (requested) openAttempt(requested).catch((e) => setError(String(e)));
    const source = new EventSource("/api/events");
    source.onmessage = () => refresh().catch(() => {});
    const timer = setInterval(() => refresh().catch(() => {}), 5000);
    return () => {
      source.close();
      clearInterval(timer);
    };
  }, []);
  const openAttempt = async (id: string) => {
    setDetail(await api("/attempts/" + id));
    setPage("replay");
    const url = new URL(window.location.href);
    url.searchParams.set("attempt", id);
    window.history.replaceState(null, "", url);
  };
  const props = {
    services,
    jobs,
    experiments,
    templates,
    perform,
    refresh,
    setNotice,
    setPage,
    openAttempt,
  };
  return (
    <div className="layout">
      <aside>
        <a className="logo" onClick={() => setPage("queue")}>
          <span className="brand small">N↗</span>Nav Harness
        </a>
        <p className="eyebrow">WORKSPACE / A100</p>
        <a className="view-switch" href="/harness">
          Harness 实时视图 →
        </a>
        <nav>
          {[
            ["queue", "01", "任务队列"],
            ["services", "02", "模型与服务"],
            ["experiments", "03", "实验配置"],
            ["replay", "04", "轨迹回放"],
            ["compare", "05", "结果比较"],
          ].map(([id, n, title]) => (
            <button
              key={id}
              className={page === id ? "active" : ""}
              onClick={() => setPage(id)}
            >
              <span>{n}</span>
              {title}
            </button>
          ))}
        </nav>
        <div className="aside-bottom">
          <span className="dot" /> Holo → A100<p>v0.1 · Local control plane</p>
        </div>
      </aside>
      <main>
        <header>
          <div>
            <p className="eyebrow">EXPERIMENT CONTROL</p>
            <h1>
              {
                {
                  queue: "任务队列",
                  services: "模型与服务",
                  experiments: "实验配置",
                  replay: "轨迹回放",
                  compare: "结果比较",
                }[page]
              }
            </h1>
          </div>
          <div className="header-note">可复现执行 · 可核验记录</div>
        </header>
        {error && (
          <div role="alert" className="error">
            {error}
            <button className="quiet" onClick={() => setError("")}>
              关闭
            </button>
          </div>
        )}
        {notice && <div className="notice">{notice}</div>}

        {page === "queue" && <QueuePage {...props} />}
        {page === "services" && <ServicesPage {...props} />}
        {page === "experiments" && <ExperimentsPage {...props} />}
        {page === "replay" && (
          <ReplayPage
            key={detail?.id}
            detail={detail}
            refresh={async () => {
              if (detail) setDetail(await api("/attempts/" + detail.id));
            }}
          />
        )}
        {page === "compare" && <ComparePage {...props} />}
        <footer>Nav Harness · 每次决策都有可回溯的证据</footer>
      </main>
    </div>
  );
}
