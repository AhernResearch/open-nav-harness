import { useEffect, useState } from "react";
import { api, type Row } from "./api";
import { Logo } from "./Logo";
import { ServicesPage } from "./pages/ServicesPage";

/** Operations console: registered services, their health, logs and owned start/stop. */
export function Workbench() {
  const [services, setServices] = useState<Row[]>([]),
    [templates, setTemplates] = useState<Row[]>([]);
  const [error, setError] = useState(""),
    [notice, setNotice] = useState("");
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
    const [s, t] = await Promise.all([api<Row[]>("/services"), api<Row[]>("/templates")]);
    setServices(s);
    setTemplates(t);
  };
  useEffect(() => {
    refresh().catch((e) => setError(String(e)));
    const source = new EventSource("/api/events");
    source.onmessage = () => refresh().catch(() => {});
    const timer = setInterval(() => refresh().catch(() => {}), 5000);
    return () => {
      source.close();
      clearInterval(timer);
    };
  }, []);
  return (
    <div className="layout">
      <aside>
        <a className="logo" href="/workbench">
          <Logo size={30} />
          open nav harness
        </a>
        <p className="eyebrow">WORKSPACE / A100</p>
        <a className="view-switch" href="/harness">
          Harness 实时视图 →
        </a>
        <nav>
          <button className="active">
            <span>01</span>模型与服务
          </button>
        </nav>
        <div className="aside-bottom">
          <span className="dot" /> Holo → A100<p>v0.1 · Local control plane</p>
        </div>
      </aside>
      <main>
        <header>
          <div>
            <p className="eyebrow">SERVICE CONTROL</p>
            <h1>模型与服务</h1>
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
        <ServicesPage services={services} templates={templates} perform={perform} refresh={refresh} setNotice={setNotice} />
        <footer>Open Nav Harness · 每次决策都有可回溯的证据</footer>
      </main>
    </div>
  );
}
