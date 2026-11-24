import { useEffect, useRef, useState } from "react";
import { api, type Row } from "../api";
import { Pill } from "./Composer";

const purposes: [string, string][] = [["evaluation", "评测"], ["collection", "采集（仅 train）"]];

/** A kernel run over chosen dataset episodes. Only planner and environment names leave the
 *  browser; the server builds the frozen spec from its own settings. */
export function RunComposer({ onStarted }: { onStarted: (run: string, episode: string) => void }) {
  const [options, setOptions] = useState<Row>({ models: [], navigation: [] });
  const [planner, setPlanner] = useState("");
  const [environment, setEnvironment] = useState("");
  const [catalog, setCatalog] = useState<Row | null>(null);
  const [scene, setScene] = useState("");
  const [episodes, setEpisodes] = useState<Row[]>([]);
  const [chosen, setChosen] = useState<string[]>([]);
  const [lanes, setLanes] = useState(1);
  const [purpose, setPurpose] = useState("evaluation");
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const submitting = useRef(false);
  const target = (options.navigation as Row[]).find((n) => n.name === environment);

  useEffect(() => {
    api<Row>("/conversations/options").then((o) => {
      setOptions(o);
      setPlanner(o.default_model ?? o.models[0] ?? "");
      setEnvironment(o.navigation[0]?.name ?? "");
    }).catch((e) => setError(String(e)));
  }, []);
  useEffect(() => {
    setCatalog(null); setScene(""); setEpisodes([]); setChosen([]);
    if (!target) return;
    api<Row[]>("/catalog").then((rows) => {
      const found = rows.find((row) => row.name === target.dataset);
      setCatalog(found ?? null);
      setScene(found?.scenes?.[0]?.scene ?? "");
    }).catch((e) => setError(String(e)));
  }, [target?.dataset]);
  useEffect(() => {
    if (!catalog || !scene) return;
    api<Row[]>(`/catalog/${catalog.name}/episodes?scene=${encodeURIComponent(scene)}&limit=200`)
      .then(setEpisodes).catch((e) => setError(String(e)));
  }, [catalog?.name, scene]);

  const toggle = (index: string) =>
    setChosen((current) => current.includes(index) ? current.filter((i) => i !== index) : [...current, index]);
  const submit = async () => {
    if (submitting.current || !chosen.length) return;
    submitting.current = true; setBusy(true); setError("");
    try {
      const { id } = await api<Row>("/runs/compose", {
        name: name.trim() || `${environment} ${new Date().toLocaleString()}`,
        purpose, planner, environment, episodes: chosen, lanes,
      });
      onStarted(id, chosen[0]);
    } catch (e) {
      setError(String(e));
    } finally {
      submitting.current = false; setBusy(false);
    }
  };

  return (
    <div className="hero run-composer">
      <h1>新建运行</h1>
      <p className="muted">选择 planner、环境和 episode；端点与预算来自服务端配置。</p>
      <div className="composer">
        <input aria-label="运行名称" className="run-name" placeholder="运行名称（可选）" value={name}
          onChange={(e) => setName(e.target.value)} />
        <div className="composer-bar">
          <Pill label="planner" value={planner} options={(options.models as string[]).map((m) => [m, m])} onChange={setPlanner} />
          <Pill label="环境" value={environment} options={(options.navigation as Row[]).map((n) => [n.name, `${n.name} · ${n.dataset}`])} onChange={setEnvironment} />
          {catalog && <Pill label="场景" value={scene}
            options={catalog.scenes.map((s: Row) => [s.scene, `${String(s.scene).split("/").at(-1)?.replace(".glb", "")} · ${s.count}`] as [string, string])}
            onChange={setScene} />}
          <Pill label="用途" value={purpose} options={purposes} onChange={setPurpose} />
          <label className="pill"><span>lanes</span>
            <input aria-label="lanes" type="number" min={1} max={8} value={lanes}
              onChange={(e) => setLanes(Math.min(8, Math.max(1, Number(e.target.value) || 1)))} />
          </label>
        </div>
      </div>
      <div className="episode-list" role="group" aria-label="episode">
        {episodes.map((e) => {
          const index = String(e.index);
          return (
            <label key={index} className={"episode-row " + (chosen.includes(index) ? "chosen" : "")}>
              <input type="checkbox" checked={chosen.includes(index)} onChange={() => toggle(index)} />
              <span className="episode-index">#{index}</span>
              <span className="episode-text">{e.instruction}</span>
            </label>
          );
        })}
        {!episodes.length && <p className="muted">{target ? "这个场景没有可用 episode。" : "服务端还没有配置可运行的环境。"}</p>}
      </div>
      <div className="composer-bar">
        <span className="muted">已选 {chosen.length} 条{chosen.length > 0 && `：${chosen.map((i) => "#" + i).join(" ")}`}</span>
        <span className="spacer" />
        <button className="new-run" disabled={!chosen.length || !planner || !environment || busy} onClick={() => void submit()}>
          开始运行
        </button>
      </div>
      {error && <p className="composer-error" role="alert">{error}</p>}
    </div>
  );
}
