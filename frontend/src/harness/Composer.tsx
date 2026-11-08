import { useEffect, useState } from "react";
import { api, type Row } from "../api";
import { Icon } from "./icons";

type Pick = { index: number; episode_id: number; scene: string; instruction: string };

function Pill({ label, value, options, onChange }: { label: string; value: string; options: [string, string][]; onChange: (v: string) => void }) {
  return (
    <label className="pill">
      <span>{label}</span>
      <select aria-label={label} value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map(([v, text]) => (
          <option key={v} value={v}>
            {text}
          </option>
        ))}
      </select>
    </label>
  );
}

/** Empty state: choose a scene and an instruction, then run it as a one-episode experiment. */
export function Composer({ onStarted }: { onStarted: (job: string) => void }) {
  const [catalogs, setCatalogs] = useState<Row[]>([]);
  const [services, setServices] = useState<Row[]>([]);
  const [experiments, setExperiments] = useState<Row[]>([]);
  const [dataset, setDataset] = useState("");
  const [scene, setScene] = useState("");
  const [query, setQuery] = useState("");
  const [matches, setMatches] = useState<Pick[]>([]);
  const [pick, setPick] = useState<Pick | null>(null);
  const [chosen, setChosen] = useState<Record<string, string>>({});
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    Promise.all([api<Row[]>("/catalog"), api<Row[]>("/services"), api<Row[]>("/experiments")])
      .then(([c, s, e]) => {
        setCatalogs(c);
        setServices(s);
        setExperiments(e);
        if (c[0]) {
          setDataset(c[0].name);
          setScene(c[0].scenes[0]?.scene ?? "");
        }
      })
      .catch((e) => setError(String(e)));
  }, []);
  const ready = (kind: string) => services.filter((s) => s.spec.kind === kind && s.status === "ready");
  const pickService = (kind: string) => chosen[kind] ?? ready(kind)[0]?.id ?? "";
  const catalog = catalogs.find((c) => c.name === dataset);
  useEffect(() => {
    if (!dataset || !scene) return;
    const timer = setTimeout(() => {
      api<Pick[]>(`/catalog/${dataset}/episodes?scene=${encodeURIComponent(scene)}&q=${encodeURIComponent(query)}&limit=6`)
        .then(setMatches)
        .catch(() => setMatches([]));
    }, 150);
    return () => clearTimeout(timer);
  }, [dataset, scene, query]);
  const options = (kind: string): [string, string][] => ready(kind).map((s) => [s.id, s.spec.name]);
  const environment = pickService("environment");
  const vla = pickService("vla");
  const planner = pickService("planner");
  // Reuse the latest frozen configuration for the same environment and actor when available.
  const base = experiments.find((e) => e.spec.environment_service === environment && e.spec.vla_service === vla)?.spec.config;
  const backend = chosen.backend ?? base?.vla?.backend ?? "lightnav";

  const start = async () => {
    if (!pick || !catalog) return;
    setBusy(true);
    setError("");
    try {
      const config = base
        ? { ...base, dataset: { name: catalog.dataset, split: catalog.split }, vla: { ...base.vla, backend } }
        : {
            dataset: { name: catalog.dataset, split: catalog.split },
            harness: { mode: "agentic" },
            vla: { backend },
            planner: { max_turns: 100, episode_timeout_s: 1200 },
            environment: { max_steps: 500 },
          };
      const { id } = await api<Row>("/experiments", {
        name: `Harness · ${pick.scene} · ${pick.index}`,
        planner_service: planner || null,
        environment_service: environment,
        vla_service: vla || null,
        episodes: [pick.index],
        purpose: "development",
        config,
      });
      const job = await api<Row>("/jobs", { experiment_id: id });
      onStarted(job.id);
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  };

  return (
    <div className="hero">
      <h1>今天让机器人去哪里？</h1>
      <p className="muted">选择一个场景和一条导航指令，智能体会规划、执行并实时汇报。</p>
      <div className="composer">
        {pick && (
          <div className="chip-row">
            <span className="chip">
              <Icon name="sparkle" size={13} />
              {pick.scene} · #{pick.index}
              <button aria-label="清除指令" onClick={() => setPick(null)}>
                <Icon name="x" size={12} />
              </button>
            </span>
          </div>
        )}
        <textarea
          aria-label="指令"
          rows={pick ? 3 : 2}
          placeholder="搜索这个场景里的指令，例如 “stairs” 或 “kitchen”…"
          value={pick ? pick.instruction : query}
          readOnly={!!pick}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && pick) {
              e.preventDefault();
              start();
            }
          }}
        />
        <div className="composer-bar">
          <Pill label="数据集" value={dataset} options={catalogs.map((c) => [c.name, `${c.dataset} ${c.split}`])} onChange={setDataset} />
          <Pill
            label="场景"
            value={scene}
            options={(catalog?.scenes ?? []).map((s: Row) => [s.scene, `${s.scene} · ${s.count}`])}
            onChange={(v) => {
              setScene(v);
              setPick(null);
            }}
          />
          <Pill label="Planner" value={planner} options={options("planner")} onChange={(v) => setChosen({ ...chosen, planner: v })} />
          <Pill label="环境" value={environment} options={options("environment")} onChange={(v) => setChosen({ ...chosen, environment: v })} />
          <Pill label="VLA" value={vla} options={options("vla")} onChange={(v) => setChosen({ ...chosen, vla: v })} />
          <Pill label="动作模型" value={backend} options={[["lightnav", "LightNav"], ["navgpt", "NavGPT"]]} onChange={(v) => setChosen({ ...chosen, backend: v })} />
          <span className="spacer" />
          <button className="send" aria-label="开始运行" disabled={!pick || !environment || busy} onClick={start}>
            <Icon name="send" />
          </button>
        </div>
      </div>
      {error && <p className="composer-error">{error}</p>}
      {!pick && (
        <div className="suggestions">
          {matches.map((m) => (
            <button key={m.index} className="suggestion" onClick={() => setPick(m)}>
              {m.instruction}
            </button>
          ))}
          {!matches.length && dataset && <p className="muted">这个场景里没有匹配的指令。</p>}
          {!catalogs.length && <p className="muted">还没有可用的指令数据集。</p>}
        </div>
      )}
      <p className="hint muted">自定义指令将在下一个版本支持；目前起点与指令来自所选数据集条目。</p>
    </div>
  );
}

export function usePendingJob(job: string | null, onAttempt: (id: string) => void) {
  useEffect(() => {
    if (!job) return;
    const timer = setInterval(async () => {
      const rows = await api<Row[]>(`/jobs/${job}/attempts`).catch(() => []);
      if (rows[0]) onAttempt(rows[0].attempt_id);
    }, 1500);
    return () => clearInterval(timer);
  }, [job]);
}
