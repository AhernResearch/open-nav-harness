import { useEffect, useRef, useState } from "react";
import { api, type Row } from "../api";
import { Icon } from "./icons";
import { ModelPicker } from "./ModelPicker";
import { agenticPlanner } from "../run-profile";

type Pick = { index: number; episode_id: number; scene: string; instruction: string };

function bindings(services: Row[], vla: string) {
  const inference = services.find((s) => s.id === vla);
  return services.filter((s) => s.spec.kind === "execution" && s.spec.metadata?.inference_url === inference?.spec.url)
    .flatMap((worker) => {
      const environment = services.find((s) => s.spec.kind === "environment" && s.spec.url === worker.spec.metadata?.environment_url);
      return environment ? [{ worker, environment }] : [];
    });
}

function Pill({ label, value, options, onChange }: { label: string; value: string; options: [string, string, boolean?][]; onChange: (v: string) => void }) {
  return (
    <label className="pill">
      <span>{label}</span>
      <select aria-label={label} value={value} onChange={(e) => onChange(e.target.value)}>
        {!options.length && <option value="">暂无可用选项</option>}
        {options.map(([v, text, disabled]) => (
          <option key={v} value={v} disabled={disabled}>
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
  const submitting = useRef(false);
  const [servicesError, setServicesError] = useState("");

  useEffect(() => {
    Promise.all([api<Row[]>("/catalog"), api<Row[]>("/experiments")])
      .then(([c, e]) => {
        setCatalogs(c);
        setExperiments(e);
        if (c[0]) {
          setDataset(c[0].name);
          setScene(c[0].scenes[0]?.scene ?? "");
        }
      })
      .catch((e) => setError(String(e)));
  }, []);
  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const rows = await api<Row[]>("/harness/services");
        if (active) {
          setServices(rows);
          setChosen((previous) => {
            const next = { ...previous };
            for (const kind of ["planner", "environment", "vla"]) {
              if (next[kind] !== undefined) continue;
              const options = rows.filter((s) => s.spec.kind === kind);
              const initial = options.find((s) => s.available) ?? options[0];
              if (initial) next[kind] = initial.id;
            }
            if (rows.find((s) => s.id === next.vla)?.spec.metadata?.shared_inference && next.execution === undefined) {
              const pool = bindings(rows, next.vla);
              const first = pool.find((b) => b.worker.available && b.environment.available) ?? pool[0];
              next.environment = first?.environment.id ?? "";
              next.execution = first?.worker.id ?? "";
            }
            return next;
          });
          setServicesError("");
        }
      } catch {
        if (active) {
          setServices((rows) => rows.map((s) => ({ ...s, available: false, unavailable_reason: "暂时无法连接平台" })));
          setServicesError("暂时无法确认服务状态，请稍后重试。");
        }
      }
    };
    void refresh();
    const timer = setInterval(refresh, 10000);
    return () => { active = false; clearInterval(timer); };
  }, []);
  const choices = (kind: string) => services.filter((s) => s.spec.kind === kind);
  const pickService = (kind: string) => chosen[kind] ?? "";
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
  const options = (kind: string): [string, string, boolean][] => choices(kind).map((s) => [s.id, s.spec.name + (s.available ? "" : " · 不可用"), !s.available]);
  const environment = pickService("environment");
  const vla = pickService("vla");
  const planner = pickService("planner");
  const execution = pickService("execution");
  const shared = services.find((s) => s.id === vla)?.spec.metadata?.shared_inference === true;
  const pool = bindings(services, vla);
  const environments = choices("environment").filter((s) => !shared || pool.some((b) => b.environment.id === s.id));
  const workers = pool.filter((b) => b.environment.id === environment).map((b) => b.worker);
  const serviceOptions = (rows: Row[]): [string, string, boolean][] => rows.map((s) => [s.id, s.spec.name + (s.available ? "" : " · 不可用"), !s.available]);
  const selectVla = (id: string) => {
    const first = bindings(services, id).find((b) => b.worker.available && b.environment.available) ?? bindings(services, id)[0];
    const isShared = services.find((s) => s.id === id)?.spec.metadata?.shared_inference;
    setChosen({ ...chosen, vla: id, ...(isShared ? { environment: first?.environment.id ?? "", execution: first?.worker.id ?? "" } : {}) });
  };
  // Reuse the latest frozen configuration for the same environment and actor when available.
  const base = experiments.find((e) => e.spec.vla_service === vla && (shared
    ? e.spec.execution_pool?.some((b: Row) => b.environment_service === environment && b.execution_service === execution)
    : e.spec.environment_service === environment))?.spec.config;
  const backend = chosen.backend ?? base?.vla?.backend ?? "lightnav";
  const canStart = !!pick && [environment, vla, planner, ...(shared ? [execution] : [])].every((id) => services.some((s) => s.id === id && s.available))
    && (!shared || workers.some((s) => s.id === execution)) && !busy;

  const start = async () => {
    if (!canStart || !pick || !catalog || submitting.current) return;
    submitting.current = true;
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
      config.harness = { ...config.harness, mode: "agentic" };
      config.planner = agenticPlanner(config.planner);
      const { id } = await api<Row>("/experiments", {
        name: `Harness · ${pick.scene} · ${pick.index}`,
        planner_service: planner || null,
        environment_service: shared ? null : environment,
        execution_pool: shared ? [{ environment_service: environment, execution_service: execution }] : [],
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
      submitting.current = false;
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
          <Pill label="数据集" value={dataset} options={catalogs.map((c) => [c.name, `${c.dataset} ${c.split}`])} onChange={(value) => {
            setDataset(value);
            setScene(catalogs.find((c) => c.name === value)?.scenes[0]?.scene ?? "");
            setPick(null);
            setMatches([]);
          }} />
          <Pill
            label="场景"
            value={scene}
            options={(catalog?.scenes ?? []).map((s: Row) => [s.scene, `${s.scene} · ${s.count}`])}
            onChange={(v) => {
              setScene(v);
              setPick(null);
            }}
          />
          <ModelPicker services={choices("planner")} value={planner} onChange={(value) => setChosen({ ...chosen, planner: value })} />
          <Pill label="环境" value={environment} options={serviceOptions(environments)} onChange={(id) => {
            const first = pool.find((b) => b.environment.id === id && b.worker.available) ?? pool.find((b) => b.environment.id === id);
            setChosen({ ...chosen, environment: id, ...(shared ? { execution: first?.worker.id ?? "" } : {}) });
          }} />
          <Pill label="VLA" value={vla} options={options("vla")} onChange={selectVla} />
          {shared && <Pill label="执行器" value={execution} options={serviceOptions(workers)} onChange={(id) => setChosen({ ...chosen, execution: id })} />}
          <Pill label="动作模型" value={backend} options={[["lightnav", "LightNav"], ["navgpt", "NavGPT"]]} onChange={(v) => setChosen({ ...chosen, backend: v })} />
          <span className="spacer" />
          <button className="send" aria-label="开始运行" disabled={!canStart} onClick={start}>
            <Icon name="send" />
          </button>
        </div>
      </div>
      {(error || servicesError) && <p className="composer-error" role="alert">{error || servicesError}</p>}
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
