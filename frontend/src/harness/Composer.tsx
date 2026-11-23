import { useEffect, useRef, useState } from "react";
import { api, type Row } from "../api";
import { Icon } from "./icons";
import { ModelPicker } from "./ModelPicker";
import type { Conversation } from "./useConversation";
import { agenticPlanner } from "../run-profile";

type Pick = { index: number; episode_id: number | string; scene: string; instruction: string };

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
  const environmentMetadata = services.find((s) => s.id === environment)?.spec.metadata ?? {};
  const matchesDataset = !environmentMetadata.dataset || (environmentMetadata.dataset === catalog?.dataset
    && environmentMetadata.split === catalog?.split && (!environmentMetadata.scene_id || environmentMetadata.scene_id === scene));
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
    && matchesDataset && (!shared || workers.some((s) => s.id === execution)) && !busy;

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
      if (environmentMetadata.backend === "habitat_gs") {
        config.environment = { ...config.environment, ...environmentMetadata.environment_config,
          backend: "habitat_gs", scene_manifest: environmentMetadata.scene_manifest };
      }
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
  const [pending, setPending] = useState<Row | null>(null);
  useEffect(() => {
    setPending(null);
    if (!job) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const row = await api<Row>(`/harness/jobs/${encodeURIComponent(job)}`);
        if (!active) return;
        setPending(row);
        if (row.attempt_id) { onAttempt(row.attempt_id); return; }
        if (["cancelled", "failed", "interrupted", "completed"].includes(row.status)) return;
      } catch {
        if (active) setPending({ connection_error: true });
      }
      if (active) timer = setTimeout(poll, 1500);
    };
    void poll();
    return () => { active = false; clearTimeout(timer); };
  }, [job]);
  return pending;
}

/** An environment alias plus one of its dataset episodes: what a binding opens. */
function EnvironmentPicker({ conversation: c }: { conversation: Conversation }) {
  const [catalog, setCatalog] = useState<Row | null>(null);
  const [scene, setScene] = useState("");
  const [episodes, setEpisodes] = useState<Row[]>([]);
  const target = (c.options.navigation as Row[]).find((n) => n.name === c.navigation);
  useEffect(() => {
    setCatalog(null); setScene(""); setEpisodes([]);
    if (!target) return;
    api<Row[]>("/catalog").then(rows => {
      const found = rows.find((row) => row.name === target.dataset);
      setCatalog(found ?? null);
      setScene(found?.scenes?.[0]?.scene ?? "");
    }).catch(e => c.setError(String(e)));
  }, [target?.dataset]);
  useEffect(() => {
    if (!catalog || !scene) return;
    api<Row[]>(`/catalog/${catalog.name}/episodes?scene=${encodeURIComponent(scene)}&limit=200`)
      .then(setEpisodes).catch(e => c.setError(String(e)));
  }, [catalog?.name, scene]);
  const choose = (index: string) => void c.bind(c.navigation, index || null);  // the alias is already bound
  return <>
    <Pill label="环境" value={c.navigation} options={[["", "暂不绑定"], ...(c.options.navigation as Row[]).map((n) => [n.name, n.name] as [string, string])]} onChange={value => void c.bind(value, null)}/>
    {target && !c.episode && catalog && <>
      <Pill label="场景" value={scene} options={catalog.scenes.map((s: Row) => [s.scene, String(s.scene).split("/").at(-1)?.replace(".glb", "") ?? s.scene] as [string, string])} onChange={setScene}/>
      <Pill label="episode" value="" options={[["", "选择 episode"], ...episodes.map((e) => [String(e.index), `#${e.index} ${String(e.instruction ?? "").slice(0, 28)}`] as [string, string])]} onChange={choose}/>
    </>}
    {target && c.episode && <span className="pill"><Icon name="route" size={13}/>#{c.episode}</span>}
  </>;
}

/** The existing input surface also accepts arbitrary conversation instructions. */
export function ConversationComposer({ conversation: c }: { conversation: Conversation }) {
  const [message, setMessage] = useState("");
  useEffect(() => setMessage(""), [c.id]);
  const submitting = useRef(false);
  const send = async () => {
    if (submitting.current || !message.trim()) return;
    submitting.current = true;
    try { if (await c.send(message)) setMessage(""); }
    finally { submitting.current = false; }
  };
  const services = c.options.models.map((model: string) => ({ id: model, available: true, unavailable_reason: "新建会话可选择其他模型", spec: { name: "对话模型", planner: { model } } }));
  const input = <>
    <div className="composer">
      <textarea aria-label="指令" rows={2} placeholder="自由聊天、查找地图物体，或描述你想去的地方…" value={message} onChange={e => setMessage(e.target.value)} onKeyDown={e => {
        if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); void send(); }
      }}/>
      <div className="composer-bar">
        <ModelPicker disabled={!!c.id} services={services} value={c.model} onChange={c.setModel}/>
        {!c.id && <label className="pill"><input type="checkbox" checked={c.useMcp} onChange={e => c.setUseMcp(e.target.checked)}/>MCP 工具</label>}
        {c.id && c.useMcp && <span className="pill"><Icon name="map" size={13}/>MCP 工具</span>}
        {c.options.navigation.length > 0 && <EnvironmentPicker conversation={c}/>}
        <span className="spacer"/>
        {c.status === "running" && <button className="pill-button danger" onClick={() => void c.stop()}><Icon name="stop" size={14}/>停止</button>}
        <button className="send" aria-label="发送指令" disabled={!message.trim() || c.busy || !c.model} onClick={() => void send()}><Icon name="send"/></button>
      </div>
    </div>
    {c.error && <p className="composer-error" role="alert">{c.error}</p>}
  </>;
  return c.id ? <div className="conversation-input">{input}</div> : <div className="hero"><h1>今天让机器人去哪里？</h1><p className="muted">自由交流、查询地图，或下达导航指令。</p>{input}<p className="hint muted">{c.options.navigation.length ? "可以随时补充条件、改口或停止。" : "可直接聊天和查地图；连接执行环境后可进行导航。"}</p></div>;
}
