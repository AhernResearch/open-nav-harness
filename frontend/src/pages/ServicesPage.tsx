import { useState, useEffect } from "react";
import { api, type Row, type ServiceSpec } from "../api";
import { status, Empty } from "../components";
import type { WorkspaceProps } from "../workspace";
export function ServicesPage({
  services,
  templates,
  perform,
  refresh,
  setNotice,
}: WorkspaceProps) {
  const [health, setHealth] = useState<Record<string, Row>>({}),
    [gpus, setGpus] = useState<Row[]>([]);
  const [modelName, setModelName] = useState("Qwen3.6-27B-MM"),
    [serviceName, setServiceName] = useState("qwen27b"),
    [serviceUrl, setServiceUrl] = useState("http://127.0.0.1:8001/v1"),
    [kind, setKind] = useState<"planner" | "environment" | "vla" | "execution">("planner");
  const [managed, setManaged] = useState(false),
    [template, setTemplate] = useState(""),
    [gpuIds, setGpuIds] = useState("3"),
    [reserve, setReserve] = useState(60000);
  const [runtime, setRuntime] = useState<
    "openai_compatible" | "codex" | "claude"
  >("openai_compatible");
  const [contextLimit, setContextLimit] = useState(16384),
    [outputTokens, setOutputTokens] = useState(2048),
    [concurrency, setConcurrency] = useState(1),
    [shared, setShared] = useState(false),
    [environmentUrl, setEnvironmentUrl] = useState(""),
    [inferenceUrl, setInferenceUrl] = useState("");
  const [contextPolicy, setContextPolicy] = useState<"bounded" | "full">(
      "bounded",
    ),
    [thinking, setThinking] = useState(false);
  const register = async () => {
    const spec: ServiceSpec = {
      name: serviceName,
      kind,
      url: serviceUrl,
      managed,
      max_concurrency: concurrency,
      metadata: kind === "execution" ? { environment_url: environmentUrl, inference_url: inferenceUrl }
        : kind === "vla" ? { shared_inference: shared } : {},
      template: managed ? template : null,
      gpu_ids: managed && gpuIds.trim() ? gpuIds.split(",").map(Number) : [],
      reserve_mib: managed ? reserve : 0,
      planner:
        kind === "planner"
          ? {
              model: modelName,
              base_url: serviceUrl,
              runtime,
              api_format: "chat_completions",
              replay_reasoning: true,
              token_count_mode: "remote",
              image_token_estimate: 4096,
              context_limit: contextLimit,
              output_tokens: outputTokens,
              max_images: 32,
              context_policy:
                runtime === "openai_compatible" ? contextPolicy : "full",
              thinking,
              temperature: 0,
              request_timeout_s: 120,
              max_concurrency: concurrency,
            }
          : null,
    };
    await api("/services", spec);
    await refresh();
    setNotice("服务已登记；注册不会启动或修改服务。");
  };
  useEffect(() => {
    api<Row[]>("/gpus")
      .then(setGpus)
      .catch(() => {});
  }, []);
  return (
    <>
      <section>
        <div className="section-head">
          <h2>服务清单</h2>
          <button
            onClick={() =>
              perform(async () => {
                const pairs = await Promise.all(
                  services.map(async (s) => [
                    s.id,
                    await api("/services/" + s.id + "/health"),
                  ]),
                );
                setHealth(Object.fromEntries(pairs));
                setGpus(await api("/gpus"));
              })
            }
          >
            检查状态
          </button>
        </div>
        <div className="cards">
          {services.map((s) => (
            <article key={s.id}>
              <div className="section-head">
                <h3>{s.spec.name}</h3>
                {status(
                  health[s.id]
                    ? health[s.id].ok
                      ? "ready"
                      : "unreachable"
                    : s.status,
                )}
              </div>
              <p>
                {s.spec.kind} ·{" "}
                {s.spec.managed ? "平台托管" : "外部服务 · 只读"}
              </p>
              <code>{s.spec.url}</code>
              {s.spec.planner && (
                <p>
                  {s.spec.planner.model}
                  <br />
                  {s.spec.planner.context_limit.toLocaleString()} context ·{" "}
                  {s.spec.planner.context_policy}
                </p>
              )}
              {health[s.id]?.error && <small>{health[s.id].error}</small>}
              {s.spec.managed && (
                <div className="actions">
                  {["start", "stop"].map((action) => (
                    <button
                      key={action}
                      onClick={() =>
                        perform(async () => {
                          await api("/services/" + s.id + "/action", {
                            action,
                          });
                          await refresh();
                        })
                      }
                    >
                      {action === "start" ? "启动" : "停止"}
                    </button>
                  ))}
                  <button
                    onClick={() =>
                      perform(async () =>
                        setNotice(
                          (await api("/services/" + s.id + "/logs")).text,
                        ),
                      )
                    }
                  >
                    日志
                  </button>
                </div>
              )}
            </article>
          ))}
        </div>
        {!services.length && (
          <Empty
            title="连接你的第一个模型"
            text="支持注册已有服务，或使用管理员配置的托管模板。"
          />
        )}
      </section>
      <section>
        <h2>注册服务</h2>
        <div className="form-grid">
          <label>
            名称
            <input
              value={serviceName}
              onChange={(e) => setServiceName(e.target.value)}
            />
          </label>
          <label>
            类型
            <select
              aria-label="类型"
              value={kind}
              onChange={(e) => setKind(e.target.value as typeof kind)}
            >
              <option value="planner">Planner</option>
              <option value="environment">Environment</option>
              <option value="vla">VLA</option>
              <option value="execution">Execution</option>
            </select>
          </label>
          <label>
            API 地址
            <input
              value={serviceUrl}
              onChange={(e) => setServiceUrl(e.target.value)}
            />
          </label>
          {kind === "planner" && (
            <label>
              模型 ID
              <input
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
              />
            </label>
          )}
          {kind === "planner" && (
            <>
              <label>
                运行接口
                <select
                  value={runtime}
                  onChange={(e) => setRuntime(e.target.value as typeof runtime)}
                >
                  <option value="openai_compatible">
                    OpenAI compatible API
                  </option>
                  <option value="codex">Codex SDK（本机登录）</option>
                  <option value="claude">Claude SDK</option>
                </select>
              </label>
              <label>
                上下文窗口
                <input
                  type="number"
                  value={contextLimit}
                  onChange={(e) => setContextLimit(Number(e.target.value))}
                />
              </label>
              <label>
                输出上限
                <input
                  type="number"
                  value={outputTokens}
                  onChange={(e) => setOutputTokens(Number(e.target.value))}
                />
              </label>
              {runtime === "openai_compatible" && (
                <>
                  <label>
                    历史管理
                    <select
                      value={contextPolicy}
                      onChange={(e) =>
                        setContextPolicy(e.target.value as typeof contextPolicy)
                      }
                    >
                      <option value="bounded">保留任务状态，按预算裁剪</option>
                      <option value="full">完整历史</option>
                    </select>
                  </label>
                  <label className="checkbox">
                    <input
                      type="checkbox"
                      checked={thinking}
                      onChange={(e) => setThinking(e.target.checked)}
                    />
                    启用 thinking
                  </label>
                </>
              )}
            </>
          )}
          <label className="checkbox">
            <input
              type="checkbox"
              checked={managed}
              onChange={(e) => setManaged(e.target.checked)}
            />
            平台托管
          </label>
          {managed && (
            <>
              <label>
                服务模板
                <select
                  value={template}
                  onChange={(e) => setTemplate(e.target.value)}
                >
                  <option value="">选择预设</option>
                  {templates.map((t) => (
                    <option key={t.name}>{t.name}</option>
                  ))}
                </select>
              </label>
              <label>
                GPU 编号
                <input
                  value={gpuIds}
                  onChange={(e) => setGpuIds(e.target.value)}
                />
              </label>
              <label>
                每卡预留 MiB
                <input
                  type="number"
                  value={reserve}
                  onChange={(e) => setReserve(Number(e.target.value))}
                />
              </label>
            </>
          )}
        </div>
        <div className="form-grid">
          {(kind === "planner" || kind === "vla") && <label>并发容量
            <input type="number" min={1} max={32} value={concurrency} onChange={e => setConcurrency(Number(e.target.value))} />
          </label>}
          {kind === "vla" && <label><input type="checkbox" checked={shared} onChange={e => setShared(e.target.checked)} />共享推理服务</label>}
          {kind === "execution" && <>
            <label>环境地址<input value={environmentUrl} onChange={e => setEnvironmentUrl(e.target.value)} /></label>
            <label>共享推理地址<input value={inferenceUrl} onChange={e => setInferenceUrl(e.target.value)} /></label>
          </>}
        </div>
        <button onClick={() => perform(register)}>注册服务</button>
      </section>
      <section>
        <h2>GPU 状态</h2>
        {gpus.length ? (
          gpus.map((g) => (
            <pre key={g.host}>{g.host + "\n" + (g.csv || g.error)}</pre>
          ))
        ) : (
          <p>配置远程服务模板后显示 GPU 状态。</p>
        )}
      </section>
    </>
  );
}
