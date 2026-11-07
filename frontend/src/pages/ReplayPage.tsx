import { useState, useEffect } from "react";
import { api, type Row } from "../api";
import { pretty, Empty } from "../components";
export function ReplayPage({
  detail,
  refresh,
}: {
  detail: Row | null;
  refresh: () => Promise<void>;
}) {
  const [eventIndex, setEventIndex] = useState(0);
  useEffect(() => {
    if (detail?.status !== "running") return;
    const timer = setInterval(() => refresh().catch(() => {}), 3000);
    return () => clearInterval(timer);
  }, [detail?.id, detail?.status]);
  const sourceEvents = detail?.requests?.length
    ? detail.requests
    : (detail?.events || []).map((e: Row, i: number) => ({ ...e, seq: i + 1 }));
  const traceEvents = sourceEvents.filter((e: Row) =>
    [
      "tool_result",
      "model_request",
      "model_response",
      "context_selection",
      "session_error",
      "tool_use",
      "assistant_text",
      "session_inputs",
    ].includes(e.kind),
  );
  const current = traceEvents[eventIndex];
  const [recoveryError, setRecoveryError] = useState("");
  let taskState: Row | null = null;
  for (const event of sourceEvents) {
    if (event.seq > (current?.seq ?? 0)) break;
    if (event.task_state) taskState = event.task_state;
    if (event.feedback?.content) {
      try {
        taskState =
          JSON.parse(event.feedback.content).subtask_state || taskState;
      } catch {}
    }
  }
  const policyEvents = (detail?.subtask_events || []).filter(
    (e: Row) => e.kind === "policy_context",
  );
  const [policyIndex, setPolicyIndex] = useState(0);
  const policy = policyEvents[policyIndex];
  const observedFrames = (detail?.subtask_events || []).filter(
    (e: Row) =>
      e.kind === "observation" && policy?.frame_ids.includes(e.frame_id),
  );
  const trajectory = (detail?.events || []).find(
    (e: Row) => e.kind === "episode_trajectory",
  );
  const images: Row[] = [];
  function collect(v: any) {
    if (!v || typeof v !== "object") return;
    if (v.artifact) {
      images.push(v);
      return;
    }
    Object.values(v).forEach((x) =>
      Array.isArray(x) ? x.forEach(collect) : collect(x),
    );
  }
  if (current) collect(current);
  const uniqueImages = images.filter(
    (x, i, a) => a.findIndex((y) => y.artifact === x.artifact) === i,
  );

  return !detail ? (
    <Empty
      title="选择一次执行记录"
      text="在任务队列中打开实验，选择某条 episode 查看实际输入和工具事件。"
    />
  ) : (
    <>
      <div className="section-head">
        <h2>
          Episode {detail.episode} / Attempt {detail.ordinal}
        </h2>
        <a href={"/api/attempts/" + detail.id + "/export"}>导出 trace ↓</a>
      </div>
      {detail.status === "blocked" && (
        <section>
          <h3>资源待确认</h3>
          <p>
            先在服务页停止本次使用的托管 Environment 和
            VLA，再释放本条记录的资源锁。
          </p>
          <button
            onClick={async () => {
              try {
                await api("/attempts/" + detail.id + "/reconcile", {});
                await refresh();
              } catch (e) {
                setRecoveryError(String(e));
              }
            }}
          >
            校验并释放资源锁
          </button>
          {recoveryError && <p role="alert">{recoveryError}</p>}
        </section>
      )}
      {!detail.requests?.length && (
        <p>SDK／历史记录：仅能展示运行时公开的事件。</p>
      )}
      <div className="replay-grid">
        <section className="timeline">
          <h3>事件时间线</h3>
          {traceEvents.map((e: Row, i: number) => (
            <button
              key={e.seq}
              className={i === eventIndex ? "selected" : ""}
              onClick={() => setEventIndex(i)}
            >
              <span>{String(e.seq).padStart(3, "0")}</span>
              {e.kind}
              <small>{e.name || ""}</small>
            </button>
          ))}
        </section>
        <section>
          <div className="section-head">
            <h2>{current?.kind || "执行记录"}</h2>
            <span>
              {eventIndex + 1} / {traceEvents.length}
            </span>
          </div>
          {taskState && (
            <article>
              <h3>该时刻的计划 · r{taskState.revision}</h3>
              <p>{taskState.summary}</p>
              <ol>
                {(taskState.tasks || []).map((t: Row) => (
                  <li key={t.id}>
                    {t.instruction}{" "}
                    <small>
                      {t.status} · {t.completion_criterion}
                    </small>
                  </li>
                ))}
              </ol>
            </article>
          )}
          <div className="images">
            {uniqueImages.map((im) => (
              <figure key={im.artifact}>
                <img
                  src={
                    "/api/attempts/" + detail.id + "/artifact/" + im.artifact
                  }
                />
                <figcaption>{im.artifact.slice(0, 12)}</figcaption>
              </figure>
            ))}
          </div>
          <pre>{pretty(current || detail.result)}</pre>
        </section>
      </div>
      {policy && (
        <section>
          <h2>VLA 实际上下文</h2>
          <label>
            策略请求
            <select
              value={policyIndex}
              onChange={(e) => setPolicyIndex(Number(e.target.value))}
            >
              {policyEvents.map((p: Row, i: number) => (
                <option key={p.event} value={i}>
                  #{i + 1} · {p.context_key}
                </option>
              ))}
            </select>
          </label>
          <p>{policy.instruction}</p>
          <div className="images">
            {observedFrames.map((f: Row) => {
              const names = f.images.policy_front
                ? ["policy_front"]
                : ["front", "right", "back", "left"];
              return names
                .filter((n) => f.images[n])
                .map((n) => {
                  const relative = f.images[n].path.slice(
                    detail.output.length + 1,
                  );
                  return (
                    <figure key={f.frame_id + n}>
                      <img
                        src={
                          "/api/attempts/" + detail.id + "/files/" + relative
                        }
                      />
                      <figcaption>
                        {f.frame_id} · {n}
                      </figcaption>
                    </figure>
                  );
                });
            })}
          </div>
        </section>
      )}
      {trajectory && (
        <section>
          <h2>执行轨迹（评测侧）</h2>
          <Trajectory
            agent={trajectory.agent_path || []}
            reference={trajectory.reference_path || []}
          />
          <p>绿色为实际轨迹，虚线为参考轨迹。参考轨迹仅用于此处复核。</p>
        </section>
      )}
      <section>
        <h2>结果与运行日志</h2>
        <pre>{pretty(detail.result)}</pre>
        <details>
          <summary>日志</summary>
          <pre>{detail.log}</pre>
        </details>
      </section>
    </>
  );
}

function Trajectory({
  agent,
  reference,
}: {
  agent: number[][];
  reference: number[][];
}) {
  const points = [...agent, ...reference];
  if (!points.length) return null;
  const xs = points.map((p) => p[0]),
    ys = points.map((p) => p[2]);
  const minX = Math.min(...xs),
    minY = Math.min(...ys),
    span = Math.max(Math.max(...xs) - minX, Math.max(...ys) - minY, 1);
  const project = (p: number[]) => [
    ((p[0] - minX) / span) * 440 + 30,
    ((p[2] - minY) / span) * 440 + 30,
  ];
  const line = (ps: number[][]) =>
    ps.map((p) => project(p).join(",")).join(" ");
  const start = project(agent[0] || reference[0]),
    end = project(agent[agent.length - 1] || reference[0]);
  return (
    <svg
      viewBox="0 0 500 500"
      role="img"
      aria-label="导航执行轨迹"
      style={{ width: "100%", maxHeight: 420 }}
    >
      <polyline
        points={line(reference)}
        stroke="#a4afa8"
        strokeWidth="2"
        strokeDasharray="5 5"
        fill="none"
      />
      <polyline
        points={line(agent)}
        stroke="#17634d"
        strokeWidth="3"
        fill="none"
      />
      <circle cx={start[0]} cy={start[1]} r="5" fill="#17634d" />
      <circle cx={end[0]} cy={end[1]} r="7" fill="#d69246" />
    </svg>
  );
}
