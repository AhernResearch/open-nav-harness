import { useState, useEffect } from "react";
import { api, type Row } from "../api";
import { pretty, Empty } from "../components";
import { TraceFlow, TraceImages, artifactImages } from "./TraceFlow";
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
  const traceEvents = sourceEvents;
  const current = traceEvents[eventIndex];
  const [recoveryError, setRecoveryError] = useState("");
  let taskState: Row | null = null;
  for (const event of sourceEvents) {
    if (event.seq > (current?.seq ?? 0)) break;
    if (event.task_state) taskState = event.task_state;
    if (event.feedback?.content) {
      for (const line of event.feedback.content.split("\n")) {
        try {
          taskState = JSON.parse(line).subtask_state || taskState;
        } catch {}
      }
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
  const uniqueImages = artifactImages(current);
  const [view, setView] = useState("flow");
  const predictionEvents = (detail?.subtask_events || []).filter(
    (e: Row) =>
      policy?.prediction_id != null &&
      e.prediction_id === policy.prediction_id &&
      [
        "policy_prediction",
        "policy_execution_dispatch",
        "policy_execution_result",
      ].includes(e.kind),
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
      {detail.requests?.length > 0 && (
        <div className="trace-switch">
          <button
            aria-pressed={view === "flow"}
            onClick={() => setView("flow")}
          >
            完整流程
          </button>
          <button
            aria-pressed={view === "events"}
            onClick={() => setView("events")}
          >
            逐事件检查
          </button>
          <a href={`/workbench?attempt=${encodeURIComponent(detail.id)}`}>
            本次运行直达链接
          </a>
          <a href={`/harness?attempt=${encodeURIComponent(detail.id)}`}>
            在 Harness 中打开
          </a>
        </div>
      )}
      {view === "flow" && detail.requests?.length > 0 && (
        <TraceFlow detail={detail} />
      )}
      {(view === "events" || !detail.requests?.length) && (
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
            <TraceImages images={uniqueImages} attempt={detail.id} />
            <pre>{pretty(current || detail.result)}</pre>
          </section>
        </div>
      )}
      {policy && (
        <section>
          <h2>VLA 输入、轨迹输出与实际执行</h2>
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
          {predictionEvents.map((e: Row) => (
            <details key={e.event}>
              <summary>
                {
                  {
                    policy_prediction: "模型输出：原始轨迹点与动作",
                    policy_execution_dispatch: "实际下发的动作",
                    policy_execution_result: "仿真执行反馈",
                  }[e.kind as string]
                }
              </summary>
              <pre>{pretty(e)}</pre>
            </details>
          ))}
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
