import { useState, useEffect } from "react";
import { api, type Row } from "../api";
import { status, Empty, Attempts } from "../components";
import type { WorkspaceProps } from "../workspace";
export function QueuePage({
  jobs,
  services,
  perform,
  setPage,
  openAttempt,
}: WorkspaceProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null),
    [inference, setInference] = useState<Row[]>([]);
  useEffect(() => {
    let active = true;
    const update = () => api<Row[]>("/inference/status")
      .then(rows => { if (active) setInference(rows); }).catch(() => {});
    update();
    const timer = setInterval(update, 5000);
    return () => { active = false; clearInterval(timer); };
  }, []);
  const selected = jobs.find(job => job.id === selectedId);
  if (selected) return (
    <JobDetails
      key={selected.id}
      job={selected}
      back={() => setSelectedId(null)}
      open={(id) => perform(() => openAttempt(id))}
    />
  );
  return (
    <>
      <div className="stats">
        {[
          ["并发路线", jobs.reduce((n, j) => n + (j.active || 0), 0)],
          ["等待资源", jobs.filter((j) => j.status === "queued").length],
          ["已完成路线", jobs.reduce((n, j) => n + j.completed, 0)],
          ["已注册服务", services.length],
        ].map(([name, value]) => (
          <div key={name}>
            <span>{name}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
      {inference.length > 0 && <section aria-label="共享推理状态">
        <h2>共享推理</h2>
        {inference.map(s => <p key={s.service_id}>
          {s.name} · {s.ok
            ? `会话 ${s.details.sessions} · 在途 ${s.details.pending_requests} · 等待 ${s.details.waiting_requests ?? 0}`
            : "暂不可用"}
        </p>)}
      </section>}
      <section>
        <div className="section-head">
          <h2>实验运行</h2>
          <button onClick={() => setPage("experiments")}>创建实验 ＋</button>
        </div>
        {!jobs.length ? (
          <Empty
            title="还没有实验运行"
            text="先注册模型与环境，再创建一个实验。"
          />
        ) : (
          <table>
            <thead>
              <tr>
                <th>实验</th>
                <th>状态</th>
                <th>进度</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.id}>
                  <td>
                    <button
                      className="link"
                      onClick={() => {
                        setSelectedId(j.id);
                      }}
                    >
                      {j.name}
                    </button>
                    <small>{j.id.slice(0, 8)}</small>
                  </td>
                  <td>
                    {status(j.status)}
                    {j.error && <small>{j.error}</small>}
                  </td>
                  <td>
                    {j.completed} / {j.total}<small>并发 {j.active || 0}</small>
                  </td>
                  <td>
                    <div className="actions">
                      {!["completed", "cancelled", "failed"].includes(
                        j.status,
                      ) && (
                        <>
                          {["paused", "interrupted"].includes(j.status) ? (
                            <button
                              onClick={() =>
                                perform(() =>
                                  api("/jobs/" + j.id + "/action", {
                                    action: "resume",
                                  }),
                                )
                              }
                            >
                              恢复
                            </button>
                          ) : (
                            <button
                              onClick={() =>
                                perform(() =>
                                  api("/jobs/" + j.id + "/action", {
                                    action: "pause",
                                  }),
                                )
                              }
                            >
                              完成在途路线后暂停
                            </button>
                          )}
                          <button
                            className="danger"
                            onClick={() =>
                              perform(() =>
                                api("/jobs/" + j.id + "/action", {
                                  action: "cancel",
                                }),
                              )
                            }
                          >
                            取消
                          </button>
                        </>
                      )}
                      <a href={"/api/jobs/" + j.id + "/metrics.csv"}>CSV ↓</a>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

    </>
  );
}

function JobDetails({ job, back, open }: {
  job: Row;
  back: () => void;
  open: (id: string) => void;
}) {
  const [attempts, setAttempts] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    const update = async () => {
      try {
        const rows = await api<Row[]>("/jobs/" + job.id + "/attempts");
        if (active) { setAttempts(rows); setError(""); }
      } catch (e) {
        if (active) setError(String(e));
      } finally {
        if (active) setLoading(false);
      }
    };
    update();
    const timer = setInterval(update, 3000);
    return () => { active = false; clearInterval(timer); };
  }, [job.id]);
  return (
    <section aria-label="实验运行详情">
      <button className="quiet" onClick={back}>← 返回实验列表</button>
      <h2>{job.name} · 逐条记录</h2>
      <p>{status(job.status)} · 已完成 {job.completed} / {job.total} · 运行中 {job.active || 0}
        {job.pause_requested ? " · 当前路线结束后暂停，不再启动新路线" : ""}
      </p>
      {error && <p role="alert">记录加载失败：{error}</p>}
      {loading ? <p role="status">正在加载执行记录…</p> :
        <Attempts rows={attempts} open={open} />}
    </section>
  );
}
