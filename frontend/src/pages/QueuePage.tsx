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
  const [selected, setSelected] = useState<Row | null>(null),
    [attempts, setAttempts] = useState<Row[]>([]);
  useEffect(() => {
    if (!selected) return;
    const update = () =>
      perform(async () =>
        setAttempts(await api<Row[]>("/jobs/" + selected.id + "/attempts")),
      );
    update();
    const timer = setInterval(update, 3000);
    return () => clearInterval(timer);
  }, [selected]);
  return (
    <>
      <div className="stats">
        {[
          ["运行中", jobs.filter((j) => j.status === "running").length],
          ["等待资源", jobs.filter((j) => j.status === "queued").length],
          ["已完成", jobs.filter((j) => j.status === "completed").length],
          ["已注册服务", services.length],
        ].map(([name, value]) => (
          <div key={name}>
            <span>{name}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
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
                        setSelected(j);
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
                    {j.completed} / {j.total}
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
                              完成当前条后暂停
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
      {selected && (
        <section>
          <h2>{selected.name} · 逐条记录</h2>
          <Attempts
            rows={attempts}
            open={(id) => perform(() => openAttempt(id))}
          />
        </section>
      )}
    </>
  );
}
