import { useState } from "react";
import { api, type Row } from "../api";
import { pretty, Attempts } from "../components";
import type { WorkspaceProps } from "../workspace";
export function ComparePage({ jobs, perform, openAttempt }: WorkspaceProps) {
  const [comparison, setComparison] = useState<Row[]>([]),
    [compareIds, setCompareIds] = useState<string[]>([]);
  const [imports, setImports] = useState<Row[]>([]),
    [importPath, setImportPath] = useState("");
  const [legacy, setLegacy] = useState<Row | null>(null);
  return (
    <>
      <section>
        <h2>选择运行</h2>
        <div className="choices">
          {jobs.map((j) => (
            <label className="checkbox" key={j.id}>
              <input
                type="checkbox"
                checked={compareIds.includes(j.id)}
                onChange={(e) =>
                  setCompareIds(
                    e.target.checked
                      ? [...compareIds, j.id]
                      : compareIds.filter((x) => x !== j.id),
                  )
                }
              />
              {j.name} · {j.id.slice(0, 6)}
            </label>
          ))}
        </div>
        <button
          onClick={() =>
            perform(async () =>
              setComparison(
                await api("/compare?job_ids=" + compareIds.join(",")),
              ),
            )
          }
        >
          配对查看
        </button>
        <p>保留失败与所有 attempt；成功率、服务费用与 trace 质量分别解读。</p>
      </section>
      {comparison.map((c) => (
        <section key={c.job_id}>
          <h2>{jobs.find((j) => j.id === c.job_id)?.name || c.job_id}</h2>
          <p>
            完成 {c.aggregate.routes_completed} 条 · SR{" "}
            {c.aggregate.sr === null
              ? "—"
              : (c.aggregate.sr * 100).toFixed(1) + "%"}{" "}
            · 平均 NE {c.aggregate.ne?.toFixed(2) ?? "—"} m · 平均耗时{" "}
            {c.aggregate.wall_s?.toFixed(1) ?? "—"} s · 执行错误（含超时）{" "}
            {c.aggregate.execution_errors}
          </p>
          <Attempts
            rows={c.rows}
            open={(id) => perform(() => openAttempt(id))}
          />
        </section>
      ))}
      <section>
        <h2>导入历史结果（只读）</h2>
        <label>
          允许目录下的运行路径
          <input
            value={importPath}
            onChange={(e) => setImportPath(e.target.value)}
          />
        </label>
        <div className="actions">
          <button
            onClick={() =>
              perform(async () => {
                await api("/imports", { path: importPath });
                setImports(await api("/imports"));
              })
            }
          >
            登记历史结果
          </button>
          <button
            onClick={() =>
              perform(async () => setImports(await api("/imports")))
            }
          >
            查看已导入
          </button>
        </div>
        {imports.map((x) => (
          <details key={x.id}>
            <summary>{x.path}</summary>
            <pre>{pretty(x.summary)}</pre>
            {(x.summary.episodes || []).map((e: Row) => (
              <details key={e.index}>
                <summary
                  onClick={() =>
                    perform(async () => {
                      const d = await api(
                        "/imports/" + x.id + "/episodes/" + e.index,
                      );
                      setLegacy({ ...d, id: x.id });
                    })
                  }
                >
                  查看 Episode {e.index}
                </summary>
              </details>
            ))}
          </details>
        ))}
        {legacy && (
          <article>
            <h3>历史 Episode · 只读</h3>
            <p>历史记录仅展示可观测事件，不代表完整模型输入。</p>
            <div className="images">
              {legacy.images.slice(0, 12).map((path: string) => (
                <img
                  key={path}
                  src={"/api/imports/" + legacy.id + "/files/" + path}
                />
              ))}
            </div>
            <pre>{pretty(legacy.events)}</pre>
          </article>
        )}
      </section>
    </>
  );
}
