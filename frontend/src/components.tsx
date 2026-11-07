import type { Row } from "./api";
const labels: Record<string, string> = {
  queued: "等待资源",
  running: "运行中",
  completed: "已完成",
  cancelled: "已取消",
  paused: "已暂停",
  interrupted: "需要处理",
  failed: "失败",
  blocked: "资源状态待确认",
};
export const status = (s: string) => (
  <span className={"badge " + s}>{labels[s] || s}</span>
);
export const pretty = (v: unknown) => JSON.stringify(v, null, 2);
export function Empty({ title, text }: { title: string; text: string }) {
  return (
    <div className="empty">
      <span>↗</span>
      <h2>{title}</h2>
      <p>{text}</p>
    </div>
  );
}
export function Attempts({
  rows,
  open,
}: {
  rows: Row[];
  open: (id: string) => void;
}) {
  return rows.length ? (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Episode / 尝试</th>
            <th>运行状态</th>
            <th>任务成功</th>
            <th>NE / m</th>
            <th>SPL</th>
            <th>调用</th>
            <th>时间 / s</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.attempt_id}>
              <td>
                <button className="link" onClick={() => open(r.attempt_id)}>
                  {r.episode} / {r.attempt}
                </button>
              </td>
              <td>
                {status(r.status)}
                {r.error && <small>{r.error}</small>}
              </td>
              <td>{r.success === null ? "—" : r.success ? "成功" : "失败"}</td>
              <td>{r.ne?.toFixed(2) ?? "—"}</td>
              <td>{r.spl?.toFixed(3) ?? "—"}</td>
              <td>{r.tool_calls}</td>
              <td>{r.wall_s?.toFixed(1) ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  ) : (
    <p>尚无执行记录。</p>
  );
}
