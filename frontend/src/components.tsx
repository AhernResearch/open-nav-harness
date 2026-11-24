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
export function Empty({ title, text }: { title: string; text: string }) {
  return (
    <div className="empty">
      <span>↗</span>
      <h2>{title}</h2>
      <p>{text}</p>
    </div>
  );
}
