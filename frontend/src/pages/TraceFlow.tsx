import { type Row } from "../api";
import { pretty } from "../components";

export function artifactImages(value: unknown): Row[] {
  const found = new Map<string, Row>();
  function visit(v: any) {
    if (!v || typeof v !== "object") return;
    if (v.artifact) found.set(v.artifact, v);
    else Object.values(v).forEach(visit);
  }
  visit(value);
  return [...found.values()];
}

export function TraceImages({
  images,
  attempt,
}: {
  images: Row[];
  attempt: string;
}) {
  return (
    <div className="images">
      {images.map((im) => (
        <figure key={im.artifact}>
          <a
            href={`/api/attempts/${attempt}/artifact/${im.artifact}`}
            target="_blank"
            rel="noreferrer"
          >
            <img
              loading="lazy"
              alt="本次调用返回的观测图像"
              src={`/api/attempts/${attempt}/artifact/${im.artifact}`}
            />
          </a>
          <figcaption>{im.artifact.slice(0, 12)} · 点击查看原图</figcaption>
        </figure>
      ))}
    </div>
  );
}

function parsed(value: unknown) {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

export function TraceFlow({ detail }: { detail: Row }) {
  const events: Row[] = detail.requests || [];
  const responses = events.filter((e) => e.kind === "model_response");

  return (
    <section aria-label="完整模型与工具流程">
      <h2>完整流程 · {responses.length} 次模型响应</h2>
      <p>
        以下文字、参数和反馈均来自原始日志。只输出工具调用的轮次会明确标注；未补写模型没有输出的解释。图像可点击查看原图。
      </p>
      {responses.map((event, index) => {
        const message = event.response?.choices?.[0]?.message;
        const following = events.filter(
          (e) =>
            e.seq > event.seq &&
            e.seq < (responses[index + 1]?.seq ?? Infinity),
        );
        const request = events
          .filter((e) => e.kind === "model_request" && e.seq < event.seq)
          .at(-1);
        return (
          <article
            className="trace-turn"
            key={event.seq}
            id={`turn-${index + 1}`}
          >
            <div className="section-head">
              <h3>第 {index + 1} 轮 · 模型输出</h3>
              <small>
                事件 #{event.seq} · {Number(event.elapsed_s || 0).toFixed(1)} 秒
              </small>
            </div>
            {message?.content ? (
              <pre className="model-text">
                {typeof message.content === "string"
                  ? message.content
                  : pretty(message.content)}
              </pre>
            ) : (
              <p>
                {message?.tool_calls?.length
                  ? "本轮只输出工具调用，没有额外文字。"
                  : "本轮没有记录到文字输出。"}
              </p>
            )}
            {(message?.reasoning || message?.reasoning_content) && (
              <details>
                <summary>模型接口返回的推理字段</summary>
                <pre>
                  {pretty(message.reasoning || message.reasoning_content)}
                </pre>
              </details>
            )}
            {(message?.tool_calls || []).map((call: Row) => {
              const dispatch = following.find(
                (e) => e.kind === "tool_dispatch" && e.call_id === call.id,
              );
              const result = following.find(
                (e) => e.kind === "tool_result" && e.call_id === call.id,
              );
              return (
                <div className="trace-tool" key={call.id}>
                  <h4>
                    {call.function?.name} ·{" "}
                    {result?.is_error
                      ? "返回错误"
                      : dispatch
                        ? "已调用"
                        : "未派发"}
                  </h4>
                  <pre>{pretty(parsed(call.function?.arguments))}</pre>
                  {result ? (
                    <details open={Boolean(result.is_error)}>
                      <summary>
                        工具返回
                        {result.is_error ? " · 拒绝原因" : " · 文字及图像"}
                      </summary>
                      <pre>
                        {(result.feedback?.content || "")
                          .split("\n")
                          .map((line: string) => {
                            const value = parsed(line);
                            return typeof value === "string"
                              ? value
                              : pretty(value);
                          })
                          .join("\n")}
                      </pre>
                      <TraceImages
                        images={artifactImages(result.images)}
                        attempt={detail.id}
                      />
                    </details>
                  ) : (
                    <p>尚无工具返回记录。</p>
                  )}
                </div>
              );
            })}
            <details>
              <summary>本轮完整输入与原始响应</summary>
              <h4>实际发送的输入</h4>
              <pre>{pretty(request?.request || null)}</pre>
              <TraceImages
                images={artifactImages(request)}
                attempt={detail.id}
              />
              <h4>原始响应</h4>
              <pre>{pretty(event.response)}</pre>
            </details>
          </article>
        );
      })}
      {events
        .filter((e) => e.kind === "session_error")
        .map((e) => (
          <pre key={e.seq}>{pretty(e)}</pre>
        ))}
    </section>
  );
}
