import { useEffect, useMemo, useSyncExternalStore } from "react";
import { Trace } from "./trace";

/** Follows one episode's event stream under `base`; a finished episode is the same stream replayed. */
export function useTrace(base: string | null): Trace {
  const trace = useMemo(() => new Trace(), [base]);
  useSyncExternalStore(trace.subscribe, trace.snapshot);
  useEffect(() => {
    if (!base) return;
    let source: EventSource;
    const open = () => {
      source = new EventSource(`${base}/stream`);
      source.onmessage = (m) => trace.apply(JSON.parse(m.data));
      source.addEventListener("end", (m) => {
        trace.apply([{ type: "attempt.status", ...JSON.parse((m as MessageEvent).data) }]);
        trace.finish();
        source.close();
      });
      source.addEventListener("reset", () => {
        source.close();
        trace.clear();
        open();
      });
      // Transport errors reconnect automatically with Last-Event-ID; nothing to raise.
      source.onerror = () => {};
    };
    open();
    return () => source.close();
  }, [trace]);
  return trace;
}
