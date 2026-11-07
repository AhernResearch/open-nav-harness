import { useState } from "react";
import { api, type ExperimentSpec } from "../api";
import { pretty } from "../components";
import type { WorkspaceProps } from "../workspace";
export function ExperimentsPage({
  services,
  experiments,
  perform,
  refresh,
  setNotice,
  setPage,
}: WorkspaceProps) {
  const [expName, setExpName] = useState("Local Planner comparison"),
    [planner, setPlanner] = useState(""),
    [environment, setEnvironment] = useState(""),
    [vla, setVla] = useState(""),
    [backend, setBackend] = useState("lightnav"),
    [episodes, setEpisodes] = useState("0,1,2"),
    [split, setSplit] = useState("train"),
    [mode, setMode] = useState("agentic");
  const choices = (k: string) =>
    services
      .filter((s) => s.spec.kind === k)
      .map((s) => (
        <option key={s.id} value={s.id}>
          {s.spec.name}
        </option>
      ));
  const createExperiment = async () => {
    const spec: ExperimentSpec = {
      name: expName,
      planner_service: planner || null,
      environment_service: environment,
      vla_service: vla || null,
      episodes: episodes.split(",").map((x) => Number(x.trim())),
      purpose: "development",
      config: {
        dataset: { name: "r2r", split },
        harness: { mode },
        vla: { backend },
        planner: { max_turns: 100, episode_timeout_s: 1200 },
        environment: { max_steps: 500 },
      },
    };
    await api("/experiments", spec);
    await refresh();
    setNotice("实验配置已冻结，可以加入队列。");
  };
  return (
    <>
      <section>
        <h2>新建实验</h2>
        <p>配置保存后不可变。新参数请创建新实验，避免结果混用。</p>
        <div className="form-grid">
          <label>
            实验名称
            <input
              value={expName}
              onChange={(e) => setExpName(e.target.value)}
            />
          </label>
          <label>
            Planner
            <select
              aria-label="Planner"
              value={planner}
              onChange={(e) => setPlanner(e.target.value)}
            >
              <option value="">不使用 Planner</option>
              {choices("planner")}
            </select>
          </label>
          <label>
            Environment
            <select
              aria-label="Environment"
              value={environment}
              onChange={(e) => setEnvironment(e.target.value)}
            >
              <option value="">选择环境</option>
              {choices("environment")}
            </select>
          </label>
          <label>
            VLA
            <select aria-label="VLA" value={vla} onChange={(e) => setVla(e.target.value)}>
              <option value="">选择执行器</option>
              {choices("vla")}
            </select>
          </label>
          <label>
            执行器类型
            <select
              value={backend}
              onChange={(e) => setBackend(e.target.value)}
            >
              <option>lightnav</option>
              <option>navgpt</option>
            </select>
          </label>
          <label>
            模式
            <select aria-label="模式" value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="agentic">自主规划与上下文管理</option>
              <option value="subtask">严格子任务闭环（基线）</option>
              <option value="global">全局委托</option>
            </select>
          </label>
          <label>
            数据划分
            <input value={split} onChange={(e) => setSplit(e.target.value)} />
          </label>
          <label>
            Episode 索引（逗号分隔）
            <input
              value={episodes}
              onChange={(e) => setEpisodes(e.target.value)}
            />
          </label>
        </div>
        <button onClick={() => perform(createExperiment)}>
          校验并冻结配置
        </button>
      </section>
      <section>
        <h2>已冻结实验</h2>
        {experiments.map((e) => (
          <article key={e.id}>
            <div className="section-head">
              <div>
                <h3>{e.spec.name}</h3>
                <p>
                  {e.spec.episodes.length} episodes ·{" "}
                  {e.spec.config.harness?.mode}
                </p>
              </div>
              <button
                onClick={() =>
                  perform(async () => {
                    await api("/jobs", { experiment_id: e.id });
                    setPage("queue");
                    await refresh();
                  })
                }
              >
                加入队列 →
              </button>
            </div>
            <details>
              <summary>配置快照</summary>
              <pre>{pretty(e.spec)}</pre>
            </details>
          </article>
        ))}
      </section>
    </>
  );
}
