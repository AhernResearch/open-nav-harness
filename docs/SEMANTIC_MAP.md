# 语义地图MCP客户端接入

Harness只负责连接外部MCP、发现工具、传递参数和图片、展示结果，以及取消请求。地图加载、检索、渲染、HTTP/stdio服务和deep_search均由相邻`sementic_map_mcp`项目提供；Harness不导入地图Python包，也不维护地图Agent循环或模型配置。

## 客户端配置

在平台`conversations_config`指定的外部JSON中，给`mcp`增加连接（其他profiles/default_model配置保持原样）：

```json
{
  "mcp": {
    "map": {
      "url": "http://127.0.0.1:8880/mcp",
      "credential_env": "MAP_MCP_TOKEN"
    }
  }
}
```

URL填写实际服务地址；令牌通过Harness进程环境注入。新会话选择`map`连接后，客户端自动发现服务端工具，名称加连接别名前缀，如`map__deep_search`。工具的参数、默认值和说明以服务端公布的schema为准，不在Harness复制。

通用客户端位于`open_nav_harness/platform/conversations.py`的`CombinedTools`。它执行MCP initialize/list_tools/call_tool，传递JSON和图片，并管理连接生命周期。服务端未启用deep_search时，客户端仅看到已启用的底层工具。

## 服务端迁移

地图模块已迁至`sementic_map_mcp`。旧`NAVGPT_MAP_*`本地地图接入已移除；服务端使用`MAP_*`会话参数或命令行参数。Harness无需场景目录、编码器或deep_search配置，也无需地图项目PYTHONPATH。

服务端启动、依赖、授权和历史验证见相邻项目`docs/MCP_MIGRATION_20261116.md`及`docs/DEEP_SEARCH.md`。既有端点和Bearer不因源码迁移改变；实际运行服务切换必须按其拥有者记录执行。


## 整理与验证（2026-11-16）

交互会话复用ManagedRuntime，数据库迁移0003/0004保存会话、事件、输入和冻结配置。前端复用Harness时间线，外部工具显示服务端工具名，不依赖固定的map连接别名或地图工具清单。地图专用服务扩展已移出。

同时修正禁用reset_context后的工具说明清理：只匹配说明行本身，不依赖Python版本对docstring的缩进处理。未改变导航执行行为。

在既有外部Python环境执行：

```bash
export UV_PROJECT_ENVIRONMENT=/home/agiuser/lx_workspace/envs/open-nav-harness
# uv为该环境bin/uv；--no-sync沿用已安装依赖。
uv run --no-sync pytest -q
uv run --no-sync ruff check open_nav_harness
uv run --no-sync mypy open_nav_harness
npm --prefix frontend run build
ONH_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python" ONH_CHROME=/usr/bin/google-chrome npm --prefix frontend test
```

结果：253项Python测试和4个子测试通过；17项Playwright测试通过；Ruff、mypy、TypeScript和前端构建通过。保留FastAPI测试客户端的一项弃用警告。首次reset_context修复仍受docstring缩进影响，复测失败后改为匹配说明行，最终全量通过。浏览器使用隔离临时状态与模拟工具，没有调用真实模型或机器人，也没有重启已有服务。
