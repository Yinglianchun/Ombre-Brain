# Zeabur 部署说明（二改版）

更新时间：2026-06-15

## 结论

本 fork 不是原版 Ombre-Brain 的 Zeabur 一键部署形态。

原版 quick deploy 通常只需要一个 `server.py` 服务和一个 buckets 目录；本 fork 的完整生产形态是：

- `server.py`：MCP / Dashboard / bucket API，容器内监听 `8000`
- `gateway.py`：OpenAI-compatible Gateway / 记忆召回注入，默认监听 `8010`
- `/data`：bucket Markdown 文件和 `embeddings.db`
- `/state`：`gateway_state.db`、`memory_moments.sqlite`、`memory_nodes.sqlite`、`memory_edges.jsonl`、persona/portrait/dream 等运行态文件
- `config.yaml`：私有身份、上游模型、召回、persona、dream、reflection 等配置

所以它可以在 Zeabur 上试跑，但不应宣传成“直接点一键部署即可完整可用”。

## 为什么不是直接一键

当前仓库的 `Dockerfile` 默认只启动：

```text
python server.py
```

而完整二改版需要同时运行 `server.py` 和 `gateway.py`，并且两边看到同一份 `/data` 和 `/state`。

普通 Zeabur Git 服务更适合单 Web 端口。Zeabur 文档也明确 Git service 只允许一个端口；多端口和更细的端口配置更偏 Docker service / Gateway / 模板形态。

Zeabur Volume 可以持久化目录，但如果把 `server.py` 和 `gateway.py` 拆成两个 Zeabur 服务，必须确认两个服务能读写同一份 bucket/state 数据。否则会出现：

```text
server.py 看到一套 buckets/state
gateway.py 看到另一套 buckets/state
=> Dashboard 和 Gateway 记忆不同步
```

这一点没有验证前，不要给用户完整一键承诺。

## 2C2G 能不能跑

可以勉强跑。当前 VPS 也是 2 核 2G，生产可用。

但 2C2G 的边界是：

- 正常聊天、少量 bucket、轻量召回可以跑。
- 大量 embedding backfill、批量迁移、moment graph rebuild 会慢。
- 构建镜像时也可能吃内存。
- 如果平台有 idle sleep，Gateway 第一次请求会有冷启动延迟。

建议把 `backfill_embeddings.py` 的 batch size 降到 `5` 或 `10`，不要一上来跑大批量重建。

## 推荐部署等级

### A. 完整生产：继续用 VPS / Docker Compose

这是当前已验证形态。

参考：

```text
compose.hk.yml
```

它启动两个容器：

```text
ombre-brain    -> python server.py   -> 8000
ombre-gateway  -> python gateway.py  -> 8010
```

两者共享：

```text
/srv/ombre-brain/buckets -> /data
/srv/ombre-brain/state   -> /state
/srv/ombre-brain/config.yaml -> /app/config.yaml
```

### B. Zeabur 单服务：MCP / Dashboard-only

适合只是想先跑 Dashboard、bucket 管理、MCP 工具。

限制：

- 没有 OpenAI-compatible Gateway 自动注入。
- 不能直接作为聊天客户端 `/v1/chat/completions` 网关。

配置：

```text
Start command: python server.py
Port: 8000
```

环境变量：

```text
OMBRE_TRANSPORT=streamable-http
OMBRE_BUCKETS_DIR=/data
OMBRE_STATE_DIR=/state
```

Volumes：

```text
/data
/state
```

### C. Zeabur 单服务：Gateway-only

适合只想接聊天客户端，测试记忆召回注入。

限制：

- 没有 Dashboard。
- 没有 MCP bucket 工具。
- 不适合需要频繁管理/修桶的用户。

配置：

```text
Start command: python gateway.py
Port: 8010
```

环境变量：

```text
OMBRE_TRANSPORT=streamable-http
OMBRE_BUCKETS_DIR=/data
OMBRE_STATE_DIR=/state
OMBRE_GATEWAY_TOKEN=<your-token-if-enabled>
```

还需要按 `config.yaml` 里的 `gateway.upstreams` 配对应上游 key，例如：

```text
OMBRE_GATEWAY_PROVIDER_A_API_KEY=<your-model-api-key>
OMBRE_EMBEDDING_API_KEY=<your-embedding-key>
```

Volumes：

```text
/data
/state
```

### D. Zeabur 完整双服务：只建议验证后再写模板

理论目标：

```text
service 1: ombre-brain    -> python server.py
service 2: ombre-gateway  -> python gateway.py
shared /data
shared /state
same config.yaml
```

实际风险：

- 普通 Git service 单端口限制明显。
- 两个服务是否能共享同一份 Volume，需要按当前 Zeabur 项目形态验证。
- 如果用两个独立 Volume，不能算完整部署。
- 如果用一个容器启动两个进程，又要处理多端口暴露和进程管理。

因此暂时不建议把这个写成仓库里的 `zeabur.yaml` 一键模板。

## config.yaml 怎么处理

不要把私有 `config.yaml` 打进公开镜像或公开仓库。

推荐方式：

1. 从 `config.example.yaml` 复制一份。
2. 改自己的：
   - `identity`
   - `persona.profile_id`
   - `gateway.default_session_id`
   - `gateway.upstreams`
   - `embedding`
   - `reflection`
   - `dream`
3. 在 Zeabur 里通过文件管理 / 配置编辑 / 启动命令写入 `/app/config.yaml`。

如果只能使用环境变量，不方便挂 `config.yaml`，不要硬部署完整二改版。当前配置项太多，全部环境变量化会比 VPS 复杂。

## 旧 bucket 是否需要迁移

需要看来源。

### 1. 原版 Ombre-Brain buckets

bucket 本体是 Markdown 文件，可以迁移。

通常需要迁移：

```text
permanent/
dynamic/
archive/
```

如果旧部署里还有：

```text
embeddings.db
```

它可以不迁移。`embeddings.db` 是派生索引，不是记忆本体。

只有在以下条件都满足时，才考虑直接带过去：

- embedding 模型相同
- embedding 维度相同
- bucket 文本没有被迁移脚本改写

否则建议迁移 Markdown 后重建 embedding。

### 2. 二改版到二改版

如果是同一个用户、同一个部署迁移，可以迁移：

```text
/data
/state
```

其中 `/state` 里包含 session、persona、portrait、moment graph 等运行态。换用户或新部署不要复制别人的 `/state`。

### 3. 只迁移记忆，不迁移运行态

这是最安全的方式：

```text
只迁移 /data 下的 bucket Markdown
不迁移 /state
到新环境后重建索引
```

## 迁移步骤

下面用目标路径 `/data` 和 `/state` 表示 Zeabur 容器内路径。

### 1. 在旧环境打包 buckets

如果旧环境是普通目录：

```bash
cd /path/to/old/Ombre-Brain
tar -czf ombre-buckets.tar.gz buckets
```

如果旧环境已经是二改版 VPS：

```bash
tar -czf ombre-buckets.tar.gz /srv/ombre-brain/buckets
```

### 2. 上传到 Zeabur 服务

可以用 Zeabur Files / Command 功能上传或拉取压缩包。

目标是让容器里出现：

```text
/tmp/ombre-buckets.tar.gz
```

### 3. 解压到 `/data`

如果压缩包里顶层目录是 `buckets/`：

```bash
mkdir -p /data
tar -xzf /tmp/ombre-buckets.tar.gz -C /data --strip-components=1
```

如果压缩包里是 `/srv/ombre-brain/buckets/...` 这种绝对路径打出来的结构，先看一下：

```bash
tar -tzf /tmp/ombre-buckets.tar.gz | head
```

再按实际层级调整 `--strip-components`。

### 4. 刷新 moment index

仓库里有迁移脚本，适合从旧 buckets 目录复制到新 buckets 目录时使用：

```bash
python scripts/migrate_bucket_files.py \
  --source /tmp/old-buckets \
  --target-buckets-dir /data \
  --target-state-dir /state \
  --output /state/bucket_file_migration_plan.json
```

确认报告后再 apply：

```bash
python scripts/migrate_bucket_files.py \
  --source /tmp/old-buckets \
  --target-buckets-dir /data \
  --target-state-dir /state \
  --apply \
  --yes \
  --refresh-moments \
  --output /state/bucket_file_migration_apply.json
```

如果你是手工解压覆盖到 `/data`，至少重启服务，让运行时重新读 buckets。需要更完整的 moment graph，再跑：

```bash
python scripts/build_moment_graph.py --write --force
```

### 5. 重建 embedding

如果新环境配置了 embedding key：

```bash
python backfill_embeddings.py --refresh-all --batch-size 10
```

2C2G 更保守：

```bash
python backfill_embeddings.py --refresh-all --batch-size 5
```

### 6. 可选：重建 Word Map

如果启用了 Word Map Lite：

```bash
python scripts/build_word_map.py
```

## 部署后检查

MCP / Dashboard-only：

```bash
curl http://127.0.0.1:8000/health
```

Gateway-only：

```bash
curl http://127.0.0.1:8010/health
```

检查 buckets 数量：

```bash
find /data -name "*.md" | wc -l
```

检查索引文件：

```bash
ls -lh /data/embeddings.db /state/memory_moments.sqlite /state/gateway_state.db 2>/dev/null
```

## 给用户的简短解释

可以这样回复想部署的人：

```text
二改版不是原版 Zeabur 一键部署。它完整运行需要 server.py + gateway.py 两个服务，并共享 buckets/state/config。2C2G 可以勉强跑，但普通 Zeabur Git 部署更适合单服务，所以建议先选 MCP/Dashboard-only 或 Gateway-only 试跑。旧 bucket 可以迁移，记忆本体是 Markdown 文件；embedding/state 属于索引和运行态，最好在新环境重建。
```

## 参考

- Zeabur Dockerfile 部署说明：https://zeabur.com/docs/en-US/deploy/methods/dockerfile
- Zeabur 环境变量说明：https://zeabur.com/docs/en-US/deploy/config/environment-variables
- Zeabur Volume 说明：https://zeabur.com/docs/en-US/data-management/volumes
- Zeabur Public Networking 端口说明：https://zeabur.com/docs/en-US/deploy/networking/public-networking
- Zeabur 自定义 Docker Image 说明：https://zeabur.com/docs/en-US/deploy/methods/custom-docker-image
