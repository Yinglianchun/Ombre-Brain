# Ombre Brain

一个给 Claude 用的长期情绪记忆系统。基于 Russell 效价/唤醒度坐标打标，Obsidian 做存储层，MCP 接入，带遗忘曲线和向量语义检索。

A long-term emotional memory system for Claude. Tags memories using Russell's valence/arousal coordinates, stores them as Obsidian-compatible Markdown, connects via MCP, with forgetting curve and vector semantic search.

> **⚠️ 备用链接 / Backup link**
> Gitea 备用地址（GitHub 访问有问题时用）：
> **https://git.p0lar1s.uk/P0lar1s/Ombre_Brain**

---

## 快速开始 / Quick Start（Docker Hub 预构建镜像，最简单）

> 不需要 clone 代码，不需要 build，三步搞定。
> 完全不会？没关系，往下看，一步一步跟着做。

### 第零步：装 Docker Desktop

1. 打开 [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
2. 下载对应你系统的版本（Mac / Windows / Linux）
3. 安装、打开，看到 Docker 图标在状态栏里就行了
4. **Windows 用户**：安装时会提示启用 WSL 2，点同意，重启电脑

### 第一步：打开终端

| 系统 | 怎么打开 |
|---|---|
| **Mac** | 按 `⌘ + 空格`，输入 `终端` 或 `Terminal`，回车 |
| **Windows** | 按 `Win + R`，输入 `cmd`，回车；或搜索「PowerShell」 |
| **Linux** | `Ctrl + Alt + T` |

打开后你会看到一个黑色/白色的窗口，可以输入命令。下面所有代码块里的内容，都是**复制粘贴到这个窗口里，然后按回车**。

### 第二步：创建一个工作文件夹

```bash
mkdir ombre-brain && cd ombre-brain
```

> 这会在你当前位置创建一个叫 `ombre-brain` 的文件夹，并进入它。

### 第三步：获取 API Key（免费）

1. 打开 [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. 用 Google 账号登录
3. 点击 **「Create API key」**
4. 复制生成的 key（一长串字母数字），待会要用

> 没有 Google 账号？也行，API Key 留空也能跑，只是脱水压缩效果差一点。

### 第四步：创建配置文件并启动

**一行一行复制粘贴执行：**

```bash
# 下载用户版 compose 文件
curl -O https://raw.githubusercontent.com/P0luz/Ombre-Brain/main/docker-compose.user.yml
```

```bash
# 创建 .env 文件——把 your-key-here 换成第三步拿到的 key
echo "OMBRE_API_KEY=your-key-here" > .env
```

```bash
# 拉取镜像并启动（第一次会下载约 500MB，等一会儿）
docker compose -f docker-compose.user.yml up -d
```

### 第五步：验证

```bash
curl http://localhost:8000/health
```

看到类似这样的输出就是成功了：
```json
{"status":"ok","buckets":0,"decay_engine":"stopped"}
```

浏览器打开前端 Dashboard：**http://localhost:8000/dashboard**

> 如果你用的是 `docker-compose.user.yml` 默认端口，地址就是 `http://localhost:8000/dashboard`。
> 如果你改了端口映射（比如 `18001:8000`），则是 `http://localhost:18001/dashboard`。

> **看到错误？** 检查 Docker Desktop 是否正在运行（状态栏有图标）。

### 第六步：接入 Claude

在 Claude Desktop 的配置文件里加上这段（Mac: `~/Library/Application Support/Claude/claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "ombre-brain": {
      "type": "streamable-http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

重启 Claude Desktop，你应该能在工具列表里看到 `breath`、`hold`、`grow` 等工具了。

> **想挂载 Obsidian？** 用任意文本编辑器打开 `docker-compose.user.yml`，把 `./buckets:/data` 改成你的 Vault 路径，例如：
> ```yaml
> - /Users/你的用户名/Documents/Obsidian Vault/Ombre Brain:/data
> ```
> 然后 `docker compose -f docker-compose.user.yml down && docker compose -f docker-compose.user.yml up -d` 重启。

> **后续更新镜像：**
> ```bash
> docker pull p0luz/ombre-brain:latest
> docker compose -f docker-compose.user.yml down && docker compose -f docker-compose.user.yml up -d
> ```

---

## 从源码部署 / Deploy from Source（Docker）

> 适合想自己改代码、或者不想用预构建镜像的用户。

**前置条件：** 电脑上装了 [Docker Desktop](https://www.docker.com/products/docker-desktop/)，并且已经打开。

**第一步：拉取代码**

(💡 如果主链接访问有困难，可用备用 Gitea 地址：https://git.p0lar1s.uk/P0lar1s/Ombre_Brain)

```bash
git clone https://github.com/P0luz/Ombre-Brain.git
cd Ombre-Brain
```

**第二步：创建 `.env` 文件**

在项目目录下新建一个叫 `.env` 的文件（注意有个点），内容填：

```
OMBRE_API_KEY=你的API密钥
```

> **🔑 推荐免费方案：Google AI Studio**
> 1. 打开 [aistudio.google.com/apikey](https://aistudio.google.com/apikey)，登录 Google 账号
> 2. 点击「Create API key」生成一个 key
> 3. 把 key 填入 `.env` 文件的 `OMBRE_API_KEY=` 后面
> 4. 免费额度（截至 2025 年，请以官网实时信息为准）：
>    - **脱水/打标模型**（`gemini-2.5-flash-lite`）：免费层 30 req/min
>    - **向量化模型**（`gemini-embedding-001`）：免费层 1500 req/day，3072 维
> 5. 在 `config.yaml` 中 `dehydration.base_url` 设为 `https://generativelanguage.googleapis.com/v1beta/openai`
>
> 也支持 DeepSeek、Ollama、LM Studio、vLLM 等任意 OpenAI 兼容 API。
>
> **Recommended free option: Google AI Studio**
> 1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and create an API key
> 2. Free tier (as of 2025, check official site for current limits):
>    - Dehydration model (`gemini-2.5-flash-lite`): 30 req/min free
>    - Embedding model (`gemini-embedding-001`): 1500 req/day free, 3072 dims
> 3. Set `dehydration.base_url` to `https://generativelanguage.googleapis.com/v1beta/openai` in `config.yaml`
> Also supports DeepSeek, Ollama, LM Studio, vLLM, or any OpenAI-compatible API.

没有 API key 则脱水压缩和自动打标功能不可用（会报错），但记忆的读写和检索仍正常工作。如果暂时不用脱水功能，可以留空：

```
OMBRE_API_KEY=
```

**第三步：配置 `docker-compose.yml`（指向你的 Obsidian Vault）**

用文本编辑器打开 `docker-compose.yml`，找到这一行：

```yaml
- ./buckets:/data
```

改成你的 Obsidian Vault 里 `Ombre Brain` 文件夹的路径，例如：

```yaml
- /Users/你的用户名/Documents/Obsidian Vault/Ombre Brain:/data
```

> 不知道路径？在 Obsidian 里右键那个文件夹 → 「在访达中显示」，然后把地址栏的路径复制过来。
> 不想挂载 Obsidian 也行，保持 `./buckets:/data` 不动，数据会存在项目目录的 `buckets/` 文件夹里。

**第四步：启动**

```bash
docker compose up -d
```

等它跑完，看到 `Started` 就好了。

**验证是否正常运行：**

```bash
docker logs ombre-brain
```

看到 `Uvicorn running on http://0.0.0.0:8000` 说明成功了。

浏览器打开前端 Dashboard：**http://localhost:18001/dashboard**（`docker-compose.yml` 默认端口映射 `18001:8000`）

---

**接入 Claude.ai（远程访问）**

需要额外配置 Cloudflare Tunnel，把服务暴露到公网。参考下面「接入 Claude.ai (远程)」章节。

**接入 Claude Desktop（本地）**

不需要 Docker，直接用 Python 本地跑。参考下面「安装 / Setup」章节。

---

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/P0luz/Ombre-Brain)
[![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/templates/OMBRE-BRAIN?referralCode=P0luz)
[![Docker Hub](https://img.shields.io/docker/v/p0luz/ombre-brain?label=Docker%20Hub&logo=docker)](https://hub.docker.com/r/p0luz/ombre-brain)

---

## 它是什么 / What is this

Claude 没有跨对话记忆。每次对话结束，之前聊过的所有东西都会消失。

Ombre Brain 给了它一套持久记忆——不是那种冷冰冰的键值存储，而是带情感坐标的、会自然衰减的、像人类记忆一样会遗忘和浮现的系统。

Claude has no cross-conversation memory. Everything from a previous chat vanishes once it ends.

Ombre Brain gives it persistent memory — not cold key-value storage, but a system with emotional coordinates, natural decay, and forgetting/surfacing mechanics that loosely mimic how human memory works.

核心特点 / Key features:

- **情感坐标打标 / Emotional tagging**: 每条记忆用 Russell 环形情感模型的 valence（效价）和 arousal（唤醒度）两个连续维度标记。不是"开心/难过"这种离散标签。
  Each memory is tagged with two continuous dimensions from Russell's circumplex model: valence and arousal. Not discrete labels like "happy/sad".

- **双通道检索 / Dual-channel search**: 关键词模糊匹配 + 向量语义相似度并联检索。关键词通道用 rapidfuzz 做模糊匹配；语义通道用独立配置的 embedding 模型计算 cosine similarity，能在"今天很累"这种没有精确关键词的查询里找到"身体不适"、"睡眠问题"等语义相关记忆。两个通道去重合并，token 预算截断。
  Keyword fuzzy matching + vector semantic similarity in parallel. Keyword channel uses rapidfuzz; semantic channel uses independently configured embeddings with cosine similarity — finds semantically related memories even without exact keyword matches (e.g. "feeling tired" → "health issues", "sleep problems"). Results are deduplicated and truncated by token budget.

- **自然遗忘 / Natural forgetting**: 改进版艾宾浩斯遗忘曲线。不活跃的记忆自动衰减归档，高情绪强度的记忆衰减更慢。
  Modified Ebbinghaus forgetting curve. Inactive memories naturally decay and archive. High-arousal memories decay slower.

- **权重池浮现 / Weight pool surfacing**: 记忆不是被动检索的，它们会主动浮现——未解决的、情绪强烈的记忆权重更高，会在对话开头自动推送。
  Memories aren't just passively retrieved — they actively surface. Unresolved, emotionally intense memories carry higher weight and get pushed at conversation start.

- **记忆重构 / Memory reconstruction**: 检索时根据当前情绪状态微调记忆的 valence 展示值（±0.1），模拟人类"此刻的心情影响对过去的回忆"的认知偏差。
  During retrieval, memory valence display is subtly shifted (±0.1) based on current mood, simulating the human cognitive bias of "current mood colors past memories".

- **Obsidian 原生 / Obsidian-native**: 每个记忆桶就是一个 Markdown 文件，YAML frontmatter 存元数据。可以直接在 Obsidian 里浏览、编辑、搜索。自动注入 `[[双链]]`。
  Each memory bucket is a Markdown file with YAML frontmatter. Browse, edit, and search directly in Obsidian. Wikilinks are auto-injected.

- **API 脱水 + 缓存 / API dehydration + cache**: 脱水压缩和自动打标通过 LLM API（DeepSeek / Gemini 等）完成，结果缓存到本地 SQLite（`dehydration_cache.db`），相同内容不重复调用 API。向量检索不可用时降级到 fuzzy matching。
  Dehydration and auto-tagging are done via LLM API (DeepSeek / Gemini etc.), with results cached locally in SQLite (`dehydration_cache.db`) to avoid redundant API calls. Embedding search degrades to fuzzy matching when unavailable.

- **历史对话导入 / Conversation history import**: 将过去与 Claude / ChatGPT / DeepSeek 等的对话批量导入为记忆桶。支持 Claude JSON 导出、ChatGPT 导出、Markdown、纯文本等格式，分块处理带断点续传，通过 Dashboard「导入」Tab 操作。
  Batch-import past conversations (Claude / ChatGPT / DeepSeek etc.) as memory buckets. Supports Claude JSON export, ChatGPT export, Markdown, and plain text. Chunked processing with resume support, via the Dashboard "Import" tab.

## 网关注入模式 / Gateway Injection Mode

除了 MCP 服务器，现在还可以单独启动一个 OpenAI 兼容网关：

```bash
python gateway.py
```

它暴露两个接口：
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

这个网关会在转发到上游模型前自动注入四段上下文：
- `Current Inner State`：人格、情绪、关系状态和回复姿态
- `Core Memory`：`pinned / protected` 桶
- `Recent Context`：最近 `72h` 内的普通记忆摘要
- `Recalled Memory`：从关键词 + embedding 候选里挑出的 `0~2` 条动态记忆

v1 已实现的召回约束：
- 同一 `session` 最近 `5` 轮注入过的桶优先跳过
- 同一桶在 `48h` 内应用冷却折扣，倍率从 `0.3` 线性恢复到 `1.0`
- 如果最近 `5` 轮过滤后一个候选都不剩，会放宽轮次过滤，只保留冷却折扣

请求要求：
- 认证头：`Authorization: Bearer <OMBRE_GATEWAY_TOKEN>`
- 会话头：`X-Ombre-Session-Id`
- 支持非流式和 OpenAI-compatible SSE 流式请求；流式响应会在上游完整结束后写入本轮注入历史
- 透传 OpenAI-compatible 工具调用字段，包括 `tools`、`tool_choice`、`parallel_tool_calls`、消息里的 `tool_calls` 和 `tool` 结果消息
- `GET /v1/models` 会返回 Gateway 聚合后的模型列表；单上游模式读 `gateway.upstream_models`，多上游模式读 `gateway.upstreams[*].models`

需要额外设置的环境变量：
- `OMBRE_GATEWAY_TOKEN`
- `OMBRE_GATEWAY_UPSTREAM_API_KEY`
- `OMBRE_GATEWAY_UPSTREAM_BASE_URL`
- `OMBRE_GATEWAY_UPSTREAM_MODEL`
- `OMBRE_GATEWAY_UPSTREAM_MODELS`（逗号分隔，用于 `/v1/models`）
- `OMBRE_PERSONA_API_KEY`（可选，缺省回退 `OMBRE_API_KEY`）

上游模型地址、默认模型和模型列表写在 `config.yaml` 的 `gateway` 段里，也可以用上面的环境变量覆盖。

多上游示例：

```yaml
gateway:
  upstream_default_model: "deepseek-chat"
  upstreams:
    - name: "deepseek"
      base_url: "https://api.deepseek.com/v1"
      api_key_env: "OMBRE_GATEWAY_DEEPSEEK_API_KEY"
      default_model: "deepseek-chat"
      models:
        - "deepseek-chat"
        - "deepseek-reasoner"
    - name: "siliconflow"
      base_url: "https://api.siliconflow.cn/v1"
      api_key_env: "OMBRE_GATEWAY_SILICONFLOW_API_KEY"
      models:
        - "Qwen/Qwen3-32B"
        - "THUDM/GLM-4-32B"
```

客户端依然只需要填写一组 Gateway 地址和 token。Gateway 会把 `/v1/models` 展平成一个统一模型列表，再按请求里的 `model` 把聊天请求路由到对应厂商。

### 人格状态引擎 / Persona State Engine

网关会维护一份独立的 `persona_state.db`：
- `persona_global_state`：长期人格和关系状态
- `persona_session_state`：按 `X-Ombre-Session-Id` 保存的短期心情
- `persona_events`：每轮事件评估、状态增量和原始响应

默认用便宜的 DeepSeek 做事件评估：
- `persona.base_url`: `https://api.deepseek.com/v1`
- `persona.model`: `deepseek-chat`

DeepSeek 每轮只看最后一条 user message，输出固定 JSON。系统会裁剪每次变化幅度：人格慢慢变，关系中速变，当前心情最快变，并按半衰期回落到默认状态。

## 边界说明 / Design boundaries

官方记忆功能已经在做身份层的事了——你是谁，你有什么偏好，你们的关系是什么。那一层交给它，Ombre Brain不打算造重复的轮子。

Ombre Brain 的边界是时间里发生的事，不是你是谁。它记住的是：你们聊过什么，经历了什么，哪些事情还悬在那里没有解决。两层配合用，才是完整的。

每次新对话，Claude 从零开始——但它能从 Ombre Brain 里找回跟你有关的一切。不是重建，是接续。

---

Official memory already handles the identity layer — who you are, what you prefer, what your relationship is. That layer belongs there. Ombre Brain isn't trying to duplicate it.

Ombre Brain's boundary is *what happened in time*, not *who you are*. It holds conversations, experiences, unresolved things. The two layers together are what make it feel complete.

Each new conversation starts fresh — but Claude can reach back through Ombre Brain and find everything that happened between you. Not a rebuild. A continuation.

## 架构 / Architecture

```
Claude ←→ MCP Protocol ←→ server.py
                              │
              ┌───────────────┼───────────────┐
              │               │               │
        bucket_manager   dehydrator     decay_engine
         (CRUD + 搜索)    (压缩 + 打标)   (遗忘曲线)
              │               │
        Obsidian Vault   embedding_engine
       (Markdown files)  (向量语义检索)
                              │
                         embeddings.db
                         (SQLite, 3072-dim)
```

### 检索架构 / Search Architecture

```
breath(query="今天很累")
         │
    ┌────┴────┐
    │         │
 Channel 1  Channel 2
 关键词匹配   向量语义
 (rapidfuzz)  (cosine similarity)
    │         │
    └────┬────┘
         │
    去重 + 合并
    token 预算截断
         │
    [语义关联] 标注 vector 来源
         │
    返回 ≤20 条结果
```

6 个 MCP 工具 / 6 MCP tools:

| 工具 Tool | 作用 Purpose |
|-----------|-------------|
| `breath` | 浮现或检索记忆。无参数=推送未解决记忆；有参数=关键词+向量语义双通道检索。支持 domain/valence/arousal 过滤 / Surface or search memories. No args = surface unresolved; with query = keyword + vector dual-channel search. Supports domain/valence/arousal filters |
| `hold` | 存储单条记忆，自动打标+合并相似桶+生成 embedding。`feel=True` 写模型自己的感受 / Store a single memory with auto-tagging, merging, and embedding. `feel=True` for model's own reflections |
| `grow` | 日记归档，自动拆分长内容为多个记忆桶，每个桶自动生成 embedding / Diary digest, auto-split into multiple buckets with embeddings |
| `trace` | 修改元数据、标记已解决、删除 / Modify metadata, mark resolved, delete |
| `pulse` | 系统状态 + 所有记忆桶列表 / System status + bucket listing |
| `dream` | 对话开头自省消化——读最近记忆，有沉淀写 feel，能放下就 resolve / Self-reflection at conversation start |

## 安装 / Setup

### 环境要求 / Requirements

- Python 3.11+
- 一个 Obsidian Vault（可选，不用也行，会在项目目录下自建 `buckets/`）
  An Obsidian vault (optional — without one, it uses a local `buckets/` directory)

### 步骤 / Steps

```bash
git clone https://github.com/P0luz/Ombre-Brain.git
cd Ombre-Brain

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

复制配置文件并按需修改 / Copy config and edit as needed:

```bash
cp config.example.yaml config.yaml
```

如果你要用 API 做脱水压缩和自动打标（推荐，效果好很多），设置环境变量：
If you want API-powered dehydration and tagging (recommended, much better quality):

```bash
export OMBRE_API_KEY="your-api-key"
```

支持任何 OpenAI 兼容 API。在 `config.yaml` 里改 `base_url` 和 `model` 就行。
Supports any OpenAI-compatible API. Just change `base_url` and `model` in `config.yaml`.

> **💡 向量化检索（Embedding）**
> Ombre Brain 内置双通道检索：关键词匹配 + 向量语义搜索。每次 `hold`/`grow` 存入记忆时自动生成 embedding 并存入 `embeddings.db`（SQLite）。
> 推荐：把 `embedding.base_url` / `embedding.model` / `OMBRE_EMBEDDING_API_KEY` 单独配置给 embedding 服务，例如硅基流动 `Qwen/Qwen3-Embedding-0.6B`。不配置 `embedding.api_key/base_url` 时会回退复用脱水 API。
> 不配置 embedding 也能用，系统会降级到纯 fuzzy matching 模式。
>
> **已有存量桶需要补生成 embedding**：运行 `backfill_embeddings.py`：
> ```bash
> OMBRE_EMBEDDING_API_KEY="your-key" python backfill_embeddings.py --batch-size 20
> ```
> Docker 用户：`docker exec -e OMBRE_BUCKETS_DIR=/data -e OMBRE_EMBEDDING_API_KEY="your-key" ombre-brain python3 backfill_embeddings.py --batch-size 20`
>
> **Embedding support**: Built-in dual-channel search: keyword + vector semantic. Embeddings are auto-generated on each `hold`/`grow` and stored in `embeddings.db` (SQLite). Configure `embedding.base_url`, `embedding.model`, and `OMBRE_EMBEDDING_API_KEY` separately when using a dedicated embedding provider. Without it, falls back to fuzzy matching. For existing buckets, run `backfill_embeddings.py`.

### 接入 Claude Desktop / Connect to Claude Desktop

在 Claude Desktop 配置文件中添加（macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`）：

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "ombre-brain": {
      "command": "python",
      "args": ["/path/to/Ombre-Brain/server.py"],
      "env": {
        "OMBRE_API_KEY": "your-api-key"
      }
    }
  }
}
```

### 接入 Claude.ai (远程) / Connect to Claude.ai (remote)

需要 HTTP 传输 + 隧道。可以用 Docker：
Requires HTTP transport + tunnel. Docker setup:

```bash
echo "OMBRE_API_KEY=your-api-key" > .env
docker-compose up -d
```

`docker-compose.yml` 里配好了 Cloudflare Tunnel。你需要自己在 `~/.cloudflared/` 下放凭证和路由配置。
The `docker-compose.yml` includes Cloudflare Tunnel. You'll need your own credentials under `~/.cloudflared/`.

### 指向 Obsidian / Point to Obsidian

在 `config.yaml` 里设置 `buckets_dir`：
Set `buckets_dir` in `config.yaml`:

```yaml
buckets_dir: "/path/to/your/Obsidian Vault/Ombre Brain"
```

不设的话，默认用项目目录下的 `buckets/`。
If not set, defaults to `buckets/` in the project directory.

## 配置 / Configuration

所有参数在 `config.yaml`（从 `config.example.yaml` 复制）。关键的几个：
All parameters in `config.yaml` (copy from `config.example.yaml`). Key ones:

| 参数 Parameter | 说明 Description | 默认 Default |
|---|---|---|
| `transport` | `stdio`（本地）/ `streamable-http`（远程）| `stdio` |
| `buckets_dir` | 记忆桶存储路径 / Bucket storage path | `./buckets/` |
| `dehydration.model` | 脱水用的 LLM 模型 / LLM model for dehydration | `deepseek-chat` |
| `dehydration.base_url` | API 地址 / API endpoint | `https://api.deepseek.com/v1` |
| `embedding.enabled` | 启用向量语义检索 / Enable embedding search | `true` |
| `embedding.model` | Embedding 模型 / Embedding model | `Qwen/Qwen3-Embedding-0.6B` |
| `embedding.base_url` | Embedding API 地址 / Embedding API endpoint | `https://api.siliconflow.cn/v1` |
| `gateway.upstream_base_url` | 网关上游 OpenAI 兼容地址 / Gateway upstream base URL | `""` |
| `gateway.upstream_default_model` | 网关默认模型 / Gateway default model | `""` |
| `gateway.upstreams` | 多上游路由配置 / Multi-upstream routing config | `[]` |
| `gateway.skip_recent_rounds` | 最近几轮跳过注入 / Skip recently injected rounds | `5` |
| `gateway.cooldown_hours` | 召回冷却时长 / Recall cooldown hours | `48` |
| `persona.enabled` | 启用人格状态注入 / Enable persona state injection | `true` |
| `persona.model` | 人格事件评估模型 / Persona event evaluator model | `deepseek-chat` |
| `persona.session_mood_half_life_minutes` | 短期心情半衰期 / Session mood half-life | `90` |
| `decay.lambda` | 衰减速率，越大越快忘 / Decay rate | `0.05` |
| `decay.threshold` | 归档阈值 / Archive threshold | `0.3` |
| `merge_threshold` | 合并相似度阈值 (0-100) / Merge similarity | `75` |

敏感配置用环境变量：
Sensitive config via env vars:
- `OMBRE_API_KEY` — LLM API 密钥
- `OMBRE_TRANSPORT` — 覆盖传输方式
- `OMBRE_BUCKETS_DIR` — 覆盖存储路径
- `OMBRE_CONFIG_PATH` — 指向自定义 `config.yaml`
- `OMBRE_EMBEDDING_API_KEY` — 独立 embedding API key，缺省回退 `OMBRE_API_KEY`
- `OMBRE_EMBEDDING_BASE_URL` — 覆盖 embedding API 地址
- `OMBRE_EMBEDDING_MODEL` — 覆盖 embedding 模型
- `OMBRE_EMBEDDING_ENABLED` — 覆盖是否启用 embedding
- `OMBRE_GATEWAY_HOST` — 覆盖网关监听地址
- `OMBRE_GATEWAY_PORT` — 覆盖网关监听端口
- `OMBRE_GATEWAY_UPSTREAM_BASE_URL` — 覆盖网关上游地址
- `OMBRE_GATEWAY_UPSTREAM_MODEL` — 覆盖网关默认模型
- `OMBRE_GATEWAY_UPSTREAM_MODELS` — 覆盖单上游模型列表
- `OMBRE_GATEWAY_TOKEN` — 网关鉴权 token
- `OMBRE_GATEWAY_UPSTREAM_API_KEY` — 网关转发上游时使用的 API key
- `gateway.upstreams[*].api_key_env` — 每个多上游节点可单独引用自己的环境变量
- `OMBRE_PERSONA_API_KEY` — 人格事件评估 API key，缺省回退 `OMBRE_API_KEY`
- `OMBRE_PERSONA_BASE_URL` — 覆盖人格事件评估 API 地址
- `OMBRE_PERSONA_MODEL` — 覆盖人格事件评估模型

## 衰减公式 / Decay Formula

$$final\_score = Importance \times activation\_count^{0.3} \times e^{-\lambda \times days} \times combined\_weight \times resolved\_factor \times urgency\_boost$$

### 短期/长期权重分离 / Short-term vs Long-term Weight Separation

系统对记忆的权重计算采用**分段策略**，模拟人类记忆的时效特征：
The system uses a **segmented weighting strategy** that mimics how human memory prioritizes:

| 阶段 Phase | 时间范围 | 权重分配 | 直觉解释 |
|---|---|---|---|
| 短期 Short-term | ≤ 3 天 | 时间 70% + 情感 30% | 刚发生的事，鲜活度最重要 |
| 长期 Long-term | > 3 天 | 情感 70% + 时间 30% | 时间淡了，情感强度决定能记多久 |

$$combined\_weight = \begin{cases} time\_weight \times 0.7 + emotion\_weight \times 0.3 & \text{if } days \leq 3 \\ emotion\_weight \times 0.7 + time\_weight \times 0.3 & \text{if } days > 3 \end{cases}$$

### 时间系数（新鲜度加成）/ Time Weight (Freshness Bonus)

连续指数衰减，无跳变：
Continuous exponential decay, no discontinuities:

$$freshness = 1.0 + 1.0 \times e^{-t/36}$$

| 距存入时间 Time since creation | 新鲜度乘数 Multiplier |
|---|---|
| 刚存入 (t=0) | ×2.0 |
| 约 25 小时 | ×1.5 |
| 约 50 小时 | ×1.25 |
| 72 小时 (3天) | ×1.14 |
| 1 周+ | ≈ ×1.0 |

t 为小时，36 为衰减常数。老记忆不被惩罚（下限 ×1.0），新记忆获得额外加成。

### 情感权重 / Emotion Weight

$$emotion\_weight = base + arousal \times arousal\_boost$$

- 默认 `base=1.0`, `arousal_boost=0.8`
- arousal=0.3（平静）→ 1.24；arousal=0.9（激动）→ 1.72

### 权重池修正因子 / Weight Pool Modifiers

| 状态 State | 修正因子 Factor | 说明 |
|---|---|---|
| 未解决 Unresolved | ×1.0 | 正常权重 |
| 已解决 Resolved | ×0.05 | 沉底，等关键词唤醒 |
| 已解决+已消化 Resolved+Digested | ×0.02 | 加速淡化，归档为无限小 |
| 高唤醒+未解决 Urgent | ×1.5 | arousal>0.7 的未解决记忆额外加权 |
| 钉选 Pinned | 999.0 | 不衰减、不合并、importance=10 |
| Feel | 50.0 | 固定分数，不参与衰减 |

### 参数说明 / Parameters

- `importance`: 1-10，记忆重要性 / memory importance
- `activation_count`: 被检索的次数，越常被想起衰减越慢 / retrieval count; more recalls = slower decay
- `days`: 距上次激活的天数 / days since last activation
- `arousal`: 唤醒度，越强烈的记忆越难忘 / arousal; intense memories are harder to forget
- `λ` (decay_lambda): 衰减速率，默认 0.05 / decay rate, default 0.05

## Dreaming 与 Feel / Dreaming & Feel

### Dreaming — 做梦
每次新对话开始时，Claude 会自动执行 `dream()`——读取最近的记忆桶，用第一人称思考：哪些事还有重量？哪些可以放下了？

At the start of each conversation, Claude runs `dream()` — reads recent memory buckets and reflects in first person: what still carries weight? What can be let go?

- 值得放下的 → `trace(resolved=1)` 让它沉底
- 有沉淀的 → 写 `feel`，记录模型自己的感受
- 没有沉淀就不写，不强迫产出

### Feel — 带走的东西
Feel 不是事件记录，是**模型带走的东西**——一句感受、一个未解答的问题、一个观察到的变化。

Feel is not an event log — it's **what the model carries away**: a feeling, an unanswered question, a noticed change.

- `hold(content="...", feel=True, source_bucket="源记忆ID", valence=模型自己的感受)`
- `valence` 是模型的感受，不是事件情绪。同一段争吵，事件 V0.2，但模型可能 V0.4（「我从中看到了成长」）
- `source_bucket` 指向被消化的记忆，会被标记为「已消化」→ 加速淡化到无限小，但不会被删除
- Feel 不参与普通浮现、不衰减、不参与 dreaming
- 用 `breath(domain="feel")` 读取之前的 feel

### 对话启动完整流程 / Conversation Start Sequence
```
1. breath()              — 睁眼，看有什么浮上来
2. dream()               — 消化最近记忆，有沉淀写 feel
3. breath(domain="feel") — 读之前的 feel
4. 开始和用户说话
```

## 给 Claude 的使用指南 / Usage Guide for Claude

`CLAUDE_PROMPT.md` 是写给 Claude 看的使用说明。放到你的 system prompt 或 custom instructions 里就行。

`CLAUDE_PROMPT.md` is the usage guide written for Claude. Put it in your system prompt or custom instructions.

## 工具脚本 / Utility Scripts

| 脚本 Script | 用途 Purpose |
|---|---|
| `embedding_engine.py` | 向量化引擎，管理 embedding 的生成、存储、相似度搜索 / Embedding engine: generate, store, and search embeddings |
| `backfill_embeddings.py` | 为存量桶批量生成 embedding / Batch-generate embeddings for existing buckets |
| `write_memory.py` | 手动写入记忆，绕过 MCP / Manually write memories, bypass MCP |
| `migrate_to_domains.py` | 迁移平铺文件到域子目录 / Migrate flat files to domain subdirs |
| `reclassify_domains.py` | 基于关键词重分类 / Reclassify by keywords |
| `reclassify_api.py` | 用 API 重打标未分类桶 / Re-tag uncategorized buckets via API |
| `test_tools.py` | MCP 工具集成测试（8 项） / MCP tool integration tests (8 tests) |
| `test_smoke.py` | 冒烟测试 / Smoke test |

## 部署 / Deploy

### Docker Hub 预构建镜像

[![Docker Hub](https://img.shields.io/docker/v/p0luz/ombre-brain?label=Docker%20Hub&logo=docker)](https://hub.docker.com/r/p0luz/ombre-brain)

不用 clone 代码、不用 build，直接拉取预构建镜像：

```bash
docker pull p0luz/ombre-brain:latest
curl -O https://raw.githubusercontent.com/P0luz/Ombre-Brain/main/docker-compose.user.yml
echo "OMBRE_API_KEY=你的key" > .env
docker compose -f docker-compose.user.yml up -d
```

验证：`curl http://localhost:8000/health`
Dashboard：浏览器打开 `http://localhost:8000/dashboard`

### Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/P0luz/Ombre-Brain)

> ⚠️ **免费层不可用**：Render 免费层**不支持持久化磁盘**，服务重启后记忆数据会丢失，且会在无流量时休眠。**必须使用 Starter（$7/mo）或以上**才能正常使用。
> **Free tier won't work**: Render free tier has **no persistent disk** — all memory data is lost on restart. It also sleeps on inactivity. **Starter plan ($7/mo) or above is required.**

项目根目录已包含 `render.yaml`，点击按钮后：
1. （可选）设置 `OMBRE_API_KEY`：任何 OpenAI 兼容 API 的 key，不填则自动降级为本地关键词提取
2. （可选）设置 `OMBRE_BASE_URL`：API 地址，支持任意 OpenAI 化地址，如 `https://api.deepseek.com/v1` / `http://123.1.1.1:7689/v1` / `http://your-ollama:11434/v1`
3. Render 自动挂载持久化磁盘到 `/opt/render/project/src/buckets`
4. Dashboard：`https://<你的服务名>.onrender.com/dashboard`
5. 部署后 MCP URL：`https://<你的服务名>.onrender.com/mcp`

`render.yaml` is included. After clicking the button:
1. (Optional) `OMBRE_API_KEY`: any OpenAI-compatible key; omit to fall back to local keyword extraction
2. (Optional) `OMBRE_BASE_URL`: any OpenAI-compatible endpoint, e.g. `https://api.deepseek.com/v1`, `http://123.1.1.1:7689/v1`, `http://your-ollama:11434/v1`
3. Persistent disk auto-mounts at `/opt/render/project/src/buckets`
4. Dashboard: `https://<your-service>.onrender.com/dashboard`
5. MCP URL after deploy: `https://<your-service>.onrender.com/mcp`

### Zeabur

> 💡 **Zeabur 的定价模式**：Zeabur 是「买 VPS + 平台托管」，你先购买一台服务器（最低腾讯云新加坡 $2/mo、火山引擎 $3/mo），Volume 直接挂在该服务器上，**数据天然持久化，无丢失问题**。另需订阅 Zeabur 管理方案（Developer $5/mo），总计约 $7-8/mo 起。
> **Zeabur pricing model**: You buy a VPS first (cheapest: Tencent Cloud Singapore ~$2/mo, Volcano Engine ~$3/mo), then add Zeabur's Developer plan ($5/mo) for management. Volumes mount directly on your server — **data is always persistent, no cold-start data loss**. Total ~$7-8/mo minimum.

**步骤 / Steps：**

1. **创建项目 / Create project**
   - 打开 [zeabur.com](https://zeabur.com) → 购买一台服务器 → **New Project** → **Deploy from GitHub**
   - 先 Fork 本仓库到自己 GitHub 账号，然后在 Zeabur 选择 `你的用户名/Ombre-Brain`
   - Zeabur 会自动检测到根目录的 `Dockerfile` 并使用 Docker 方式构建
   - Go to [zeabur.com](https://zeabur.com) → buy a server → **New Project** → **Deploy from GitHub**
   - Fork this repo first, then select `your-username/Ombre-Brain` in Zeabur
   - Zeabur auto-detects the `Dockerfile` in root and builds via Docker

2. **设置环境变量 / Set environment variables**（服务页面 → **Variables** 标签页）
   - `OMBRE_API_KEY`（可选）— LLM API 密钥，不填则自动降级为本地关键词提取
   - `OMBRE_BASE_URL`（可选）— API 地址，如 `https://api.deepseek.com/v1`

   > ⚠️ **不需要**手动设置 `OMBRE_TRANSPORT` 和 `OMBRE_BUCKETS_DIR`，Dockerfile 里已经设好了默认值。Zeabur 对单阶段 Dockerfile 会自动注入控制台设置的环境变量。
   > You do **NOT** need to set `OMBRE_TRANSPORT` or `OMBRE_BUCKETS_DIR` — defaults are baked into the Dockerfile. Zeabur auto-injects dashboard env vars for single-stage Dockerfiles.

3. **挂载持久存储 / Mount persistent volume**（服务页面 → **Volumes** 标签页）
   - Volume ID：填 `ombre-buckets`（或任意名）
   - 挂载路径 / Path：**`/app/buckets`**
   - ⚠️ 不挂载的话，每次重新部署记忆数据会丢失
   - ⚠️ Without this, memory data is lost on every redeploy

4. **配置端口 / Configure port**（服务页面 → **Networking** 标签页）
   - Port Name：`web`（或任意名）
   - Port：**`8000`**
   - Port Type：**`HTTP`**
   - 然后点 **Generate Domain** 生成一个 `xxx.zeabur.app` 域名
   - Then click **Generate Domain** to get a `xxx.zeabur.app` domain

5. **验证 / Verify**
   - 访问 `https://<你的域名>.zeabur.app/health`，应返回 JSON
   - Visit `https://<your-domain>.zeabur.app/health` — should return JSON
   - Dashboard：`https://<你的域名>.zeabur.app/dashboard`
   - 最终 MCP 地址 / MCP URL：`https://<你的域名>.zeabur.app/mcp`

**常见问题 / Troubleshooting：**

| 现象 Symptom | 原因 Cause | 解决 Fix |
|---|---|---|
| 域名无法访问 / Domain unreachable | 没配端口 / Port not configured | Networking 标签页加 port 8000 (HTTP) |
| 域名无法访问 / Domain unreachable | `OMBRE_TRANSPORT` 未设置，服务以 stdio 模式启动，不监听任何端口 / Service started in stdio mode — no port is listened | **Variables 标签页确认设置 `OMBRE_TRANSPORT=streamable-http`，然后重新部署** |
| 构建失败 / Build failed | Dockerfile 未被识别 / Dockerfile not detected | 确认仓库根目录有 `Dockerfile`（大小写敏感） |
| 服务启动后立刻退出 | `OMBRE_TRANSPORT` 被覆盖为 `stdio` | 检查 Variables 里有没有多余的 `OMBRE_TRANSPORT=stdio`，删掉即可 |
| 重启后记忆丢失 / Data lost on restart | Volume 未挂载 | Volumes 标签页挂载到 `/app/buckets` |

### 使用 Cloudflare Tunnel 或 ngrok 连接 / Connecting via Cloudflare Tunnel or ngrok

> ℹ️ 自 v1.1 起，server.py 在 HTTP 模式下已自动添加 CORS 中间件，无需额外配置。
> Since v1.1, server.py automatically enables CORS middleware in HTTP mode — no extra config needed.

使用隧道连接时，确保以下条件满足：
When connecting via tunnel, ensure:

1. **服务器必须运行在 HTTP 模式** / Server must use HTTP transport
   ```bash
   OMBRE_TRANSPORT=streamable-http python server.py
   ```
   或 Docker：
   ```bash
   docker-compose up -d
   ```

2. **在 Claude.ai 网页版添加 MCP 服务器** / Adding to Claude.ai web
   - URL 格式 / URL format: `https://<tunnel-subdomain>.trycloudflare.com/mcp`
   - 或 ngrok / or ngrok: `https://<xxxx>.ngrok-free.app/mcp`
   - 先访问 `/health` 验证连接 / Verify first: `https://<your-tunnel>/health` should return `{"status":"ok",...}`

3. **已知限制 / Known limitations**
   - Cloudflare Tunnel 免费版有空闲超时（约 10 分钟），系统内置保活 ping 可缓解但不能完全消除
   - Free Cloudflare Tunnel has idle timeout (~10 min); built-in keepalive pings mitigate but can't fully prevent it
   - ngrok 免费版有请求速率限制 / ngrok free tier has rate limits
   - 如果连接仍失败，检查隧道是否正在运行、服务是否以 `streamable-http` 模式启动
   - If connection still fails, verify the tunnel is running and the server started in `streamable-http` mode

| 现象 Symptom | 原因 Cause | 解决 Fix |
|---|---|---|
| 网页版无法连接隧道 URL / Web can't connect to tunnel URL | 服务以 stdio 模式运行 / Server in stdio mode | 设置 `OMBRE_TRANSPORT=streamable-http` 后重启 |
| 网页版无法连接隧道 URL / Web can't connect to tunnel URL | 旧版 server.py 缺少 CORS 头 / Missing CORS headers | 拉取最新代码，CORS 已内置 / Pull latest — CORS is now built-in |
| `/health` 返回 200 但 MCP 连不上 / `/health` 200 but MCP fails | 路径错误 / Wrong path | MCP URL 末尾必须是 `/mcp` 而非 `/` |
| 隧道连接偶尔断开 / Tunnel disconnects intermittently | Cloudflare Tunnel 空闲超时 / Idle timeout | 保活 ping 已内置，若仍断开可缩短隧道超时配置 |

---

### Session Start Hook（自动 breath）

部署后，如果你使用 Claude Code，可以在项目内激活自动浮现 hook：
`.claude/settings.json` 已配置好 `SessionStart` hook，每次新会话或恢复会话时自动触发 `breath`，把最高权重未解决记忆推入上下文。

**仅在远程 HTTP 模式下有效**（`OMBRE_TRANSPORT=streamable-http`）。本地 stdio 模式下 hook 会安静退出，不影响正常使用。

可以通过 `OMBRE_HOOK_URL` 环境变量指定服务器地址（默认 `http://localhost:8000`），或者设置 `OMBRE_HOOK_SKIP=1` 临时禁用。

If using Claude Code, `.claude/settings.json` configures a `SessionStart` hook that auto-calls `breath` on each new or resumed session, surfacing your highest-weight unresolved memories as context. Only active in remote HTTP mode. Set `OMBRE_HOOK_SKIP=1` to disable temporarily.

## 更新 / How to Update

不同部署方式的更新方法。

Different update procedures depending on your deployment method.

### Docker Hub 预构建镜像用户 / Docker Hub Pre-built Image

```bash
# 拉取最新镜像
docker pull p0luz/ombre-brain:latest

# 重启容器（记忆数据在 volume 里，不会丢失）
docker compose -f docker-compose.user.yml down
docker compose -f docker-compose.user.yml up -d
```

> 你的记忆数据挂载在 `./buckets:/data`，pull + restart 不会影响已有数据。
> Your memory data is mounted at `./buckets:/data` — pull + restart won't affect existing data.

### 从源码部署用户 / Source Code Deploy (Docker)

```bash
cd Ombre-Brain

# 拉取最新代码
git pull origin main

# 重新构建并重启
docker compose down
docker compose build
docker compose up -d
```

> `docker compose build` 会重新构建镜像。volume 挂载的记忆数据不受影响。
> `docker compose build` rebuilds the image. Volume-mounted memory data is unaffected.

### 本地 Python 用户 / Local Python (no Docker)

```bash
cd Ombre-Brain

# 拉取最新代码
git pull origin main

# 更新依赖（如有新增）
pip install -r requirements.txt

# 重启服务
# Ctrl+C 停止旧进程，然后：
python server.py
```

### Render

Render 连接了你的 GitHub 仓库，**自动部署**：

1. 如果你 Fork 了仓库 → 在 GitHub 上同步上游更新（Sync fork），Render 会自动重新部署
2. 或者手动：Render Dashboard → 你的服务 → **Manual Deploy** → **Deploy latest commit**

> 持久化磁盘（`/opt/render/project/src/buckets`）上的记忆数据在重新部署时保留。
> Persistent disk data at `/opt/render/project/src/buckets` is preserved across deploys.

### Zeabur

Zeabur 也连接了你的 GitHub 仓库：

1. 在 GitHub 上同步 Fork 的最新代码 → Zeabur 自动触发重新构建部署
2. 或者手动：Zeabur Dashboard → 你的服务 → **Redeploy**

> Volume 挂载在 `/app/buckets`，重新部署时数据保留。
> Volume mounted at `/app/buckets` — data persists across redeploys.

### VPS / 自有服务器 / Self-hosted VPS

```bash
cd Ombre-Brain

# 拉取最新代码
git pull origin main

# 方式 A：Docker 部署
docker compose down
docker compose build
docker compose up -d

# 方式 B：直接 Python 运行
pip install -r requirements.txt
# 重启你的进程管理器（systemd / supervisord / pm2 等）
sudo systemctl restart ombre-brain   # 示例
```

> **通用注意事项 / General notes:**
> - 更新不会影响你的记忆数据（存在 volume 或 buckets 目录里）
> - 如果 `requirements.txt` 有变化，Docker 用户重新 build 即可自动处理；非 Docker 用户需手动 `pip install -r requirements.txt`
> - 更新后访问 `/health` 验证服务正常
> - Updates never affect your memory data (stored in volumes or buckets directory)
> - If `requirements.txt` changed, Docker rebuild handles it automatically; non-Docker users need `pip install -r requirements.txt`
> - After updating, visit `/health` to verify the service is running

## License

MIT
