#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ops_common.sh"

cd "$(ombre_repo_root)"

LOCAL_COMPOSE_FILE="compose.local.yml"
DEPLOY_TARGET="vps"
DEPLOY_LABEL="VPS 部署"
DEFAULT_BRAIN_PORT="18001"
DEFAULT_GATEWAY_PORT="18002"
CLIENT_HOST="127.0.0.1"
FEATURE_SCOPE="full"
FEATURE_LABEL="部署全部"

line() {
  printf '%s\n' '------------------------------------------------------------'
}

pause() {
  printf '\n按 Enter 返回菜单...'
  read -r _ || true
}

backup_file() {
  local path="$1"
  [[ -f "${path}" ]] || return 0
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  cp "${path}" "${path}.bak.${stamp}"
  printf '已备份 %s -> %s.bak.%s\n' "${path}" "${path}" "${stamp}"
}

resolve_dir() {
  local path="$1"
  (cd "${path}" 2>/dev/null && pwd -P)
}

path_depth() {
  local path="${1#/}"
  local count=0
  local IFS='/'
  read -r -a parts <<< "${path}"
  for part in "${parts[@]}"; do
    [[ -n "${part}" ]] && ((count += 1))
  done
  printf '%s\n' "${count}"
}

is_too_broad_source_dir() {
  local path="$1"
  local depth
  [[ "${path}" == "/" ]] && return 0
  depth="$(path_depth "${path}")"
  (( depth < 2 ))
}

prompt_text() {
  local label="$1"
  local default="$2"
  local value
  if [[ -n "${default}" ]]; then
    read -r -p "${label} [${default}]: " value
    printf '%s\n' "${value:-${default}}"
  else
    read -r -p "${label}: " value
    printf '%s\n' "${value}"
  fi
}

prompt_yes_no() {
  local label="$1"
  local default="$2"
  local value
  local suffix="[y/N]"
  [[ "${default}" == "y" ]] && suffix="[Y/n]"
  while true; do
    read -r -p "${label} ${suffix}: " value
    value="${value:-${default}}"
    case "${value}" in
      y|Y|yes|YES) return 0 ;;
      n|N|no|NO) return 1 ;;
      *) printf '请输入 y 或 n。\n' ;;
    esac
  done
}

prompt_secret() {
  local label="$1"
  local required="${2:-false}"
  local value
  while true; do
    read -r -s -p "${label}: " value
    printf '\n' >&2
    if [[ -n "${value}" || "${required}" != "true" ]]; then
      printf '%s\n' "${value}"
      return 0
    fi
    printf '这个值必填。\n'
  done
}

random_token() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 24
  elif command -v uuidgen >/dev/null 2>&1; then
    uuidgen | tr -d '-'
  else
    printf 'ombre-%s-%s\n' "$(date +%s)" "$RANDOM"
  fi
}

yaml_quote() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "${value}"
}

env_line() {
  local key="$1"
  local value="$2"
  printf '%s=%s\n' "${key}" "${value}"
}

env_file_has_value() {
  local key="$1"
  if [[ -n "${!key:-}" ]]; then
    return 0
  fi
  [[ -f ".env" ]] || return 1
  awk -v key="${key}" '
    BEGIN { found = 1 }
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*$/ { next }
    {
      line = $0
      sub(/^[[:space:]]*export[[:space:]]+/, "", line)
      split(line, pair, "=")
      name = pair[1]
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
      if (name != key) next
      value = substr(line, index(line, "=") + 1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
      gsub(/^["'\'']|["'\'']$/, "", value)
      if (value != "" && value !~ /^(your-key|changeme|replace-me|xxx)$/) {
        found = 0
      }
    }
    END { exit found }
  ' ".env"
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

slugify() {
  local value="$1"
  value="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')"
  value="$(printf '%s' "${value}" | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')"
  printf '%s' "${value:-provider}"
}

env_prefix_for_provider() {
  local value="$1"
  value="$(printf '%s' "${value}" | tr '[:lower:]' '[:upper:]')"
  value="$(printf '%s' "${value}" | sed -E 's/[^A-Z0-9]+/_/g; s/^_+//; s/_+$//')"
  printf 'OMBRE_GATEWAY_%s' "${value:-PROVIDER}"
}

declare -a GW_NAMES=()
declare -a GW_SLUGS=()
declare -a GW_BASE_URLS=()
declare -a GW_DEFAULT_MODELS=()
declare -a GW_MODELS=()
declare -a GW_KEY_ENVS=()
GATEWAY_ENV_LINES=""
GATEWAY_UPSTREAMS_YAML=""

append_gateway_env() {
  local key="$1"
  local value="$2"
  GATEWAY_ENV_LINES+="${key}=${value}"$'\n'
}

add_gateway_provider_interactive() {
  local index="$1"
  local default_name="$2"
  local default_base_url="$3"
  local default_model="$4"
  local dehy_key="$5"

  line
  printf 'Gateway 上游 #%s\n' "${index}"

  local name slug base_url models_raw default_model_value prefix key_envs key_value key_count
  name="$(prompt_text 'Provider 名称（用于同名模型自动别名）' "${default_name}")"
  slug="$(slugify "${name}")"
  base_url="$(prompt_text 'Provider base_url' "${default_base_url}")"
  models_raw="$(prompt_text '模型列表（多个用英文逗号分隔）' "${default_model}")"
  default_model_value="$(prompt_text '默认模型' "$(trim "${models_raw%%,*}")")"

  prefix="$(env_prefix_for_provider "${name}")"
  key_envs=""
  if prompt_yes_no '这个 Provider 要配置多 key 吗' 'n'; then
    key_count="$(prompt_text 'key 数量' '2')"
    if ! [[ "${key_count}" =~ ^[0-9]+$ ]] || (( key_count < 1 )); then
      key_count=2
    fi
    local key_index
    for ((key_index = 1; key_index <= key_count; key_index++)); do
      local env_name="${prefix}_API_KEY_${key_index}"
      key_value="$(prompt_secret "${name} 第 ${key_index} 个 key（${env_name}）" true)"
      append_gateway_env "${env_name}" "${key_value}"
      key_envs+="${env_name},"
    done
  else
    local env_name="${prefix}_API_KEY"
    if [[ "${index}" == "1" ]] && prompt_yes_no '这个 Provider 的 key 复用脱水 key 吗' 'y'; then
      key_value="${dehy_key}"
    else
      key_value="$(prompt_secret "${name} key（${env_name}）" true)"
    fi
    append_gateway_env "${env_name}" "${key_value}"
    key_envs="${env_name},"
  fi
  key_envs="${key_envs%,}"

  GW_NAMES+=("${name}")
  GW_SLUGS+=("${slug}")
  GW_BASE_URLS+=("${base_url}")
  GW_DEFAULT_MODELS+=("${default_model_value}")
  GW_MODELS+=("${models_raw}")
  GW_KEY_ENVS+=("${key_envs}")
}

build_gateway_upstreams_yaml() {
  local yaml=$'  upstreams:\n'
  local duplicate_count=0
  declare -A model_counts=()

  for models_raw in "${GW_MODELS[@]}"; do
    IFS=',' read -r -a models <<< "${models_raw}"
    for raw_model in "${models[@]}"; do
      local model
      model="$(trim "${raw_model}")"
      [[ -z "${model}" ]] && continue
      model_counts["${model}"]=$(( ${model_counts["${model}"]:-0} + 1 ))
    done
  done

  for ((idx = 0; idx < ${#GW_NAMES[@]}; idx++)); do
    yaml+="    - name: $(yaml_quote "${GW_NAMES[$idx]}")"$'\n'
    yaml+="      base_url: $(yaml_quote "${GW_BASE_URLS[$idx]}")"$'\n'
    IFS=',' read -r -a key_envs <<< "${GW_KEY_ENVS[$idx]}"
    if (( ${#key_envs[@]} > 1 )); then
      yaml+="      api_key_envs:"$'\n'
      for env_name in "${key_envs[@]}"; do
        env_name="$(trim "${env_name}")"
        [[ -z "${env_name}" ]] && continue
        yaml+="        - $(yaml_quote "${env_name}")"$'\n'
      done
    else
      yaml+="      api_key_env: $(yaml_quote "${key_envs[0]}")"$'\n'
    fi
    yaml+="      default_model: $(yaml_quote "${GW_DEFAULT_MODELS[$idx]}")"$'\n'
    yaml+="      prompt_cache: \"\""$'\n'
    yaml+="      models:"$'\n'
    IFS=',' read -r -a models <<< "${GW_MODELS[$idx]}"
    for raw_model in "${models[@]}"; do
      local model alias
      model="$(trim "${raw_model}")"
      [[ -z "${model}" ]] && continue
      if (( ${model_counts["${model}"]:-0} > 1 )); then
        duplicate_count=$((duplicate_count + 1))
        alias="${GW_SLUGS[$idx]}/${model}"
        yaml+="        - id: $(yaml_quote "${alias}")"$'\n'
        yaml+="          upstream_model: $(yaml_quote "${model}")"$'\n'
      else
        yaml+="        - $(yaml_quote "${model}")"$'\n'
      fi
    done
  done

  if (( duplicate_count > 0 )); then
    printf '检测到同名模型，已自动写成 provider/模型名 的 Gateway alias。\n' >&2
  fi
  printf '%s' "${yaml}"
}

configure_gateway_upstreams() {
  local dehy_base_url="$1"
  local dehy_model="$2"
  local dehy_key="$3"
  local choice count

  GW_NAMES=()
  GW_SLUGS=()
  GW_BASE_URLS=()
  GW_DEFAULT_MODELS=()
  GW_MODELS=()
  GW_KEY_ENVS=()
  GATEWAY_ENV_LINES=""

  line
  printf 'Gateway 模型和 key 配置\n'
  printf '1. 单上游：复用脱水模型站点\n'
  printf '2. 单上游：自定义站点\n'
  printf '3. 多上游：多个 provider，可分别配置多 key\n'
  read -r -p '输入（1-3）：' choice
  case "${choice}" in
    1)
      add_gateway_provider_interactive 1 "deepseek" "${dehy_base_url}" "${dehy_model}" "${dehy_key}"
      ;;
    2)
      add_gateway_provider_interactive 1 "provider-a" "${dehy_base_url}" "${dehy_model}" "${dehy_key}"
      ;;
    3)
      count="$(prompt_text 'Provider 数量' '2')"
      if ! [[ "${count}" =~ ^[0-9]+$ ]] || (( count < 1 )); then
        count=2
      fi
      local provider_index
      for ((provider_index = 1; provider_index <= count; provider_index++)); do
        add_gateway_provider_interactive "${provider_index}" "provider-${provider_index}" "${dehy_base_url}" "${dehy_model}" "${dehy_key}"
      done
      ;;
    *)
      printf '未识别输入，按单上游复用脱水模型站点处理。\n'
      add_gateway_provider_interactive 1 "deepseek" "${dehy_base_url}" "${dehy_model}" "${dehy_key}"
      ;;
  esac

  GATEWAY_UPSTREAMS_YAML="$(build_gateway_upstreams_yaml)"
}

choose_feature_scope() {
  local choice
  line
  printf '选择功能范围\n'
  printf '1. 只用 Ombre MCP 部分（MCP 工具 + Dashboard + 画像/Darkroom，不启动 Gateway）\n'
  printf '2. 部署全部（MCP + Dashboard + Gateway 自动注入）\n'
  read -r -p '输入（1-2）：' choice
  case "${choice}" in
    1)
      FEATURE_SCOPE="mcp"
      FEATURE_LABEL="只用 Ombre MCP 部分（不启动 Gateway）"
      ;;
    *)
      FEATURE_SCOPE="full"
      FEATURE_LABEL="部署全部"
      ;;
  esac
}

write_env_file() {
  local dehy_key="$1"
  local embedding_key="$2"
  local gateway_token="$3"
  local dream_key="$4"
  local persona_key="$5"
  local reflection_key="$6"
  local reranker_key="$7"

  backup_file ".env"
  {
    printf '# Generated by scripts/one_click.sh\n'
    env_line "OMBRE_API_KEY" "${dehy_key}"
    env_line "OMBRE_EMBEDDING_API_KEY" "${embedding_key}"
    env_line "OMBRE_RERANKER_API_KEY" "${reranker_key}"
    env_line "OMBRE_GATEWAY_TOKEN" "${gateway_token}"
    env_line "OMBRE_PERSONA_API_KEY" "${persona_key}"
    env_line "OMBRE_REFLECTION_API_KEY" "${reflection_key}"
    env_line "OMBRE_DREAM_API_KEY" "${dream_key}"
    if [[ -n "${GATEWAY_ENV_LINES}" ]]; then
      printf '%s' "${GATEWAY_ENV_LINES}"
    fi
  } > ".env"
  printf '已写入 .env（不会显示 key）\n'
}

write_config_file() {
  local ai_name="$1"
  local user_name="$2"
  local user_display_name="$3"
  local dehy_base_url="$4"
  local dehy_model="$5"
  local embedding_enabled="$6"
  local embedding_base_url="$7"
  local embedding_model="$8"
  local reranker_enabled="$9"
  local reranker_base_url="${10}"
  local reranker_model="${11}"
  local gateway_upstreams_yaml="${12}"
  local dream_enabled="${13}"
  local dream_base_url="${14}"
  local dream_model="${15}"
  local dream_probability="${16}"
  local brain_port="${17}"
  local gateway_port="${18}"

  backup_file "config.yaml"
  cat > "config.yaml" <<EOF
transport: "streamable-http"
log_level: "INFO"

identity:
  ai_name: $(yaml_quote "${ai_name}")
  user_name: $(yaml_quote "${user_name}")
  user_display_name: $(yaml_quote "${user_display_name}")
  user_aliases:
    - "宝宝"
    - "老婆"
    - "亲爱的"
    - "她"

self_anchor:
  entry_bucket_id: ""

dehydration:
  model: $(yaml_quote "${dehy_model}")
  base_url: $(yaml_quote "${dehy_base_url}")
  thinking_mode: ""
  max_tokens: 1024
  temperature: 0.1

embedding:
  enabled: ${embedding_enabled}
  model: $(yaml_quote "${embedding_model}")
  base_url: $(yaml_quote "${embedding_base_url}")
  max_chars: 6000
  query_instruction: "Given a memory search query, retrieve relevant long-term memory passages."
  document_instruction: ""

reranker:
  enabled: ${reranker_enabled}
  model: $(yaml_quote "${reranker_model}")
  base_url: $(yaml_quote "${reranker_base_url}")
  candidate_limit: 20
  score_weight: 0.65
  timeout_seconds: 12

memory_write_gate:
  enabled: true
  auto_sources:
    - "operit"
    - "workflow"
    - "worker"
    - "auto"
  pending_threshold: 0.42
  grow_threshold: 0.72
  duplicate_similarity: 0.88
  repeat_similarity: 0.82
  repeat_promote_count: 2
  candidate_log: "memory_write_candidates.jsonl"
  max_recent_candidates: 120

recall:
  query_resurface_enabled: false

word_map:
  enabled: false
  max_terms_per_bucket: 16
  edge_top_k: 10
  min_term_len: 2
  stopwords: []
  private_terms: []
  stopword_prefixes: []
  overview_stopwords: []
  overview_stopword_prefixes: []
  overview_aliases: {}
  overview_priority_terms: []
  overview_hub_terms: []
  weak_hint_terms: []
  weak_hint_weight: 0.25

raw_events:
  db_path: ""
  max_ingest_batch: 1000

identity_semantics:
  enabled: false
  private_config_path: ""
  min_confidence: 0.78
  evidence_tags:
    - "profile_fact"
    - "ai_favorite"
    - "favorite_memory"

moment_annotations:
  enabled: true
  max_summary_chars: 160
  max_evidence_spans: 3
  max_evidence_chars: 120

gateway:
  host: "0.0.0.0"
  port: 8010
  default_session_id: "main"
  head_recent_hours: 72
  recent_context_reentry_idle_hours: 24
  recent_context_cooldown_hours: 6
  dynamic_top_k: 10
  inject_max_cards: 2
  skip_recent_rounds: 5
  cooldown_hours: 6
  cooldown_floor: 0.3
  inject_total_budget: 1200
  recent_context_budget: 300
  just_now_context_enabled: true
  just_now_context_hours: 6
  just_now_context_max_turns: 5
  just_now_context_budget: 420
  conversation_turns_max_entries: 500
  date_persona_trace_enabled: true
  date_persona_trace_budget: 220
  date_persona_trace_max_events: 5
  date_persona_trace_include_daily: true
  recalled_memory_budget: 400
  direct_render_mode: "auto"
  retrieval_mode: "graph"
  portrait_memory_enabled: true
  portrait_memory_budget: 360
  portrait_memory_max_sources: 8
  portrait_memory_include_anchors: false
  memory_detail_recall_enabled: false
  memory_detail_recall_max_ids: 3
  memory_detail_recall_budget: 1200
  relationship_weather_budget: 220
  favorite_memory_budget: 180
  favorite_memory_max_cards: 1
  related_memory_budget: 220
  current_inner_state_interval_rounds: 15
  relationship_weather_interval_rounds: 0
  favorite_memory_interval_rounds: 0
  upstream_key_cooldown_seconds: 300
${gateway_upstreams_yaml}

memory_diffusion:
  enabled: true
  max_hops: 2
  top_k: 4
  min_activation: 0.18
  hop_decays:
    - 0.8
    - 0.6
    - 0.4
    - 0.25
  decay: 0.55
  include_incoming: true
  max_paths_per_hit: 3
  chain_walk_enabled: false
  chain_max_hops: 6
  chain_min_strength: 0.2
  chain_min_confidence: 0.72
  chain_min_relation_priority: 60
  chain_max_frontier: 24
  chain_continue_relation_types:
    - "same_event"
    - "context_of"
    - "precedes"
    - "previous_context"
    - "next_context"
    - "updates"
    - "evidenced_by"
    - "reflects_on"
  relation_type_weights:
    same_event: 1.15
    context_of: 1.1

persona:
  enabled: true
  profile_id: "main"
  mode: "llm"
  base_url: $(yaml_quote "${dehy_base_url}")
  model: $(yaml_quote "${dehy_model}")
  thinking_mode: ""
  temperature: 0.1
  max_tokens: 500

portrait:
  enabled: true
  auto_enabled: true
  auto_initial_enabled: false
  daily_enabled: true
  timezone: "Asia/Shanghai"
  daily_hour: 4
  check_interval_minutes: 60
  state_path: ""
  base_url: ""
  model: ""
  thinking_mode: ""
  temperature: 0.1
  max_tokens: 3200
  json_response_format: true
  material_limit: 18
  first_run_material_limit: 160
  source_excerpt_chars: 900
  recent_continuity_days: 3
  persona_events_limit: 24
  recent_buffer_max: 24
  staging_pool_max: 24
  candidate_max: 40

reflection:
  enabled: true
  auto_enabled: true
  enrich_on_write: true
  enrich_backfill_enabled: true
  enrich_backfill_limit: 5
  base_url: $(yaml_quote "${dehy_base_url}")
  model: $(yaml_quote "${dehy_model}")
  thinking_mode: ""
  timezone: "Asia/Shanghai"
  daily_hour: 4
  daily_min_memory_items: 5
  daily_conversation_turn_limit: 0
  check_interval_minutes: 60

dream:
  enabled: ${dream_enabled}
  auto_enabled: ${dream_enabled}
  surface_enabled: true
  inject_enabled: false
  retain_after_inject: false
  base_url: $(yaml_quote "${dream_base_url}")
  model: $(yaml_quote "${dream_model}")
  thinking_mode: "disabled"
  temperature: 0.85
  max_tokens: 900
  timezone: "Asia/Shanghai"
  daily_hour: 3
  run_window_hours: 3
  daily_probability: ${dream_probability}
  check_interval_minutes: 60
  min_material_count: 5
  material_window_hours: 48
  material_limit: 5
  old_echo_enabled: true
  old_echo_min_age_hours: 72
  min_surface_age_hours: 3
  surface_threshold: 0.62
  attempt_threshold: 0.45
  spontaneous_surface_prob: 0.02

# Client URL hints:
#   Ombre-Brain: http://127.0.0.1:${brain_port}
EOF
  if [[ "${FEATURE_SCOPE}" == "full" ]]; then
    cat >> "config.yaml" <<EOF
#   Gateway:     http://127.0.0.1:${gateway_port}
EOF
  fi
  printf '已写入 config.yaml\n'
}

write_compose_file() {
  local brain_port="$1"
  local gateway_port="$2"

  backup_file "${LOCAL_COMPOSE_FILE}"
  cat > "${LOCAL_COMPOSE_FILE}" <<EOF
services:
  ombre-brain:
    build: .
    container_name: ombre-brain
    restart: unless-stopped
    command: ["python", "server.py"]
    env_file:
      - .env
    environment:
      OMBRE_TRANSPORT: streamable-http
      OMBRE_BUCKETS_DIR: /data
      OMBRE_STATE_DIR: /state
EOF
  if [[ "${FEATURE_SCOPE}" == "full" ]]; then
    cat >> "${LOCAL_COMPOSE_FILE}" <<EOF
      OMBRE_GATEWAY_ADMIN_URL: http://ombre-gateway:8010/api/config
EOF
  fi
  cat >> "${LOCAL_COMPOSE_FILE}" <<EOF
    ports:
      - "${brain_port}:8000"
    volumes:
      - ./buckets:/data
      - ./state:/state
      - ./config.yaml:/app/config.yaml:ro
EOF

  if [[ "${FEATURE_SCOPE}" == "full" ]]; then
    cat >> "${LOCAL_COMPOSE_FILE}" <<EOF
  ombre-gateway:
    build: .
    container_name: ombre-gateway
    restart: unless-stopped
    command: ["python", "gateway.py"]
    env_file:
      - .env
    environment:
      OMBRE_TRANSPORT: streamable-http
      OMBRE_BUCKETS_DIR: /data
      OMBRE_STATE_DIR: /state
    ports:
      - "${gateway_port}:8010"
    volumes:
      - ./buckets:/data
      - ./state:/state
      - ./config.yaml:/app/config.yaml:ro
EOF
  fi
  printf '已写入 %s\n' "${LOCAL_COMPOSE_FILE}"
}

ensure_tools() {
  if ! command -v docker >/dev/null 2>&1; then
    printf '未找到 docker，请先安装 Docker。\n'
    return 1
  fi
  if ! docker compose version >/dev/null 2>&1 && ! command -v docker-compose >/dev/null 2>&1; then
    printf '未找到 docker compose / docker-compose。\n'
    return 1
  fi
  return 0
}

choose_deploy_target() {
  local choice
  line
  printf '选择部署环境\n'
  printf '1. VPS 部署（Docker，适合服务器/云主机）\n'
  printf '2. Windows 部署（Docker Desktop + Git Bash）\n'
  printf '3. Python 直跑（手机 / Linux / Windows 无 Docker）\n'
  read -r -p '输入（1-3）：' choice
  case "${choice}" in
    2)
      DEPLOY_TARGET="windows"
      DEPLOY_LABEL="Windows 部署"
      DEFAULT_BRAIN_PORT="18001"
      DEFAULT_GATEWAY_PORT="18002"
      ;;
    3)
      DEPLOY_TARGET="python"
      DEPLOY_LABEL="Python 直跑"
      DEFAULT_BRAIN_PORT="8000"
      DEFAULT_GATEWAY_PORT="8010"
      ;;
    *)
      DEPLOY_TARGET="vps"
      DEPLOY_LABEL="VPS 部署"
      DEFAULT_BRAIN_PORT="18001"
      DEFAULT_GATEWAY_PORT="18002"
      ;;
  esac
}

detect_python_cmd() {
  if command -v python3 >/dev/null 2>&1; then
    printf 'python3\n'
  elif command -v python >/dev/null 2>&1; then
    printf 'python\n'
  else
    return 1
  fi
}

ensure_python_tools() {
  if ! detect_python_cmd >/dev/null 2>&1; then
    printf '未找到 python。Termux 可执行：pkg install python git；Windows 请先安装 Python 并加入 PATH。\n'
    return 1
  fi
  return 0
}

load_python_direct_env() {
  if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
  fi
  export OMBRE_BUCKETS_DIR="${PWD}/buckets"
  export OMBRE_STATE_DIR="${PWD}/state"
}

select_deploy_target_for_task() {
  local title="$1"
  choose_deploy_target
  line
  printf '%s：%s\n' "${title}" "${DEPLOY_LABEL}"
}

choose_client_host() {
  case "${DEPLOY_TARGET}" in
    vps)
      CLIENT_HOST="$(prompt_text '客户端访问用的 VPS 公网 IP 或域名' '<你的VPS公网IP或域名>')"
      ;;
    windows)
      if prompt_yes_no '客户端就在这台 Windows 电脑上吗' 'y'; then
        CLIENT_HOST="127.0.0.1"
      else
        CLIENT_HOST="$(prompt_text 'Windows 电脑的局域网 IP（手机同 Wi-Fi 时使用）' '<Windows局域网IP>')"
      fi
      ;;
    python)
      if prompt_yes_no '客户端就在这台机器/手机上吗' 'y'; then
        CLIENT_HOST="127.0.0.1"
      else
        CLIENT_HOST="$(prompt_text '这台机器/手机的局域网 IP（其它设备同网段时使用）' '<局域网IP>')"
      fi
      ;;
  esac
}

print_client_guide() {
  local brain_port="$1"
  local gateway_port="$2"
  local mcp_url="http://${CLIENT_HOST}:${brain_port}/mcp"
  local dashboard_url="http://${CLIENT_HOST}:${brain_port}/dashboard"
  local gateway_base_url="http://${CLIENT_HOST}:${gateway_port}/v1"

  line
  printf '客户端填写方式\n'
  printf 'Dashboard: %s\n' "${dashboard_url}"
  printf 'MCP 工具模式 URL: %s\n' "${mcp_url}"
  if [[ "${FEATURE_SCOPE}" == "full" ]]; then
    printf 'Gateway / OpenAI-compatible Base URL: %s\n' "${gateway_base_url}"
    printf 'Gateway API Key: .env 里的 OMBRE_GATEWAY_TOKEN\n'
    printf '模型名: 客户端可从 %s/models 读取；同名模型会显示成 provider/模型名。\n' "${gateway_base_url}"
  else
    printf 'Gateway / OpenAI-compatible: 未部署；客户端请使用 MCP 工具模式。\n'
  fi
  printf '会话头: 如果客户端支持自定义 header，可加 X-Ombre-Session-Id: main\n'
  printf '\n新版使用提示：新窗口用 breath(mode="handoff")；具体事件用 breath(query="关键词或原句")。\n'
  if [[ "${FEATURE_SCOPE}" == "full" ]]; then
    printf '刚刚/上一句优先看 Gateway 的 Just Now Chat Context，不要默认查长期记忆。\n'
  else
    printf '仅 MCP 不提供 Gateway Just Now 注入；刚刚/上一句优先看客户端当前对话上下文，不要默认 breath(query="刚刚...")。\n'
  fi
  printf '自我锚点用 breath(domain="self_anchor")，分段查用 breath(domain="self_anchor", query="关键词")。\n'
  printf '画像在 Dashboard 的 Persona/画像面板手动生成/刷新；profile_fact 需要证据 bucket/moment 后再确认。\n'
  printf '暗房默认读写同一个 active 房间草稿，new_room=true 才新开；darkroom_view 只有 completeness>=1 且解锁后才回显房间 revisions。\n'
  printf '完整工具说明见 docs/Tool Guide.md；Dashboard 桶列表可批量选择并删除普通记忆桶。\n'

  case "${DEPLOY_TARGET}" in
    vps)
      if [[ "${FEATURE_SCOPE}" == "full" ]]; then
        printf '\nVPS 提醒：安全组/防火墙要放行端口 %s 和 %s；公网长期使用更建议反代到 HTTPS。\n' "${brain_port}" "${gateway_port}"
      else
        printf '\nVPS 提醒：安全组/防火墙至少要放行端口 %s；公网长期使用更建议反代到 HTTPS。\n' "${brain_port}"
      fi
      ;;
    windows)
      if [[ "${FEATURE_SCOPE}" == "full" ]]; then
        printf '\nWindows 提醒：同一台电脑填 127.0.0.1；手机连 Windows 时填 Windows 局域网 IP，并确认防火墙允许端口 %s/%s。\n' "${brain_port}" "${gateway_port}"
      else
        printf '\nWindows 提醒：同一台电脑填 127.0.0.1；手机连 Windows 时填 Windows 局域网 IP，并确认防火墙允许端口 %s。\n' "${brain_port}"
      fi
      ;;
    python)
      printf '\nPython 直跑提醒：同一台机器填 127.0.0.1；其它设备连接时填局域网 IP，并保持启动脚本后台运行。\n'
      ;;
  esac

  cat > connection_guide.txt <<EOF
Ombre-Brain client connection guide

Dashboard:
  ${dashboard_url}

MCP tool mode:
  URL: ${mcp_url}

Gateway / OpenAI-compatible:
EOF
  if [[ "${FEATURE_SCOPE}" == "full" ]]; then
    cat >> connection_guide.txt <<EOF
  Base URL: ${gateway_base_url}
  API Key: value of OMBRE_GATEWAY_TOKEN in .env
  Models endpoint: ${gateway_base_url}/models
  Optional header: X-Ombre-Session-Id: main
EOF
  else
    cat >> connection_guide.txt <<EOF
  Not deployed. Use MCP tool mode instead.
EOF
  fi
  cat >> connection_guide.txt <<EOF

Current mainline usage notes:
  New window / wakeup / room switch:
    breath(mode="handoff")
    or breath(is_session_start=true)

  Specific old event, preference, boundary, project, or remembered phrase:
    breath(query="keywords or original phrase")

  Self anchor:
    breath(domain="self_anchor")
    breath(domain="self_anchor", query="keyword")
    Do not use bare breath(query="self_anchor").

  Just now / previous message / recently said password:
EOF
  if [[ "${FEATURE_SCOPE}" == "full" ]]; then
    cat >> connection_guide.txt <<EOF
    Prefer Gateway Just Now Chat Context. Do not default to breath(query="just now").
EOF
  else
    cat >> connection_guide.txt <<EOF
    MCP-only deployments do not inject Gateway Just Now Chat Context.
    Prefer the current client conversation context. Do not default to breath(query="just now").
EOF
  fi
  cat >> connection_guide.txt <<EOF

  Portrait and Profile facts:
    Use Dashboard -> Persona/Portrait panel to generate or refresh portrait_state.json.
    Portrait state may hold profile_fact_candidates; it does not write profile_fact automatically.
    Use profile_fact(...) only with evidence bucket or evidence moment.
    Gateway portrait memory reads profile_fact by default and does not include ordinary anchors unless explicitly enabled.

  Darkroom:
    Use darkroom_continue_context(limit=3) only to read the current active room draft, continue private reflection, and judge completeness.
    darkroom_enter note should default to first person; do not use third-person self-reference unless quoting external facts or Xiaoyu.
    darkroom_enter updates the current active room draft by default; pass new_room=true to open a separate room.
    External client tool lists should expose darkroom_enter(note=..., visibility="active", lock_for="6h", new_room=false) and read-only darkroom_view.
    visibility can be active / archived / retracted.
    darkroom_view returns room revision contents only when visibility is active, completeness is 1, and the lock has expired.

  Dream Context:
    dream.surface_enabled controls breath() dream surfacing.
    dream.inject_enabled controls Gateway Dream Context injection and defaults to false.

  Dashboard:
    Bucket list supports bulk select -> select current filter -> delete selected.
    Bulk delete skips protected, pinned, anchor, and permanent buckets, and requires typing DELETE.

Tool copy block:
  See docs/Tool Guide.md and paste the current copy block into external clients.
EOF
  printf '\n已写入 connection_guide.txt\n'
}

start_python_runtime() {
  local python_cmd
  python_cmd="$(detect_python_cmd)" || return 1

  if prompt_yes_no '现在安装/更新 Python 依赖吗' 'y'; then
    "${python_cmd}" -m pip install -r requirements.txt || return 1
  fi

  mkdir -p logs state buckets
  cat > start_local.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
PYTHON_CMD="${PYTHON_CMD:-python3}"
if ! command -v "${PYTHON_CMD}" >/dev/null 2>&1; then
  PYTHON_CMD="python"
fi
mkdir -p logs state buckets
set -a
source .env
set +a
export OMBRE_TRANSPORT=streamable-http
export OMBRE_BUCKETS_DIR="${PWD}/buckets"
export OMBRE_STATE_DIR="${PWD}/state"
EOF
  if [[ "${FEATURE_SCOPE}" == "full" ]]; then
    cat >> start_local.sh <<'EOF'
export OMBRE_GATEWAY_ADMIN_URL="http://127.0.0.1:8010/api/config"
EOF
  fi
  cat >> start_local.sh <<'EOF'
nohup "${PYTHON_CMD}" server.py > logs/ombre-brain.log 2>&1 &
echo $! > state/ombre-brain.pid
echo "Ombre-Brain started: http://127.0.0.1:8000/health"
EOF
  if [[ "${FEATURE_SCOPE}" == "full" ]]; then
    cat >> start_local.sh <<'EOF'
nohup "${PYTHON_CMD}" gateway.py > logs/ombre-gateway.log 2>&1 &
echo $! > state/ombre-gateway.pid
echo "Ombre-Gateway started: http://127.0.0.1:8010/health"
EOF
  fi
  chmod +x start_local.sh
  cat > start_mobile.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/start_local.sh" "$@"
EOF
  chmod +x start_mobile.sh

  cat > start_local.ps1 <<'EOF'
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$PythonCmd = $env:PYTHON_CMD
if ([string]::IsNullOrWhiteSpace($PythonCmd)) {
  $PythonCmd = "python"
}

New-Item -ItemType Directory -Force -Path logs,state,buckets | Out-Null

if (Test-Path ".env") {
  Get-Content ".env" | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq "" -or $line.StartsWith("#")) { return }
    if ($line.StartsWith("export ")) { $line = $line.Substring(7).Trim() }
    $idx = $line.IndexOf("=")
    if ($idx -le 0) { return }
    $name = $line.Substring(0, $idx).Trim()
    $value = $line.Substring($idx + 1).Trim().Trim('"').Trim("'")
    [Environment]::SetEnvironmentVariable($name, $value, "Process")
  }
}

$env:OMBRE_TRANSPORT = "streamable-http"
$env:OMBRE_BUCKETS_DIR = Join-Path $Root "buckets"
$env:OMBRE_STATE_DIR = Join-Path $Root "state"
EOF
  if [[ "${FEATURE_SCOPE}" == "full" ]]; then
    cat >> start_local.ps1 <<'EOF'
$env:OMBRE_GATEWAY_ADMIN_URL = "http://127.0.0.1:8010/api/config"
EOF
  fi
  cat >> start_local.ps1 <<'EOF'

$brainOut = Join-Path $Root "logs/ombre-brain.log"
$brainErr = Join-Path $Root "logs/ombre-brain.err.log"
$brain = Start-Process -FilePath $PythonCmd -ArgumentList "server.py" -RedirectStandardOutput $brainOut -RedirectStandardError $brainErr -PassThru -WindowStyle Hidden
Set-Content -Path (Join-Path $Root "state/ombre-brain.pid") -Value $brain.Id
Write-Host "Ombre-Brain started: http://127.0.0.1:8000/health"
EOF
  if [[ "${FEATURE_SCOPE}" == "full" ]]; then
    cat >> start_local.ps1 <<'EOF'
$gatewayOut = Join-Path $Root "logs/ombre-gateway.log"
$gatewayErr = Join-Path $Root "logs/ombre-gateway.err.log"
$gateway = Start-Process -FilePath $PythonCmd -ArgumentList "gateway.py" -RedirectStandardOutput $gatewayOut -RedirectStandardError $gatewayErr -PassThru -WindowStyle Hidden
Set-Content -Path (Join-Path $Root "state/ombre-gateway.pid") -Value $gateway.Id
Write-Host "Ombre-Gateway started: http://127.0.0.1:8010/health"
EOF
  fi

  if prompt_yes_no '现在后台启动 Python 直跑服务吗' 'y'; then
    PYTHON_CMD="${python_cmd}" ./start_local.sh
  else
    printf '已生成 start_local.sh / start_local.ps1。\n'
    printf 'Linux/Termux/Git Bash 执行：./start_local.sh\n'
    printf 'Windows PowerShell 执行：powershell -ExecutionPolicy Bypass -File .\\start_local.ps1\n'
  fi
}

update_python_runtime() {
  local python_cmd
  python_cmd="$(detect_python_cmd)" || {
    printf '未找到 python。Termux 可执行：pkg install python git；Windows 请先安装 Python 并加入 PATH。\n'
    return 1
  }

  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    ombre_update_git_checkout || return 1
  fi

  if prompt_yes_no '现在安装/更新 Python 依赖吗' 'y'; then
    "${python_cmd}" -m pip install -r requirements.txt || return 1
  fi

  printf '\nPython 直跑更新完成。\n'
  if [[ -f "start_local.sh" || -f "start_local.ps1" ]]; then
    printf '如果服务已经在跑，请手动结束旧进程后重新执行 start_local.sh 或 start_local.ps1。\n'
  else
    printf '未找到 start_local 启动脚本；如尚未首次部署，请先走菜单 1。\n'
  fi
}

doctor_python_runtime() {
  local python_cmd
  local gateway_expected="false"
  line
  printf 'Python 直跑错误排查\n'

  if python_cmd="$(detect_python_cmd)"; then
    printf 'OK   Python 可用：%s\n' "${python_cmd}"
  else
    printf 'FAIL 未找到 python。Termux 可执行：pkg install python git；Windows 请先安装 Python 并加入 PATH。\n'
  fi

  if [[ -f ".env" ]]; then
    printf 'OK   .env 存在\n'
  else
    printf 'WARN .env 不存在；请先走首次部署。\n'
  fi

  if [[ -f "config.yaml" ]]; then
    printf 'OK   config.yaml 存在\n'
  else
    printf 'WARN config.yaml 不存在；请先走首次部署。\n'
  fi

  if { [[ -f "start_local.sh" ]] && grep -q "gateway.py" "start_local.sh"; } || { [[ -f "start_local.ps1" ]] && grep -q "gateway.py" "start_local.ps1"; }; then
    gateway_expected="true"
  fi

  local keys=(OMBRE_API_KEY OMBRE_EMBEDDING_API_KEY OMBRE_DREAM_API_KEY)
  if [[ "${gateway_expected}" == "true" ]]; then
    keys+=(OMBRE_GATEWAY_TOKEN)
  else
    printf 'INFO 当前启动脚本未启用 Gateway，跳过 Gateway token 检查。\n'
  fi

  for key in "${keys[@]}"; do
    if env_file_has_value "${key}"; then
      printf 'OK   %s 已配置\n' "${key}"
    else
      printf 'WARN %s 未配置\n' "${key}"
    fi
  done

  if [[ -f "state/ombre-brain.pid" ]]; then
    local brain_pid
    brain_pid="$(cat state/ombre-brain.pid 2>/dev/null || true)"
    if [[ -n "${brain_pid}" ]] && kill -0 "${brain_pid}" >/dev/null 2>&1; then
      printf 'OK   ombre-brain 进程存在：%s\n' "${brain_pid}"
    else
      printf 'WARN ombre-brain pid 文件存在，但进程可能没在跑。\n'
    fi
  else
    printf 'WARN 未找到 state/ombre-brain.pid；可能还没启动 start_local。\n'
  fi

  if [[ "${gateway_expected}" == "true" ]]; then
    if [[ -f "state/ombre-gateway.pid" ]]; then
      local gateway_pid
      gateway_pid="$(cat state/ombre-gateway.pid 2>/dev/null || true)"
      if [[ -n "${gateway_pid}" ]] && kill -0 "${gateway_pid}" >/dev/null 2>&1; then
        printf 'OK   ombre-gateway 进程存在：%s\n' "${gateway_pid}"
      else
        printf 'WARN ombre-gateway pid 文件存在，但进程可能没在跑。\n'
      fi
    else
      printf 'WARN 未找到 state/ombre-gateway.pid；可能还没启动 start_local。\n'
    fi
  else
    printf 'INFO 当前 Python 直跑未启用 Gateway，跳过 Gateway 进程检查。\n'
  fi

  if command -v curl >/dev/null 2>&1; then
    if curl -fsS --max-time 5 "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
      printf 'OK   Ombre-Brain health 通：http://127.0.0.1:8000/health\n'
    else
      printf 'WARN Ombre-Brain health 不通：http://127.0.0.1:8000/health\n'
    fi
    if [[ "${gateway_expected}" == "true" ]]; then
      if curl -fsS --max-time 5 "http://127.0.0.1:8010/health" >/dev/null 2>&1; then
        printf 'OK   Gateway health 通：http://127.0.0.1:8010/health\n'
      else
        printf 'WARN Gateway health 不通：http://127.0.0.1:8010/health\n'
      fi
    fi
  else
    printf 'WARN 未找到 curl，跳过 health 检查。\n'
  fi

  local log_files=(logs/ombre-brain.log)
  if [[ "${gateway_expected}" == "true" ]]; then
    log_files+=(logs/ombre-gateway.log)
  fi
  for log_file in "${log_files[@]}"; do
    if [[ -f "${log_file}" ]]; then
      printf '\n最近日志：%s\n' "${log_file}"
      grep -Eai 'error|exception|traceback|401|403|429|500|502|503|504|connection refused|address already in use|api key|unauthorized|permission denied|timeout' "${log_file}" \
        | tail -n 12 \
        | sed 's/^/  /' || true
    else
      printf 'WARN 未找到日志文件：%s\n' "${log_file}"
    fi
  done

  printf '\nPython 直跑客户端常用填写：\n'
  if [[ "${gateway_expected}" == "true" ]]; then
    printf '  Gateway Base URL: http://127.0.0.1:8010/v1\n'
  else
    printf '  Gateway Base URL: 未启用；请使用 MCP 工具模式\n'
  fi
  printf '  MCP URL: http://127.0.0.1:8000/mcp\n'
  printf '  Dashboard: http://127.0.0.1:8000/dashboard\n'
}

install_shortcut() {
  local repo_root shell_rc alias_line marker
  repo_root="$(pwd)"
  marker="# Ombre-Brain one-click shortcut"
  alias_line="alias ob='cd $(printf "%q" "${repo_root}") && bash scripts/one_click.sh'"

  if [[ -n "${BASH_VERSION:-}" ]]; then
    shell_rc="${HOME}/.bashrc"
  elif [[ -n "${ZSH_VERSION:-}" ]]; then
    shell_rc="${HOME}/.zshrc"
  else
    shell_rc="${HOME}/.profile"
  fi

  touch "${shell_rc}" || return 1
  if grep -Fq "${marker}" "${shell_rc}"; then
    printf '已存在 ob 快捷命令配置：%s\n' "${shell_rc}"
  else
    {
      printf '\n%s\n' "${marker}"
      printf '%s\n' "${alias_line}"
    } >> "${shell_rc}"
    printf '已写入 ob 快捷命令：%s\n' "${shell_rc}"
  fi

  printf '当前窗口立即生效可执行：\n'
  printf '  source %s\n' "${shell_rc}"
  printf '之后直接输入：ob\n'
}

first_deploy() {
  choose_deploy_target
  choose_feature_scope
  line
  printf '当前选择：%s / %s\n' "${DEPLOY_LABEL}" "${FEATURE_LABEL}"
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    printf '首次部署会生成 .env、config.yaml、start_local.sh 和 start_local.ps1。\n'
  else
    printf '首次部署会生成 .env、config.yaml、%s，并启动容器。\n' "${LOCAL_COMPOSE_FILE}"
  fi
  printf '已有同名文件会先备份。\n'
  line
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    ensure_python_tools || return 1
  else
    ensure_tools || return 1
  fi

  local ai_name user_name user_display_name
  ai_name="$(prompt_text 'AI 名字' 'Haven')"
  user_name="$(prompt_text '用户英文/内部名' 'Rain')"
  user_display_name="$(prompt_text '用户显示名' '小雨')"

  local dehy_base_url dehy_model dehy_key
  dehy_base_url="$(prompt_text '脱水/导入抽取 base_url' 'https://api.deepseek.com')"
  dehy_model="$(prompt_text '脱水/导入抽取模型' 'deepseek-v4-flash')"
  dehy_key="$(prompt_secret '脱水模型 key（OMBRE_API_KEY，必填）' true)"

  local embedding_enabled embedding_base_url embedding_model embedding_key
  local reranker_enabled reranker_base_url reranker_model reranker_key
  if prompt_yes_no '启用 embedding 语义检索吗' 'y'; then
    embedding_enabled="true"
    embedding_base_url="$(prompt_text 'embedding base_url' 'https://api.siliconflow.cn/v1')"
    embedding_model="$(prompt_text 'embedding 模型' 'Qwen/Qwen3-Embedding-4B')"
    embedding_key="$(prompt_secret 'embedding key（OMBRE_EMBEDDING_API_KEY，建议必填）' true)"

    line
    printf '重排序 reranker 会在 embedding 找到候选后再精排，推荐 SiliconFlow 的 Qwen/Qwen3-Reranker-4B。\n'
    printf '默认复用 embedding 的 base_url/key；如果你的 embedding 服务不支持 /rerank，可以单独填写 reranker 地址和 key。\n'
    reranker_enabled="true"
    reranker_model="$(prompt_text 'reranker 模型' 'Qwen/Qwen3-Reranker-4B')"
    if prompt_yes_no 'reranker 复用 embedding base_url/key 吗' 'y'; then
      reranker_base_url=""
      reranker_key=""
    else
      reranker_base_url="$(prompt_text 'reranker base_url' "${embedding_base_url}")"
      reranker_key="$(prompt_secret 'reranker key（OMBRE_RERANKER_API_KEY）' true)"
    fi
  else
    embedding_enabled="false"
    embedding_base_url=""
    embedding_model=""
    embedding_key=""
    reranker_enabled="false"
    reranker_base_url=""
    reranker_model="Qwen/Qwen3-Reranker-4B"
    reranker_key=""
  fi

  local gateway_token
  if [[ "${FEATURE_SCOPE}" == "full" ]]; then
    configure_gateway_upstreams "${dehy_base_url}" "${dehy_model}" "${dehy_key}"
    gateway_token="$(prompt_secret 'Gateway 访问 token（回车自动生成）' false)"
    if [[ -z "${gateway_token}" ]]; then
      gateway_token="$(random_token)"
      printf '已自动生成 OMBRE_GATEWAY_TOKEN。\n'
    fi
  else
    gateway_token=""
    GATEWAY_UPSTREAMS_YAML="  upstreams: []"
  fi

  local dream_enabled dream_base_url dream_model dream_key dream_probability
  if prompt_yes_no '启用夜梦吗' 'y'; then
    dream_enabled="true"
    dream_base_url="$(prompt_text '夜梦 base_url' 'https://api.deepseek.com')"
    dream_model="$(prompt_text '夜梦模型' 'deepseek-v4-flash')"
    dream_probability="$(prompt_text '做梦概率（0-1）' '0.4')"
    if prompt_yes_no '夜梦 key 复用脱水 key 吗' 'y'; then
      dream_key="${dehy_key}"
    else
      dream_key="$(prompt_secret '夜梦 key（OMBRE_DREAM_API_KEY）' true)"
    fi
  else
    dream_enabled="false"
    dream_base_url="https://api.deepseek.com"
    dream_model="deepseek-v4-flash"
    dream_probability="0"
    dream_key=""
  fi

  local persona_key reflection_key
  persona_key="$(prompt_secret 'Persona key（可回车，默认复用 OMBRE_API_KEY）' false)"
  reflection_key="$(prompt_secret 'Reflection key（可回车，默认复用 OMBRE_API_KEY/Persona）' false)"

  local brain_port gateway_port
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    brain_port="${DEFAULT_BRAIN_PORT}"
    gateway_port="${DEFAULT_GATEWAY_PORT}"
    if [[ "${FEATURE_SCOPE}" == "full" ]]; then
      printf 'Python 直跑使用固定端口：Ombre-Brain=%s，Gateway=%s。\n' "${brain_port}" "${gateway_port}"
    else
      printf 'Python 直跑使用固定端口：Ombre-Brain=%s；Gateway 不启动。\n' "${brain_port}"
    fi
  else
    brain_port="$(prompt_text 'Ombre-Brain 对外端口' "${DEFAULT_BRAIN_PORT}")"
    if [[ "${FEATURE_SCOPE}" == "full" ]]; then
      gateway_port="$(prompt_text 'Gateway 对外端口' "${DEFAULT_GATEWAY_PORT}")"
    else
      gateway_port="${DEFAULT_GATEWAY_PORT}"
    fi
  fi
  choose_client_host

  write_env_file "${dehy_key}" "${embedding_key}" "${gateway_token}" "${dream_key}" "${persona_key}" "${reflection_key}" "${reranker_key}"
  write_config_file \
    "${ai_name}" "${user_name}" "${user_display_name}" \
    "${dehy_base_url}" "${dehy_model}" \
    "${embedding_enabled}" "${embedding_base_url}" "${embedding_model}" \
    "${reranker_enabled}" "${reranker_base_url}" "${reranker_model}" \
    "${GATEWAY_UPSTREAMS_YAML}" \
    "${dream_enabled}" "${dream_base_url}" "${dream_model}" "${dream_probability}" \
    "${brain_port}" "${gateway_port}"

  mkdir -p buckets state

  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    start_python_runtime || return 1
  else
    write_compose_file "${brain_port}" "${gateway_port}"
    export COMPOSE_FILE="${LOCAL_COMPOSE_FILE}"
    export HEALTH_URL="http://127.0.0.1:${brain_port}/health"
    printf '\n开始构建并启动容器...\n'
    "${SCRIPT_DIR}/update_deploy.sh"
  fi

  print_client_guide "${brain_port}" "${gateway_port}"
}

choose_compose_file() {
  local default="${COMPOSE_FILE:-}"
  if [[ -z "${default}" ]]; then
    if [[ -f "${LOCAL_COMPOSE_FILE}" ]]; then
      default="${LOCAL_COMPOSE_FILE}"
    else
      default="$(ombre_compose_file)"
    fi
  fi
  COMPOSE_FILE="$(prompt_text 'Compose 文件' "${default}")"
  export COMPOSE_FILE
}

update_version() {
  select_deploy_target_for_task "更新版本"
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    ensure_python_tools || return 1
  else
    choose_compose_file
    ensure_tools || return 1
  fi
  if prompt_yes_no '更新前备份记忆桶吗（包含 buckets/data、state、config.yaml、.env）' 'y'; then
    backup_current_deployment "pre_update" || return 1
  else
    printf '已跳过更新前备份。\n'
  fi
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    update_python_runtime
  else
    "${SCRIPT_DIR}/update_deploy.sh"
  fi
}

run_doctor() {
  select_deploy_target_for_task "错误排查"
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    doctor_python_runtime
  else
    choose_compose_file
    "${SCRIPT_DIR}/doctor.sh"
  fi
}

migration_prepare_target() {
  local title="$1"
  select_deploy_target_for_task "${title}"
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    ensure_python_tools || return 1
  else
    choose_compose_file
  fi
}

migration_state_dir() {
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    printf 'state\n'
  else
    printf '/state\n'
  fi
}

compose_host_volume_for() {
  local container_path="$1"
  local compose_file="${2:-${COMPOSE_FILE:-}}"
  [[ -n "${compose_file}" && -f "${compose_file}" ]] || return 1
  awk -v target="${container_path}" '
    /^[[:space:]]*-[[:space:]]*/ {
      line = $0
      sub(/^[[:space:]]*-[[:space:]]*/, "", line)
      gsub(/^["'\'']|["'\'']$/, "", line)
      split(line, parts, ":")
      if (length(parts) >= 2 && parts[2] == target) {
        print parts[1]
        exit
      }
    }
  ' "${compose_file}"
}

abs_path_from_compose() {
  local path="$1"
  local compose_file="${2:-${COMPOSE_FILE:-}}"
  [[ -n "${path}" ]] || return 1
  case "${path}" in
    /*) printf '%s\n' "${path}" ;;
    ./*|../*)
      local compose_dir
      compose_dir="$(cd "$(dirname "${compose_file}")" 2>/dev/null && pwd -P)" || return 1
      (cd "${compose_dir}" && cd "$(dirname "${path}")" 2>/dev/null && printf '%s/%s\n' "$(pwd -P)" "$(basename "${path}")")
      ;;
    *) printf '%s\n' "${path}" ;;
  esac
}

migration_default_target_buckets_dir() {
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    printf '%s/buckets\n' "$(pwd -P)"
    return 0
  fi
  local host_path
  host_path="$(compose_host_volume_for "/data" "${COMPOSE_FILE}" || true)"
  if [[ -n "${host_path}" ]]; then
    abs_path_from_compose "${host_path}" "${COMPOSE_FILE}" || true
    return 0
  fi
  printf '/srv/ombre-brain/buckets\n'
}

migration_default_target_state_dir() {
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    printf '%s/state\n' "$(pwd -P)"
    return 0
  fi
  local host_path
  host_path="$(compose_host_volume_for "/state" "${COMPOSE_FILE}" || true)"
  if [[ -n "${host_path}" ]]; then
    abs_path_from_compose "${host_path}" "${COMPOSE_FILE}" || true
    return 0
  fi
  printf '/srv/ombre-brain/state\n'
}

migration_mapping_path() {
  printf '%s/feel_comment_backfill_mapping.json\n' "$(migration_state_dir)"
}

migration_review_path() {
  printf '%s/feel_comment_backfill_review.md\n' "$(migration_state_dir)"
}

run_target_shell() {
  local script="$1"
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    bash -lc "set -euo pipefail; if [[ -f .env ]]; then set -a; source .env; set +a; fi; export OMBRE_BUCKETS_DIR=\"\${PWD}/buckets\"; export OMBRE_STATE_DIR=\"\${PWD}/state\"; ${script}"
  else
    local service="${OMBRE_SERVICE:-ombre-brain}"
    ombre_compose -f "${COMPOSE_FILE}" exec -T "${service}" sh -lc "${script}"
  fi
}

run_target_python_stdin() {
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    local python_cmd
    python_cmd="$(detect_python_cmd)" || return 1
    load_python_direct_env
    "${python_cmd}" -
  else
    local service="${OMBRE_SERVICE:-ombre-brain}"
    ombre_compose -f "${COMPOSE_FILE}" exec -T "${service}" python -
  fi
}

safe_backup_label() {
  local value="${1:-manual}"
  value="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')"
  value="$(printf '%s' "${value}" | sed -E 's/[^a-z0-9._-]+/_/g; s/^_+//; s/_+$//')"
  printf '%s\n' "${value:-manual}"
}

backup_current_deployment() {
  local label stamp archive_name archive tmp_archive
  local service config_bind_source env_bind_source
  label="$(safe_backup_label "${1:-manual}")"
  stamp="$(date +%Y%m%d_%H%M%S)"
  archive_name="ombre_backup_${label}_${stamp}.tar.gz"

  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    local items=()
    archive="state/backups/${archive_name}"
    tmp_archive="/tmp/${archive_name}"
    mkdir -p state/backups
    [[ -d buckets ]] && items+=("buckets")
    [[ -d state ]] && items+=("state")
    [[ -f config.yaml ]] && items+=("config.yaml")
    [[ -f .env ]] && items+=(".env")
    if (( ${#items[@]} == 0 )); then
      printf '没有找到可备份的 buckets/state/config.yaml/.env。\n'
      return 1
    fi
    tar --exclude='state/backups' --exclude='./state/backups' -czf "${tmp_archive}" "${items[@]}" && mv "${tmp_archive}" "${archive}" || {
      rm -f "${tmp_archive}"
      printf '备份失败。\n'
      return 1
    }
    printf '已写入备份：%s\n' "${archive}"
  else
    service="${OMBRE_SERVICE:-ombre-brain}"
    archive="/state/backups/${archive_name}"
    run_target_shell "set -e; mkdir -p /state/backups; items=''; for item in /data /state /app/config.yaml /app/.env; do [ -e \"\$item\" ] && items=\"\$items \$item\"; done; if [ -z \"\$items\" ]; then echo '没有找到可备份的 /data /state /app/config.yaml /app/.env'; exit 1; fi; tar --exclude=/state/backups --exclude=state/backups -czf '/tmp/${archive_name}' \$items; cp '/tmp/${archive_name}' '${archive}'" || return 1
    if env_bind_source="$(ombre_compose_bind_source "${COMPOSE_FILE}" "${service}" "/app/.env")"; then
      backup_file "${env_bind_source}"
    else
      backup_file ".env"
    fi
    if config_bind_source="$(ombre_compose_bind_source "${COMPOSE_FILE}" "${service}" "/app/config.yaml")"; then
      backup_file "${config_bind_source}"
    else
      backup_file "config.yaml"
    fi
    backup_file "${COMPOSE_FILE}"
    printf '已写入容器数据备份：%s\n' "${archive}"
    printf '宿主机中存在的实际 .env / config.yaml 挂载源和 compose 也已分别备份。\n'
    ombre_validate_compose_file_bind "${COMPOSE_FILE}" "${service}" "/app/config.yaml" "config.yaml" || return 1
  fi
}

backup_current_menu() {
  select_deploy_target_for_task "备份当前部署"
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    ensure_python_tools || return 1
  else
    choose_compose_file
    ensure_tools || return 1
  fi
  backup_current_deployment "manual"
}

backup_list_archives() {
  run_target_python_stdin <<'PY'
from datetime import datetime
from pathlib import Path

base = Path("/state/backups") if Path("/state").exists() else Path("state/backups")
suffixes = (".tar.gz", ".tgz", ".zip")
if not base.exists():
    raise SystemExit(0)
paths = [
    path
    for path in base.iterdir()
    if path.is_file() and any(path.name.endswith(suffix) for suffix in suffixes)
]
paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
for path in paths:
    stat = path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{path.name}\t{stat.st_size}\t{mtime}")
PY
}

format_bytes_kib() {
  local bytes="${1:-0}"
  if ! [[ "${bytes}" =~ ^[0-9]+$ ]]; then
    printf '? KiB'
    return
  fi
  printf '%s KiB' "$(((bytes + 1023) / 1024))"
}

safe_backup_archive_name() {
  local name="$1"
  [[ "${name}" =~ ^[A-Za-z0-9._-]+$ ]] || return 1
  case "${name}" in
    *.tar.gz|*.tgz|*.zip) return 0 ;;
    *) return 1 ;;
  esac
}

delete_backup_archive() {
  local name="$1"
  safe_backup_archive_name "${name}" || {
    printf '备份包文件名不安全，拒绝删除：%s\n' "${name}"
    return 1
  }
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    rm -f -- "state/backups/${name}"
  else
    run_target_shell "rm -f -- '/state/backups/${name}'" || return 1
  fi
}

delete_backup_archives_menu() {
  select_deploy_target_for_task "删除旧备份包"
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    ensure_python_tools || return 1
  else
    choose_compose_file
    ensure_tools || return 1
  fi

  local list_output rows=() names=()
  if ! list_output="$(backup_list_archives)"; then
    printf '读取备份列表失败。\n'
    return 1
  fi
  if [[ -z "${list_output}" ]]; then
    printf '没有找到备份包。\n'
    return 0
  fi
  mapfile -t rows <<< "${list_output}"

  line
  printf '旧备份包\n'
  local i=1 row name bytes mtime
  for row in "${rows[@]}"; do
    IFS=$'\t' read -r name bytes mtime <<< "${row}"
    [[ -n "${name}" ]] || continue
    names+=("${name}")
    printf '%2d. %-48s %10s  %s\n' "${i}" "${name}" "$(format_bytes_kib "${bytes}")" "${mtime}"
    i=$((i + 1))
  done
  if (( ${#names[@]} == 0 )); then
    printf '没有找到备份包。\n'
    return 0
  fi

  local raw parts=() selected=() part idx confirm
  if ! read -r -p '输入要删除的序号（多个用英文逗号，0 取消）：' raw; then
    printf '\n'
    return 0
  fi
  raw="${raw//[[:space:]]/}"
  [[ -z "${raw}" || "${raw}" == "0" ]] && return 0
  IFS=',' read -r -a parts <<< "${raw}"
  for part in "${parts[@]}"; do
    if ! [[ "${part}" =~ ^[0-9]+$ ]]; then
      printf '不是有效序号：%s\n' "${part}"
      return 1
    fi
    idx=$((part))
    if (( idx < 1 || idx > ${#names[@]} )); then
      printf '序号超出范围：%s\n' "${part}"
      return 1
    fi
    selected+=("${names[$((idx - 1))]}")
  done

  printf '将删除这些备份包：\n'
  for name in "${selected[@]}"; do
    printf '  %s\n' "${name}"
  done
  if ! read -r -p '输入 DELETE 确认删除：' confirm; then
    printf '\n'
    return 0
  fi
  if [[ "${confirm}" != "DELETE" ]]; then
    printf '确认词不匹配，已取消。\n'
    return 0
  fi
  for name in "${selected[@]}"; do
    delete_backup_archive "${name}" || return 1
    printf '已删除备份包：%s\n' "${name}"
  done
}

migration_inspect() {
  migration_prepare_target "原版迁移检查" || return 1
  run_target_python_stdin <<'PY'
import asyncio
from collections import Counter

from bucket_manager import BucketManager
from utils import load_config

RELATIONSHIP_WEATHER_TAGS = {"relationship_weather", "daily_impression", "weekly_impression"}

async def main():
    config = load_config()
    mgr = BucketManager(config)
    buckets = await mgr.list_all(include_archive=True)
    types = Counter(str((b.get("metadata") or {}).get("type") or "dynamic") for b in buckets)
    feels = []
    daily_feels = []
    migrated_feel_ids = set()
    comment_count = 0
    for bucket in buckets:
        meta = bucket.get("metadata") or {}
        tags = {str(tag) for tag in meta.get("tags", []) or []}
        if meta.get("type") == "feel":
            if tags & RELATIONSHIP_WEATHER_TAGS:
                daily_feels.append(bucket)
            else:
                feels.append(bucket)
        comments = meta.get("comments") or []
        if isinstance(comments, list):
            comment_count += len(comments)
            for comment in comments:
                if isinstance(comment, dict) and comment.get("original_feel_id"):
                    migrated_feel_ids.add(str(comment["original_feel_id"]))

    print("迁移状态检查")
    print(f"buckets_dir: {config.get('buckets_dir')}")
    print(f"state_dir: {config.get('state_dir')}")
    print(f"bucket 总数: {len(buckets)}")
    for key, value in sorted(types.items()):
        print(f"  {key}: {value}")
    print(f"旧独立 feel（可审阅迁移）: {len(feels)}")
    print(f"日印象/关系天气 feel（默认不迁移）: {len(daily_feels)}")
    print(f"年轮 comments 总数: {comment_count}")
    print(f"已带 original_feel_id 的迁移年轮: {len(migrated_feel_ids)}")
    print("下一步：先备份，再生成旧 feel 审阅表和 mapping。")

asyncio.run(main())
PY
}

migration_backup() {
  migration_prepare_target "原版迁移备份" || return 1
  backup_current_deployment "migration"
}

migration_backup_source_dir() {
  local source_dir output_dir source_abs output_abs current_abs
  local stamp source_name safe_name archive tmp_archive
  local items=()

  source_dir="$(prompt_text '原版 Ombre-Brain 目录' '')"
  if [[ -z "${source_dir}" ]]; then
    printf '原版目录不能为空。\n'
    return 1
  fi
  if [[ ! -d "${source_dir}" ]]; then
    printf '目录不存在：%s\n' "${source_dir}"
    return 1
  fi

  source_abs="$(resolve_dir "${source_dir}")" || {
    printf '无法进入目录：%s\n' "${source_dir}"
    return 1
  }
  if is_too_broad_source_dir "${source_abs}"; then
    printf '这个目录太大或太泛，不适合作为原版 Ombre-Brain 目录：%s\n' "${source_abs}"
    printf '请填到具体仓库目录，例如 /opt/Ombre-Brain。\n'
    return 1
  fi

  current_abs="$(pwd -P)"
  if [[ "${source_abs}" == "${current_abs}" ]]; then
    printf '提示：你正在备份当前脚本所在目录：%s\n' "${source_abs}"
    if ! prompt_yes_no '确认继续吗' 'n'; then
      return 0
    fi
  fi

  [[ -d "${source_abs}/buckets" ]] && items+=("buckets")
  [[ -d "${source_abs}/state" ]] && items+=("state")
  [[ -f "${source_abs}/config.yaml" ]] && items+=("config.yaml")
  [[ -f "${source_abs}/.env" ]] && items+=(".env")
  if (( ${#items[@]} == 0 )); then
    printf '没有找到可备份的 buckets/state/config.yaml/.env。\n'
    printf '这不像一个 Ombre-Brain 数据目录：%s\n' "${source_abs}"
    return 1
  fi

  output_dir="$(prompt_text '备份输出目录' 'state/backups')"
  mkdir -p "${output_dir}" || return 1
  output_abs="$(resolve_dir "${output_dir}")" || {
    printf '无法进入备份输出目录：%s\n' "${output_dir}"
    return 1
  }
  if [[ "${output_abs}" == "${source_abs}" ]]; then
    printf '备份输出目录不能等于原版目录本身。\n'
    return 1
  fi

  stamp="$(date +%Y%m%d_%H%M%S)"
  source_name="$(basename "${source_abs}")"
  safe_name="${source_name//[^A-Za-z0-9._-]/_}"
  archive="${output_abs}/original_ombre_${safe_name}_${stamp}.tar.gz"
  tmp_archive="$(mktemp)" || return 1

  local tar_args=(-czf "${tmp_archive}")
  if [[ "${output_abs}" == "${source_abs}/"* ]]; then
    local exclude_rel="${output_abs#"${source_abs}/"}"
    tar_args+=(--exclude="${exclude_rel}" --exclude="${exclude_rel}/*")
  fi

  if (cd "${source_abs}" && tar "${tar_args[@]}" "${items[@]}"); then
    mv "${tmp_archive}" "${archive}" || {
      rm -f "${tmp_archive}"
      printf '移动备份文件失败。\n'
      return 1
    }
  else
    rm -f "${tmp_archive}"
    printf '备份失败。\n'
    return 1
  fi

  printf '已备份原版目录：%s\n' "${source_abs}"
  printf '包含内容：%s\n' "${items[*]}"
  printf '已写入备份：%s\n' "${archive}"
}

migration_plan_feels() {
  migration_prepare_target "旧 feel 迁移审阅" || return 1
  local limit state_dir mapping review plan_json
  limit="$(prompt_text '最多审阅多少条旧 feel' '80')"
  if ! [[ "${limit}" =~ ^[0-9]+$ ]] || (( limit < 1 )); then
    limit="80"
  fi
  state_dir="$(migration_state_dir)"
  mapping="$(migration_mapping_path)"
  review="$(migration_review_path)"
  plan_json="${state_dir}/feel_comment_backfill_plan.json"
  run_target_shell "mkdir -p '${state_dir}' && PYTHONIOENCODING=utf-8 python scripts/plan_feel_comment_backfill.py --limit '${limit}' --top 3 --min-overlap 2 --mapping-template '${mapping}' --review-markdown '${review}' > '${plan_json}'" || return 1
  printf '已生成审阅表：%s\n' "${review}"
  printf '已生成 mapping 模板：%s\n' "${mapping}"
  printf '完整候选 JSON：%s\n' "${plan_json}"
  printf '请先人工编辑 mapping，把确认的 suggested_source_bucket_id 复制到 source_bucket_id。\n'
}

migration_review_feels_interactive() {
  migration_prepare_target "逐条确认旧 feel mapping" || return 1
  local state_dir mapping plan_json
  state_dir="$(migration_state_dir)"
  mapping="$(migration_mapping_path)"
  plan_json="${state_dir}/feel_comment_backfill_plan.json"
  printf '接下来会逐条列出旧 feel 和候选源记忆。\n'
  printf '输入 y 接受第 1 个候选；输入 n 可自己填源记忆 bucket_id；输入 w 保留为 whisper。\n'
  run_target_shell "PYTHONIOENCODING=utf-8 python scripts/review_feel_comment_backfill.py --plan '${plan_json}' --mapping '${mapping}'" || return 1
  printf '已更新 mapping：%s\n' "${mapping}"
}

migration_apply_feels_dry_run() {
  migration_prepare_target "旧 feel 写入年轮预演" || return 1
  local mapping state_dir output
  state_dir="$(migration_state_dir)"
  mapping="$(prompt_text 'mapping 路径' "$(migration_mapping_path)")"
  if [[ "${mapping}" == *"'"* ]]; then
    printf 'mapping 路径不能包含单引号。\n'
    return 1
  fi
  output="${state_dir}/feel_comment_backfill_apply_dry_run.json"
  run_target_shell "PYTHONIOENCODING=utf-8 python scripts/apply_feel_comment_backfill.py --mapping '${mapping}' > '${output}'" || return 1
  printf '预演结果：%s\n' "${output}"
}

migration_apply_feels() {
  migration_prepare_target "旧 feel 写入年轮" || return 1
  local mapping state_dir output
  state_dir="$(migration_state_dir)"
  mapping="$(prompt_text '已人工确认的 mapping 路径' "$(migration_mapping_path)")"
  if [[ "${mapping}" == *"'"* ]]; then
    printf 'mapping 路径不能包含单引号。\n'
    return 1
  fi
  printf '这一步会把 mapping 里确认的旧 feel 写入源记忆 comments，并归档旧 feel。\n'
  if ! prompt_yes_no '确认已经人工检查 mapping，可以写入吗' 'n'; then
    return 0
  fi
  output="${state_dir}/feel_comment_backfill_apply.json"
  run_target_shell "PYTHONIOENCODING=utf-8 python scripts/apply_feel_comment_backfill.py --mapping '${mapping}' --apply --archive-feel --refresh-embeddings > '${output}'" || return 1
  printf '写入结果：%s\n' "${output}"
}

migration_rebuild_embeddings() {
  migration_prepare_target "迁移后重建向量库" || return 1
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    local python_cmd batch_size
    python_cmd="$(detect_python_cmd)" || return 1
    batch_size="$(prompt_text '每批处理数量' "${BATCH_SIZE:-20}")"
    load_python_direct_env
    "${python_cmd}" backfill_embeddings.py --refresh-all --batch-size "${batch_size}"
  else
    "${SCRIPT_DIR}/embedding_rebuild.sh"
  fi
}

migration_cleanup_feels_dry_run() {
  migration_prepare_target "旧 feel 清理预演" || return 1
  local state_dir output
  state_dir="$(migration_state_dir)"
  output="${state_dir}/cleanup_migrated_feel_buckets_dry_run.json"
  run_target_shell "PYTHONIOENCODING=utf-8 python scripts/cleanup_migrated_feel_buckets.py > '${output}'" || return 1
  printf '清理预演结果：%s\n' "${output}"
}

migration_cleanup_feels_apply() {
  migration_prepare_target "删除已迁移旧 feel" || return 1
  local state_dir output
  state_dir="$(migration_state_dir)"
  printf '这一步只删除已经在年轮 comments 里带 original_feel_id 的独立旧 feel。\n'
  if ! prompt_yes_no '确认已经看过清理预演，可以删除吗' 'n'; then
    return 0
  fi
  output="${state_dir}/cleanup_migrated_feel_buckets_apply.json"
  run_target_shell "PYTHONIOENCODING=utf-8 python scripts/cleanup_migrated_feel_buckets.py --apply > '${output}'" || return 1
  printf '清理结果：%s\n' "${output}"
}

migration_bucket_files_prompt() {
  local mode="$1"
  local source_dir target_buckets target_state report python_cmd
  local include_tombstones="n"
  local overwrite="n"
  local refresh_moments="y"

  migration_prepare_target "${mode}" || return 1
  ensure_python_tools || return 1
  python_cmd="$(detect_python_cmd)" || return 1

  source_dir="$(prompt_text 'v1 原部署目录或 buckets 目录' '')"
  if [[ -z "${source_dir}" ]]; then
    printf '源目录不能为空。\n'
    return 1
  fi
  if [[ ! -d "${source_dir}" ]]; then
    printf '源目录不存在：%s\n' "${source_dir}"
    return 1
  fi

  target_buckets="$(prompt_text 'v2 目标 buckets 目录' "$(migration_default_target_buckets_dir)")"
  target_state="$(prompt_text 'v2 目标 state 目录（刷新 moment 索引用）' "$(migration_default_target_state_dir)")"
  report="$(prompt_text 'JSON 报告输出路径' "${target_state}/bucket_file_migration_plan.json")"

  if prompt_yes_no '包含 .tombstones 删除记录吗' 'n'; then
    include_tombstones="y"
  fi

  if [[ "${mode}" == *"应用"* ]]; then
    if prompt_yes_no '同 ID 内容不同也覆盖 v2 吗' 'n'; then
      overwrite="y"
    fi
    if ! prompt_yes_no '确认已经看过 dry-run 报告，可以迁移吗' 'n'; then
      return 0
    fi
    backup_current_deployment "pre_bucket_file_migration" || return 1
    if ! prompt_yes_no '迁移后刷新 moment 索引吗' 'y'; then
      refresh_moments="n"
    fi
  fi

  local cmd=(
    "${python_cmd}"
    "scripts/migrate_bucket_files.py"
    "--source" "${source_dir}"
    "--target-buckets-dir" "${target_buckets}"
    "--target-state-dir" "${target_state}"
    "--output" "${report}"
  )
  [[ "${include_tombstones}" == "y" ]] && cmd+=("--include-tombstones")
  if [[ "${mode}" == *"应用"* ]]; then
    cmd+=("--apply" "--yes")
    [[ "${overwrite}" == "y" ]] && cmd+=("--overwrite")
    [[ "${refresh_moments}" == "y" ]] && cmd+=("--refresh-moments")
  fi

  PYTHONIOENCODING=utf-8 "${cmd[@]}" || return 1
  printf '迁移报告：%s\n' "${report}"
  if [[ "${mode}" == *"应用"* ]]; then
    printf '如需语义召回立刻生效，再到“向量库相关”里补缺失向量。\n'
  fi
}

migration_bucket_files_plan() {
  migration_bucket_files_prompt "迁移 buckets/comments 预演"
}

migration_bucket_files_apply() {
  migration_bucket_files_prompt "迁移 buckets/comments 应用"
}

migration_darkroom_merge_active_once() {
  migration_prepare_target "合并旧 darkroom active 条目为一个 room（建议只用一次）" || return 1
  printf '这一步只处理 visibility=active 且没有 room_id 的旧 darkroom 条目。\n'
  printf '已经有 room_id 的条目、archived、retracted 都不会改。\n'
  printf '脚本会先备份 state/darkroom/entries.jsonl，再写入同一个 room_id 和 revision=1..n。\n'
  if ! prompt_yes_no '确认执行这个一次性合并吗' 'n'; then
    return 0
  fi
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    local python_cmd
    python_cmd="$(detect_python_cmd)" || return 1
    load_python_direct_env
    PYTHONIOENCODING=utf-8 "${python_cmd}" scripts/migrate_darkroom_active_room.py --apply --yes
  else
    run_target_shell "PYTHONIOENCODING=utf-8 python scripts/migrate_darkroom_active_room.py --apply --yes"
  fi
}

migration_menu() {
  local choice
  while true; do
    line
    printf '==== 池又雨二改版 Ombre 原版迁移 ====\n'
    printf '1. 检查旧部署和迁移状态\n'
    printf '2. 备份当前部署 buckets/state\n'
    printf '3. 备份指定原版目录\n'
    printf '4. 生成新版 config/env（走首次部署向导）\n'
    printf '5. 生成旧 feel 审阅表和 mapping\n'
    printf '6. 逐条确认旧 feel mapping\n'
    printf '7. 预演已确认 mapping 写入年轮\n'
    printf '8. 应用已确认 mapping 写入年轮\n'
    printf '9. 迁移后重建向量库\n'
    printf '10. 预演清理已迁移旧 feel\n'
    printf '11. 删除已迁移旧 feel\n'
    printf '12. 预演迁移 buckets/comments 到当前 v2\n'
    printf '13. 应用迁移 buckets/comments 到当前 v2\n'
    printf '14. 合并旧 darkroom active 条目为一个 room（建议只用一次）\n'
    printf '0. 返回上一级\n'
    if ! read -r -p '输入（0-14）：' choice; then
      printf '\n'
      return 0
    fi
    case "${choice}" in
      1) migration_inspect; pause ;;
      2) migration_backup; pause ;;
      3) migration_backup_source_dir; pause ;;
      4) first_deploy; pause ;;
      5) migration_plan_feels; pause ;;
      6) migration_review_feels_interactive; pause ;;
      7) migration_apply_feels_dry_run; pause ;;
      8) migration_apply_feels; pause ;;
      9) migration_rebuild_embeddings; pause ;;
      10) migration_cleanup_feels_dry_run; pause ;;
      11) migration_cleanup_feels_apply; pause ;;
      12) migration_bucket_files_plan; pause ;;
      13) migration_bucket_files_apply; pause ;;
      14) migration_darkroom_merge_active_once; pause ;;
      0) return 0 ;;
      *) printf '请输入 0-14。\n' ;;
    esac
  done
}

bucket_format_validate_path() {
  local path="$1"
  if [[ "${path}" == *"'"* ]]; then
    printf '路径不能包含单引号：%s\n' "${path}"
    return 1
  fi
}

bucket_format_plan() {
  migration_prepare_target "记忆桶格式转换预演" || return 1
  local state_dir scope body_mode include_archive output output_md
  state_dir="$(migration_state_dir)"
  scope="$(prompt_text '扫描范围 ordinary/core/feel/all' 'all')"
  case "${scope}" in
    ordinary|core|feel|all) ;;
    *) printf '扫描范围只能是 ordinary/core/feel/all。\n'; return 1 ;;
  esac
  body_mode="$(prompt_text '无标题正文处理 skip/title/first_sentence' 'skip')"
  case "${body_mode}" in
    skip|title|first_sentence) ;;
    *) printf '无标题正文处理只能是 skip/title/first_sentence。\n'; return 1 ;;
  esac
  include_archive=""
  if prompt_yes_no '包含 archive 桶吗' 'y'; then
    include_archive="--include-archive"
  fi
  output="$(prompt_text 'JSON plan 输出路径' "${state_dir}/affect_anchor_plan.json")"
  output_md="$(prompt_text 'Markdown 审阅输出路径' "${state_dir}/affect_anchor_plan.md")"
  bucket_format_validate_path "${state_dir}" || return 1
  bucket_format_validate_path "${output}" || return 1
  bucket_format_validate_path "${output_md}" || return 1

  run_target_shell "mkdir -p '${state_dir}' && PYTHONIOENCODING=utf-8 python scripts/migrate_affect_anchor_sections.py --scope '${scope}' ${include_archive} --body-only-moment '${body_mode}' --output '${output}' --output-md '${output_md}' --preview-chars 1200" || return 1
  printf '已生成 JSON plan：%s\n' "${output}"
  printf '已生成 Markdown 审阅：%s\n' "${output_md}"
  printf '请先看审阅文件，确认后再回到本菜单选择“应用”。\n'
}

bucket_format_apply() {
  migration_prepare_target "应用记忆桶格式转换" || return 1
  local state_dir plan output stamp
  state_dir="$(migration_state_dir)"
  plan="$(prompt_text '已审阅的 JSON plan 路径' "${state_dir}/affect_anchor_plan.json")"
  bucket_format_validate_path "${state_dir}" || return 1
  bucket_format_validate_path "${plan}" || return 1
  printf '这一步会按 plan 改写 bucket 正文，并刷新相关 embedding / moment 索引。\n'
  if ! prompt_yes_no '确认已经看过审阅文件，可以应用吗' 'n'; then
    return 0
  fi
  backup_current_deployment "pre_bucket_format" || return 1
  stamp="$(date +%Y%m%d_%H%M%S)"
  output="${state_dir}/affect_anchor_apply_${stamp}.json"
  run_target_shell "PYTHONIOENCODING=utf-8 python scripts/migrate_affect_anchor_sections.py --from-plan '${plan}' --apply --yes --output '${output}'" || return 1
  printf '格式转换结果：%s\n' "${output}"
  printf '如果之后召回不稳，再跑“向量库相关 -> 补缺失向量”或“重建整个向量库”。\n'
}

bucket_format_menu() {
  local choice
  while true; do
    line
    printf '==== 记忆桶格式转换 ====\n'
    printf '1. 预演并生成审阅文件\n'
    printf '2. 应用已审阅的 plan\n'
    printf '0. 返回上一级\n'
    if ! read -r -p '输入（0-2）：' choice; then
      printf '\n'
      return 0
    fi
    case "${choice}" in
      1) bucket_format_plan; pause ;;
      2) bucket_format_apply; pause ;;
      0) return 0 ;;
      *) printf '请输入 0-2。\n' ;;
    esac
  done
}

vector_prepare_target() {
  local title="$1"
  select_deploy_target_for_task "${title}"
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    ensure_python_tools || return 1
    load_python_direct_env
  else
    choose_compose_file
  fi
}

vector_backfill_embeddings() {
  vector_prepare_target "补缺失向量" || return 1
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    local python_cmd batch_size
    python_cmd="$(detect_python_cmd)" || return 1
    batch_size="$(prompt_text '每批处理数量' "${BATCH_SIZE:-20}")"
    "${python_cmd}" backfill_embeddings.py --batch-size "${batch_size}"
  else
    "${SCRIPT_DIR}/embedding_backfill.sh"
  fi
}

vector_rebuild_embeddings() {
  vector_prepare_target "重建整个向量库" || return 1
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    local python_cmd batch_size
    python_cmd="$(detect_python_cmd)" || return 1
    batch_size="$(prompt_text '每批处理数量' "${BATCH_SIZE:-20}")"
    printf '这会刷新所有 bucket 的 embedding，可能消耗较多 API 次数。\n'
    if ! prompt_yes_no '继续吗' 'n'; then
      printf '已取消。\n'
      return 0
    fi
    "${python_cmd}" backfill_embeddings.py --refresh-all --batch-size "${batch_size}"
  else
    "${SCRIPT_DIR}/embedding_rebuild.sh"
  fi
}

vector_cleanup_orphans() {
  vector_prepare_target "检查并删除孤儿向量" || return 1
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    local python_cmd
    python_cmd="$(detect_python_cmd)" || return 1
    "${python_cmd}" scripts/cleanup_orphan_embeddings.py --delete
  else
    "${SCRIPT_DIR}/embedding_cleanup_orphans.sh"
  fi
}

vector_import_dedupe_guide() {
  vector_prepare_target "导入重复桶清理指引" || return 1
  local service="${OMBRE_SERVICE:-ombre-brain}"
  local cleanup_choice
  line
  printf '导入重复桶清理指引\n'
  line
  printf '1. 不要直接删除全部 buckets，除非这个实例里只有这次导入的数据。\n'
  printf '2. 少量重复：优先打开 Dashboard → 导入 → 已导入记忆，用「删除」或「噪声」。\n'
  printf '   Dashboard 删除会走 /api/import/review，并同步清掉对应 embedding。\n'
  printf '3. 大量重复：先备份，再只删确认属于本次导入的 bucket 文件。\n'
  printf '   建议按导入时间、标题和正文人工确认；不要批量删除 permanent/anchor/pinned 桶。\n'
  printf '4. 本菜单会先扫描重复桶；可选人工逐组确认，或一键删除 exact duplicate 中安全的一份。\n'
  printf '   扫描结果会打印重复桶前两句话和相似度；80%% 疑似重复只在人工确认模式里删除。\n\n'

  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    printf '本地 Python 部署参考命令：\n'
    printf '  mkdir -p state/backups\n'
    printf '  tar -czf "state/backups/before-import-dedupe-$(date +%%Y%%m%%d_%%H%%M%%S).tar.gz" buckets state\n'
    printf '  python scripts/cleanup_duplicate_buckets.py\n'
    printf '  python scripts/cleanup_duplicate_buckets.py --interactive --near-threshold 80\n'
    printf '  python scripts/cleanup_duplicate_buckets.py --delete --yes\n'
    printf '  # 如果你已经手动删除过 bucket 文件，再清 orphan embeddings：\n'
    printf '  python scripts/cleanup_orphan_embeddings.py --delete --yes\n'
  else
    printf 'Docker/Compose 部署参考命令：\n'
    printf '  docker compose -f %s exec -T %s sh -lc '"'"'mkdir -p /state/backups && tar -czf "/state/backups/before-import-dedupe-$(date +%%Y%%m%%d_%%H%%M%%S).tar.gz" /data /state'"'"'\n' "${COMPOSE_FILE}" "${service}"
    printf '  docker compose -f %s exec -T %s python scripts/cleanup_duplicate_buckets.py\n' "${COMPOSE_FILE}" "${service}"
    printf '  docker compose -f %s exec -T %s python scripts/cleanup_duplicate_buckets.py --interactive --near-threshold 80\n' "${COMPOSE_FILE}" "${service}"
    printf '  docker compose -f %s exec -T %s python scripts/cleanup_duplicate_buckets.py --delete --yes\n' "${COMPOSE_FILE}" "${service}"
    printf '  # 如果你已经手动删除过 bucket 文件，再清 orphan embeddings：\n'
    printf '  docker compose -f %s exec -T %s python scripts/cleanup_orphan_embeddings.py --delete --yes\n' "${COMPOSE_FILE}" "${service}"
  fi

  printf '\n'
  printf '开始扫描重复桶...\n'
  if [[ "${DEPLOY_TARGET}" == "python" ]]; then
    local python_cmd
    python_cmd="$(detect_python_cmd)" || return 1
    "${python_cmd}" scripts/cleanup_duplicate_buckets.py --limit 30 || return 1
  else
    run_target_shell "PYTHONIOENCODING=utf-8 python scripts/cleanup_duplicate_buckets.py --limit 30" || return 1
  fi

  printf '\n选择清理方式：\n'
  printf '1. 人工逐组确认（包含相似度 >=80%% 的疑似重复；y 删建议项，1/2 删左/右）\n'
  printf '2. 一键删除确定重复（只删 exact duplicate 中安全的一份）\n'
  printf '0. 不删除\n'
  if ! read -r -p '输入（0-2）：' cleanup_choice; then
    printf '\n'
    return 0
  fi

  case "${cleanup_choice}" in
    1)
      if ! prompt_yes_no '确认已经备份；进入人工确认模式吗' 'n'; then
        printf '已跳过删除。\n'
        return 0
      fi
      if [[ "${DEPLOY_TARGET}" == "python" ]]; then
        "${python_cmd}" scripts/cleanup_duplicate_buckets.py --interactive --near-threshold 80 --limit 30
      else
        run_target_shell "PYTHONIOENCODING=utf-8 python scripts/cleanup_duplicate_buckets.py --interactive --near-threshold 80 --limit 30"
      fi
      ;;
    2)
      if ! prompt_yes_no '确认已经备份；一键删除 exact duplicate 中安全的一份吗' 'n'; then
        printf '已跳过删除。\n'
        return 0
      fi
      if [[ "${DEPLOY_TARGET}" == "python" ]]; then
        "${python_cmd}" scripts/cleanup_duplicate_buckets.py --delete --yes --limit 30
      else
        run_target_shell "PYTHONIOENCODING=utf-8 python scripts/cleanup_duplicate_buckets.py --delete --yes --limit 30"
      fi
      ;;
    0|'')
      printf '已跳过删除。\n'
      ;;
    *)
      printf '请输入 0-2。\n'
      ;;
  esac

  printf '\n如果已经误删 bucket，先从备份包恢复，不要继续重建向量库。\n'
}

vector_menu() {
  local choice
  while true; do
    line
    printf '==== 池又雨二改版 Ombre 向量库相关 ====\n'
    printf '1. 补缺失向量\n'
    printf '2. 重建整个向量库\n'
    printf '3. 检查并删除孤儿向量\n'
    printf '4. 导入重复桶清理指引\n'
    printf '0. 返回上一级\n'
    if ! read -r -p '输入（0-4）：' choice; then
      printf '\n'
      return 0
    fi
    case "${choice}" in
      1) vector_backfill_embeddings; pause ;;
      2) vector_rebuild_embeddings; pause ;;
      3) vector_cleanup_orphans; pause ;;
      4) vector_import_dedupe_guide; pause ;;
      0) return 0 ;;
      *) printf '请输入 0-4。\n' ;;
    esac
  done
}

main_menu() {
  local choice
  while true; do
    line
    printf '==== 池又雨二改版 Ombre 一键部署脚本 ====\n'
    printf '1. 首次部署\n'
    printf '2. 更新版本\n'
    printf '3. 错误排查\n'
    printf '4. 备份当前部署\n'
    printf '5. 删除旧备份包\n'
    printf '6. 记忆桶格式转换\n'
    printf '7. 向量库相关\n'
    printf '8. 安装短命令 ob\n'
    printf '9. 从原版 Ombre-Brain 迁移\n'
    printf '0. 退出\n'
    if ! read -r -p '输入（0-9）：' choice; then
      printf '\n'
      exit 0
    fi
    case "${choice}" in
      1) first_deploy; pause ;;
      2) update_version; pause ;;
      3) run_doctor; pause ;;
      4) backup_current_menu; pause ;;
      5) delete_backup_archives_menu; pause ;;
      6) bucket_format_menu ;;
      7) vector_menu ;;
      8) install_shortcut; pause ;;
      9) migration_menu ;;
      0) exit 0 ;;
      *) printf '请输入 0-9。\n' ;;
    esac
  done
}

if [[ "${OMBRE_ONE_CLICK_SOURCE_ONLY:-}" != "1" ]]; then
  main_menu
fi
