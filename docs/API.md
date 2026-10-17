# HTTP 接口文档

> `web_demo.py` 提供的本地 HTTP 接口。所有接口仅监听 `127.0.0.1`，默认端口 `8000`。
> 接口设计为**稳定契约**，后续接入其它客户端（桌面端、Agent 编排、测试工具）时可直接复用。

---

## 通用约定

| 项 | 说明 |
|---|---|
| Base URL | `http://127.0.0.1:8000` |
| 编码 | UTF-8 |
| 请求体格式 | `application/json` |
| 响应格式 | `application/json; charset=utf-8` |
| 成功标识 | 响应体 `ok: true` |
| 失败标识 | 响应体 `ok: false` + `error` 字段 |
| 鉴权 | 无（仅本地演示用；生产部署需自行增加） |

### 错误响应

```json
{ "ok": false, "error": "query_text 不能为空" }
```

| HTTP 状态码 | 含义 |
|---|---|
| 200 | 成功 |
| 400 | 请求参数非法（如 query_text 为空、area_id 不支持） |
| 404 | 路径不存在 |
| 500 | 服务内部异常（检索或 LLM 调用失败） |

---

## 1. 获取配置

返回知识库概况、预置事件列表与风险阈值，供前端初始化。

```
GET /api/config
```

### 响应示例

```json
{
  "ok": true,
  "events": [
    {
      "id": "ran_vision",
      "group": "装载区",
      "label": "视觉幻视",
      "area_id": "region_RAN",
      "query_text": "装载区AGV把货架看错了，出现幻视导致货物位置判断错误"
    }
  ],
  "area_labels": {
    "region_RAN": "装载区",
    "region_RAZ": "缓存区",
    "region_RDA": "充电区",
    "region_RAB": "避让区"
  },
  "kb_count": 16,
  "area_distribution": {
    "region_RAN": 4,
    "region_RAZ": 4,
    "region_RDA": 4,
    "region_RAB": 4
  },
  "embedding_dim": 512,
  "thresholds": { "no_match": 0.45, "mid": 0.67, "high": 0.78 }
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `events` | array | 预置现场事件（16 条，按区域分组） |
| `events[].id` | string | 事件唯一标识 |
| `events[].group` | string | 所属分组（区域名或"区域未知"） |
| `events[].area_id` | string \| null | 关联区域；`null` 表示区域未知（走全局检索） |
| `events[].query_text` | string | 事件文本 |
| `kb_count` | int | 知识库案例总数 |
| `area_distribution` | object | 各区域案例数 |
| `embedding_dim` | int | 向量维度（BGE-small-zh 为 512） |
| `thresholds` | object | 风险决策阈值（与后端一致） |

> 该接口耗时 < 5 ms，不调用 LLM。

---

## 2. 向量召回 + 规则决策（快接口）

只做检索与规则判断，**不调用 LLM**。用于快速展示召回结果与安全结论。

```
POST /api/retrieve
```

### 请求体

```json
{
  "query_text": "装载区AGV把货架看错了，出现幻视",
  "area_id": "region_RAN"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `query_text` | string | ✅ | 现场事件文本，不能为空 |
| `area_id` | string \| null | ❌ | 目标区域。传 `null` 或不传 = 全局检索 |

支持的 `area_id`：`region_RAN` / `region_RAZ` / `region_RDA` / `region_RAB`

### 响应示例

```json
{
  "ok": true,
  "query_text": "装载区AGV把货架看错了，出现幻视",
  "area_id": "region_RAN",
  "results": [
    {
      "score": 0.8314,
      "similarity": 0.8314,
      "fault_id": "fault_RAN_001",
      "fault_desc": "AGV在装载区视觉相机误识别货架轮廓，发生幻视并误判货物位置",
      "text": "AGV在装载区视觉相机误识别货架轮廓，发生幻视并误判货物位置",
      "area_id": "region_RAN",
      "risk_level": "高风险",
      "action": "AGV立即暂停，人工复核视觉识别结果并确认现场"
    }
  ],
  "max_sim": 0.8314,
  "avg_similarity": 0.6672,
  "cost_ms": 10.7,
  "risk_level": "高风险",
  "action": "AGV立即暂停，人工确认现场",
  "decision": "高匹配 → 高风险，立即暂停"
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `results` | array | 召回结果（最多 top_k=3 条），按相似度降序 |
| `results[].score` | float | 余弦相似度，越大越相似（字段别名为 `similarity`） |
| `results[].fault_desc` | string | 历史故障描述（字段别名为 `text`） |
| `results[].risk_level` | string | **该历史案例自身**的参考风险等级 |
| `results[].action` | string | **该历史案例自身**的参考处置动作 |
| `max_sim` | float | 召回结果中的最高相似度（**风险决策判据**） |
| `avg_similarity` | float | 平均相似度（仅记录，不参与决策） |
| `cost_ms` | float | 检索耗时（毫秒） |
| `risk_level` | string | **本次事件的规则判定结果**（高风险/中风险/低风险） |
| `action` | string | **本次事件的规则控制动作** |
| `decision` | string | 决策依据说明 |

> ⚠️ 注意区分两组 `risk_level` / `action`：
> - `results[].risk_level` 是**历史案例自带的**参考值
> - 顶层 `risk_level` 是**规则系统对本次事件的判定**
>
> 安全决策以顶层为准。

### 决策规则（硬编码，LLM 不可修改）

```
max_sim < 0.45   → 高风险：AGV减速，重新扫描环境，上传异常日志
max_sim >= 0.78  → 高风险：AGV立即暂停，人工确认现场
max_sim >= 0.67  → 中风险：AGV减速运行，二次校验传感器数据
否则             → 低风险：AGV继续正常作业
```

### 耗时

约 8~15 ms（模型已预加载）。

---

## 3. 完整分析（含 LLM）

召回 → 规则决策 → LLM 辅助分析。内部会重新执行一次检索（约 10 ms，可忽略）。

```
POST /api/analyze
```

### 请求体

与 `/api/retrieve` 完全一致。

### 响应示例

```json
{
  "ok": true,
  "query_text": "充电时电池温度过高报警，充电过程被迫中断",
  "area_id": "region_RDA",
  "risk_level": "中风险",
  "action": "AGV减速运行，二次校验传感器数据",
  "decision": "中等匹配 → 中风险，减速校验",
  "max_sim": 0.7667,
  "avg_similarity": 0.6911,
  "retrieval_cost_ms": 12.0,
  "report": "一、匹配评估：……\n二、诱因推测：……\n三、排查建议：\n1. ……",
  "harness": {
    "status": "success",
    "attempts": 1,
    "cost_ms": 1979.0,
    "validation": {
      "passed": true,
      "violations": [],
      "missing_sections": [],
      "cites_rule": true,
      "report_chars": 407,
      "too_long": false
    },
    "prompt_tokens": 1046,
    "completion_tokens": 630,
    "total_tokens": 1676,
    "report_chars": 407,
    "error": null
  }
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `report` | string | 三段式分析报告（纯文本，无 Markdown） |
| `harness.status` | string | `success`（真实报告）/ `degraded`（降级兜底） |
| `harness.attempts` | int | 实际尝试次数（含重试） |
| `harness.cost_ms` | float | LLM 调用总耗时（含重试与退避） |
| `harness.validation.passed` | bool | 输出安全校验是否通过 |
| `harness.validation.violations` | array | 命中的违规措辞（如 `["建议停机"]`） |
| `harness.validation.missing_sections` | array | 缺失的报告段落 |
| `harness.validation.cites_rule` | bool | 是否正确引用了规则结论 |
| `harness.validation.report_chars` | int | 报告字数 |
| `harness.validation.too_long` | bool | 是否超过篇幅软阈值（仅告警，不判失败） |
| `harness.total_tokens` | int | 输入 + 输出 token 总量 |
| `harness.error` | string \| null | 降级原因（`success` 时为 `null`） |

### 报告结构（固定三段）

```
一、匹配评估：<≤120 字，判断高匹配/弱相关/几乎无匹配>
二、诱因推测：<≤150 字，最多 3 点>
三、排查建议：<最多 4 条，每条 ≤40 字>
```

### 安全校验

LLM 返回后会经过 Harness 校验，命中以下任一情况即判定不合格并降级：

- 出现控制类违规措辞：`建议停机`、`建议断电`、`可以恢复运行`、`无需停机`、`建议降级`、`建议提高风险` 等
- 缺少三段结构

降级后返回的报告**只引用规则结论**，不会包含未经校验的模型推测。

### 耗时

约 2~4 秒（取决于模型输出长度）。

---

## 4. 典型调用流程

推荐前端按以下顺序调用，实现"快慢分离"的交互：

```
1. GET  /api/config        → 初始化界面，渲染事件按钮
2. POST /api/retrieve      → 立即展示召回结果 + 风险等级（~10ms）
   ↓（渲染完成，用户已能看到安全结论）
3. POST /api/analyze       → 异步等待 LLM 报告（~2s）
   ↓
4. 更新报告区域与指标
```

**为什么不合并成一个接口**：如果合并，用户点击后需等待 2 秒才看到任何反馈；且 LLM 故障时连风险等级都无法展示。

---

## 5. curl 示例

```bash
# 获取配置
curl -s http://127.0.0.1:8000/api/config | python -m json.tool

# 召回 + 决策（快）
curl -s -X POST http://127.0.0.1:8000/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query_text":"装载区AGV把货架看错了，出现幻视","area_id":"region_RAN"}' \
  | python -m json.tool

# 完整分析（含 LLM）
curl -s -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"query_text":"充电时电池温度过高报警","area_id":"region_RDA"}' \
  | python -m json.tool

# 全局检索（不传 area_id）
curl -s -X POST http://127.0.0.1:8000/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query_text":"AGV撞到东西后触发安全触边","area_id":null}' \
  | python -m json.tool
```

---

## 6. 接口稳定性承诺

以下内容视为**稳定契约**，后续版本的内部实现可优化，但这些不应破坏：

| 稳定项 | 说明 |
|---|---|
| 路径 | `/api/config`、`/api/retrieve`、`/api/analyze` |
| 方法 | GET / POST / POST |
| 请求字段 | `query_text`（必填）、`area_id`（可选） |
| 响应外层 | `ok` + `error` |
| 结果字段 | `score`、`fault_id`、`fault_desc`、`area_id`、`risk_level`、`action` |
| 别名兼容 | `score`↔`similarity`、`fault_desc`↔`text` 保持同时存在 |
| 状态值 | `harness.status` ∈ {`success`, `degraded`} |

**可能扩展但不会破坏现有字段的部分**：`harness.*` 下可新增观测指标；`results[]` 可补充元数据字段。

---

## 7. 后续可扩展方向

| 方向 | 说明 |
|---|---|
| WebSocket / SSE | 将 `/api/analyze` 改为流式返回，实现打字机效果 |
| 批量接口 | `POST /api/batch-analyze` 一次处理多条事件 |
| 知识库热更新 | `POST /api/kb/reload` 无需重启服务即可重载 |
| 鉴权 | 生产部署时增加 Token / mTLS |
| Agent 编排 | 该接口可被上层 Agent 框架直接调用，作为"故障分析工具" |
