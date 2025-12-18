# Desktop AI Test Agent

一个基于 LangGraph 的智能桌面自动化测试代理框架，集成 UI-TARS Desktop 实现自然语言驱动的桌面应用自动化测试。

## 📋 项目简介

Desktop AI Test Agent 是一个智能化的桌面应用自动化测试框架，通过自然语言描述任务，自动拆解、执行和验证测试步骤。框架采用 LangGraph 构建执行图，集成 UI-TARS Desktop 进行实际的桌面操作，并使用 PostgreSQL + pgvector 存储任务历史和知识库。

## ✨ 核心特性

- 🤖 **智能任务处理**
  - 任务补全与优化：基于业务知识库自动补全和优化任务描述
  - 任务可执行性判断：智能判断任务是否可执行
  - 任务自动拆解：将复杂任务拆解为可执行的步骤序列

- 🎯 **多模态执行**
  - 截图分析：自动捕获屏幕截图并分析执行结果
  - 步骤优化：根据执行结果动态优化后续步骤
  - 智能重试：失败时自动优化并重试，最多3次

- 📚 **知识库管理**
  - 业务知识库：存储业务相关的问答知识，支持向量搜索
  - 推理知识库：存储任务拆解的历史经验，提升拆解质量
  - 任务历史库：记录任务执行历史，支持相似任务检索

- 🔄 **LangGraph 执行流程**
  - 任务补全节点：使用业务知识库补全任务描述
  - 可执行性判断节点：判断任务是否可执行
  - 任务拆解节点：将任务拆解为步骤序列
  - 步骤执行节点：执行每个步骤并验证结果
  - 知识积累节点：将成功经验积累到知识库

- 🌐 **RESTful API**
  - FastAPI 提供完整的 REST API
  - 任务管理：创建、查询、更新任务
  - 知识库管理：业务知识和推理知识的 CRUD 操作
  - 向量搜索：基于语义相似度的知识检索

## 🏗️ 技术架构

### 技术栈

- **后端框架**: FastAPI
- **执行引擎**: LangGraph
- **LLM**: OpenAI API (支持自定义 base_url)
- **桌面自动化**: UI-TARS Desktop
- **数据库**: PostgreSQL + pgvector
- **向量嵌入**: Transformers (支持本地 embedding 模型)
- **截图工具**: mss / Pillow

### 项目结构

```
desktop-ai-test-agent/
├── backend/                    # 后端代码
│   ├── action/                # 动作模块
│   │   ├── enhance_task.py           # 任务补全
│   │   ├── decompose_task.py         # 任务拆解
│   │   ├── judgment_task.py          # 可执行性判断
│   │   ├── ui_tars.py                # UI-TARS 集成
│   │   ├── accumulate_knowledge.py   # 知识积累
│   │   └── multimodal_action/        # 多模态动作
│   │       ├── analyze_step.py       # 步骤分析
│   │       └── optimize_step.py      # 步骤优化
│   ├── business_knowledge/    # 业务知识库
│   │   ├── models.py          # 数据模型
│   │   ├── database.py        # 数据库连接
│   │   ├── crud.py            # CRUD 操作
│   │   └── embedding_service.py  # 向量嵌入服务
│   ├── reasoning_knowledge/   # 推理知识库
│   │   ├── models.py
│   │   ├── database.py
│   │   ├── crud.py
│   │   └── embedding_service.py
│   ├── task_storage/          # 任务存储
│   │   ├── models.py
│   │   ├── database.py
│   │   └── crud.py
│   ├── util/                  # 工具类
│   │   ├── screenshot_util.py # 截图工具
│   │   └── markdown_util.py   # Markdown 工具
│   ├── main.py                # LangGraph 执行图
│   ├── api.py                 # FastAPI 接口
│   └── config.py              # 配置管理
├── frontend/                  # 前端代码（可选）
├── requirements.txt           # Python 依赖
└── README.md                  # 项目文档
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PostgreSQL 12+ (需要安装 pgvector 扩展)
- UI-TARS Desktop 客户端（用于桌面自动化）

### 安装步骤

1. **克隆项目**

```bash
git clone <repository-url>
cd desktop-ai-test-agent
```

2. **安装 Python 依赖**

```bash
pip install -r requirements.txt
```

**注意**: 如果需要使用 GPU 加速 embedding，请先卸载 CPU 版本的 torch，然后安装 CUDA 版本：

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
# 或
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
```

3. **配置数据库**

创建 PostgreSQL 数据库并安装 pgvector 扩展：

```sql
CREATE DATABASE ai_test_db;
\c ai_test_db
CREATE EXTENSION IF NOT EXISTS vector;
```

4. **配置环境变量**

创建 `backend/config.yml` 文件（或使用环境变量）：

```yaml
# OpenAI API 配置
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选，默认使用官方 API
MODEL_NAME=gpt-4o  # 或你使用的模型名称

# UI-TARS Desktop 配置
UI_TARS_BASE_URL=http://localhost:8080  # UI-TARS Desktop API 地址
UI_TARS_API_KEY=your_ui_tars_api_key    # 可选
UI_TARS_MODEL=your_vlm_model            # VLM 模型名称

# 数据库配置
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_test_db
```

5. **初始化数据库**

```bash
cd backend
python -c "from task_storage.database import init_db; init_db()"
python -c "from business_knowledge.database import init_db; init_db()"
python -c "from reasoning_knowledge.database import init_db; init_db()"
```

6. **启动 API 服务**

```bash
cd backend
uvicorn api:app --host 0.0.0.0 --port 8000
```

API 文档将自动生成在: http://localhost:8000/docs

## 📖 使用指南

### API 使用示例

#### 1. 创建并执行任务

```bash
curl -X POST "http://localhost:8000/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "打开记事本，输入 Hello World，保存为 test.txt"
  }'
```

响应示例：

```json
{
  "task_id": 1,
  "message": "任务已创建并开始执行",
  "status": "running"
}
```

#### 2. 查询任务状态

```bash
curl "http://localhost:8000/api/tasks/1"
```

#### 3. 添加业务知识

```bash
curl -X POST "http://localhost:8000/api/business-knowledge" \
  -H "Content-Type: application/json" \
  -d '{
    "question_text": "如何打开记事本？",
    "answer_text": "在 Windows 系统中，可以通过以下方式打开记事本：1. 按 Win+R，输入 notepad 回车；2. 在开始菜单搜索记事本；3. 在文件资源管理器中找到 C:\\Windows\\System32\\notepad.exe"
  }'
```

#### 4. 搜索业务知识

```bash
curl -X POST "http://localhost:8000/api/business-knowledge/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "如何打开应用程序",
    "top_k": 5
  }'
```

### LangGraph 执行流程

框架使用 LangGraph 构建了以下执行流程：

```
原始任务
  ↓
[任务补全节点] → 使用业务知识库补全任务描述
  ↓
[可执行性判断节点] → 判断任务是否可执行
  ↓ (可执行)
[任务拆解节点] → 将任务拆解为步骤序列
  ↓
[步骤执行节点] → 循环执行每个步骤
  │   ├─ 执行步骤（UI-TARS）
  │   ├─ 截图分析
  │   ├─ 步骤优化（如需要）
  │   └─ 重试（最多3次）
  ↓
[知识积累节点] → 将成功经验存入知识库
  ↓
任务完成
```

### 直接调用执行图

```python
from backend.main import run_task

# 执行任务
result = await run_task("打开记事本，输入 Hello World，保存为 test.txt")
print(result)
```

## 🔧 配置说明

### 配置文件

项目支持两种配置方式：

1. **配置文件**: `backend/config.yml` (KEY=VALUE 格式)
2. **环境变量**: 直接设置环境变量

配置文件示例：

```yaml
# OpenAI 配置
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o

# UI-TARS 配置
UI_TARS_BASE_URL=http://localhost:8080
UI_TARS_API_KEY=xxx
UI_TARS_MODEL=your-vlm-model

# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
```

### 主要配置项

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 密钥 | - |
| `OPENAI_BASE_URL` | OpenAI API 基础 URL | `https://api.openai.com/v1` |
| `MODEL_NAME` | 使用的 LLM 模型名称 | `gpt-4o` |
| `UI_TARS_BASE_URL` | UI-TARS Desktop API 地址 | - |
| `UI_TARS_API_KEY` | UI-TARS API 密钥（可选） | - |
| `UI_TARS_MODEL` | VLM 模型名称 | - |
| `DATABASE_URL` | PostgreSQL 数据库连接字符串 | `postgresql://postgres:postgres@localhost:5432/ai_test_db` |

## 📊 数据库模型

### 任务存储 (task_storage)

- `id`: 任务 ID
- `original_task`: 原始任务描述
- `enhanced_task`: 补全后的任务描述
- `can_execute`: 是否可执行
- `execution_reason`: 执行原因
- `steps`: 步骤列表（JSON）
- `step_results`: 步骤执行结果（JSON）
- `final_result`: 最终结果（JSON）
- `all_success`: 是否全部成功
- `created_at`: 创建时间
- `updated_at`: 更新时间

### 业务知识库 (business_knowledge)

- `id`: 知识 ID
- `question_text`: 问题文本
- `answer_text`: 回答文本
- `embedding`: 向量嵌入（pgvector）
- `created_at`: 创建时间
- `updated_at`: 更新时间

### 推理知识库 (reasoning_knowledge)

- `id`: 知识 ID
- `task_text`: 任务文本
- `step_text`: 步骤文本
- `embedding`: 向量嵌入（pgvector）
- `created_at`: 创建时间
- `updated_at`: 更新时间

## 🧪 开发指南

### 添加新的 Action

1. 在 `backend/action/` 目录下创建新的 action 文件
2. 继承 `Action` 基类
3. 实现 `run()` 方法

示例：

```python
from action.action import Action

class MyAction(Action):
    def __init__(self, llm):
        super().__init__(
            name="my_action",
            description="我的动作描述",
            llm=llm
        )
    
    async def run(self, **kwargs):
        # 实现你的逻辑
        return result
```

### 修改执行图

编辑 `backend/main.py` 中的 LangGraph 执行图，添加新的节点或修改流程。

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

## 📝 许可证

[添加许可证信息]

## 🙏 致谢

- [LangGraph](https://github.com/langchain-ai/langgraph) - 执行图框架
- [UI-TARS Desktop](https://github.com/bytedance/UI-TARS-desktop) - 桌面自动化框架
- [FastAPI](https://fastapi.tiangolo.com/) - Web 框架
- [pgvector](https://github.com/pgvector/pgvector) - PostgreSQL 向量扩展
