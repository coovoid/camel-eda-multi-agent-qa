# EDA 多智能体整合问答系统

基于 [CAMEL-AI](https://github.com/camel-ai/camel) 框架与 RAG 技术的 EDA（电子设计自动化）领域多智能体协作问答系统，提供 Streamlit Web 界面。

## 功能概览

- **7 智能体流水线**：检索 → 要点提取 → 质量评估 → 拒绝检测 → 语义一致性 → 幻觉检测 → 整合回答
- **RAG 知识库**：支持上传 PDF / TXT / MD / DOCX / XLSX / JSON，自动分块与向量检索
- **魔搭 ModelScope API**：默认使用 OpenAI 兼容推理接口

## 项目结构

```
camel-eda-multi-agent-qa-main/
├── agent.py                 # Streamlit 前端
├── multi_agent_backend.py   # 多智能体与 RAG 后端
├── requirements.txt         # Python 依赖
├── run.bat                  # Windows 一键启动脚本
├── api_key.env.example      # API 密钥模板
├── api_key.env              # 本地密钥（需自行创建，勿提交）
├── .gitignore
└── README.md
```

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.10 / 3.11 / 3.12（推荐，勿用 3.13+） |
| 操作系统 | Windows（`run.bat`）；其他系统可手动安装依赖 |
| 网络 | 首次运行需联网安装依赖 |
| API | [魔搭社区](https://modelscope.cn/my/overview) 访问令牌，且需绑定阿里云账号 |

## 快速开始（Windows）

### 1. 配置 API 密钥

复制模板并填入魔搭令牌：

```bash
copy api_key.env.example api_key.env
```

编辑 `api_key.env`：

```env
API_KEY=你的魔搭访问令牌
```

### 2. 一键启动

双击 `run.bat`，脚本将自动：

1. 创建虚拟环境 `.venv`
2. 安装 `requirements.txt` 中的依赖
3. 启动 Streamlit（默认地址：<http://localhost:8501>）

### 3. 初始化并使用

1. 浏览器打开上述地址
2. 在左侧侧边栏选择**对话模型**（默认 `deepseek-ai/DeepSeek-V4-Flash`）
3. 点击 **「初始化系统」**
4. 输入问题或上传文档后开始对话

> 若更换模型或更新代码后异常，请先点击 **「重置系统」** 再重新初始化。

## 手动安装（可选）

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run agent.py
```

## 推荐对话模型

以下模型已在魔搭推理 API 上验证可用：

| 模型 | 说明 |
|------|------|
| `deepseek-ai/DeepSeek-V4-Flash` | **默认**，速度快 |
| `deepseek-ai/DeepSeek-V3.2` | 备选 |
| `moonshotai/Kimi-K2.5` | 备选 |
| `MiniMax/MiniMax-M2.5` | 备选 |

请勿使用 `Qwen/QVQ-72B-Preview` 等模型，易出现 `choices` 为空或 429 限流。

嵌入模型（RAG 向量化）固定为：`Qwen/Qwen3-Embedding-0.6B`

## 智能体说明

| 智能体 | 职责 |
|--------|------|
| 检索专员 | 基于知识库或模型知识生成初始回答 |
| 关键信息提取专家 | 提取核心要点 |
| 检索文档评估专家 | 评估检索内容与问题的相关性 |
| 拒绝评估专家 | 检测是否不当拒绝回答 |
| 语义一致性专家 | 检查逻辑矛盾与信息缺失 |
| 幻觉检测专家 | 识别虚构或错误信息 |
| 整合专家 | 汇总各专家意见，输出最终回答 |

## 常见问题

### 依赖安装失败

- 确认 Python 版本为 3.10–3.12
- 删除 `.venv` 文件夹后重新运行 `run.bat`
- 勿使用 `camel-ai[all]`，请使用项目自带的 `requirements.txt`

### 调度失败 / 模型返回为空

- 侧边栏确认已选择推荐模型
- 点击「重置系统」→「初始化系统」
- 若出现 429 限流，等待 1–2 分钟后重试

### API 密钥无效

- 确认已在魔搭绑定阿里云账号
- 检查 `api_key.env` 中 `API_KEY=` 后无多余空格
- 也可在侧边栏 **「API 密钥（可选）」** 中临时填写

## 技术栈

- [CAMEL-AI](https://github.com/camel-ai/camel) `0.2.38`
- [Streamlit](https://streamlit.io/)
- [魔搭 ModelScope](https://modelscope.cn/) 推理 API
- HybridRetriever + 自定义向量存储

## 许可证

本项目基于 CAMEL-AI 生态开发，请遵循各依赖库的许可协议。
