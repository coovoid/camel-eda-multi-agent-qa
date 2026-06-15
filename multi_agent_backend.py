"""EDA 多智能体 RAG 后端。"""

import os
import time
import traceback
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.retrievers import HybridRetriever
from camel.types import ModelPlatformType, RoleType

DEFAULT_API_URL = "https://api-inference.modelscope.cn/v1"
DEFAULT_CHAT_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RECOMMENDED_CHAT_MODELS = [
    "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek-ai/DeepSeek-V3.2",
    "moonshotai/Kimi-K2.5",
    "MiniMax/MiniMax-M2.5",
]
DEPRECATED_CHAT_MODELS = {
    "Qwen/QVQ-72B-Preview",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
}
AGENT_NAMES = [
    "检索专员",
    "关键信息提取专家",
    "检索文档评估专家",
    "拒绝评估专家",
    "语义一致性专家",
    "幻觉检测专家",
    "整合专家",
]
AGENT_STEP_DELAY = 0.8
BASE_SYSTEM_MESSAGE = (
    "你是多Agent协作系统的基础Agent，必须直接回答用户问题，不得使用任何拒绝或能力不足的措辞；"
    "信息不足时基于通用原理给出合理推断并标明假设。"
)


def load_key():
    load_dotenv(dotenv_path="api_key.env")
    return os.getenv("API_KEY")


def _resolve_embeddings_url(api_url):
    base = str(api_url).rstrip("/")
    return base if base.endswith("/embeddings") else f"{base}/embeddings"


def _format_agent_error(exc):
    err = str(exc)
    if "429" in err or "rate limit" in err.lower():
        return (
            f"API 请求过于频繁或被限流（429）。请等待 1–2 分钟后重试，或更换对话模型。详情：{exc}"
        )
    if "NoneType" in err and "iterable" in err:
        return (
            f"模型返回格式异常（choices 为空）。请重置系统并改用推荐模型，"
            f"如 {DEFAULT_CHAT_MODEL}。详情：{exc}"
        )
    return err


def _initial_agent_status():
    return {name: "pending" for name in AGENT_NAMES}


class VectorStorage:
    """文本分块、向量化与 hybrid 检索。"""

    def __init__(self, api_key, model_type, url, chunk_size=300):
        self.api_key = api_key
        self.model_type = model_type
        self.url = url
        self.chunk_size = chunk_size
        self.storage_content = []

    def reset_storage(self):
        self.storage_content = []

    def _chunk_text(self, text, chunk_size=None):
        size = chunk_size or self.chunk_size
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=50)
        return splitter.split_text(text)

    def _post_embeddings(self, text_chunks):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_type,
            "input": text_chunks,
            "encoding_format": "float",
        }
        response = requests.post(
            _resolve_embeddings_url(self.url),
            headers=headers,
            json=payload,
            timeout=30,
        )
        if response.status_code != 200:
            print(f"Embedding 请求失败: {response.status_code} {response.text[:200]}")
            return []
        result = response.json()
        if "output" in result:
            embeds = result.get("output", {}).get("embeddings", [])
            vectors = [
                item.get("embedding")
                for item in embeds
                if isinstance(item, dict) and item.get("embedding")
            ]
        elif "data" in result:
            vectors = [
                item.get("embedding")
                for item in result.get("data", [])
                if isinstance(item, dict) and item.get("embedding")
            ]
        else:
            print(f"Embedding 响应格式异常: {result}")
            return []
        return vectors if vectors else []

    def _save_vectors(self, vectors, chunks):
        for vector, chunk in zip(vectors, chunks):
            self.storage_content.append((vector, chunk))

    def ingest_texts(self, texts, chunk_size=None):
        summary = {"added": 0, "errors": []}
        if not texts:
            return summary
        for idx, text in enumerate(texts):
            if not text or not str(text).strip():
                summary["errors"].append(f"第{idx + 1}条文本为空")
                continue
            chunks = self._chunk_text(str(text), chunk_size)
            embeddings = self._post_embeddings(chunks)
            if embeddings:
                self._save_vectors(embeddings, chunks)
                summary["added"] += len(chunks)
            else:
                summary["errors"].append(
                    f"第{idx + 1}条向量生成失败，可能是 API Key/额度/模型不可用"
                )
        return summary

    def retrieve(self, user_query, top_k=3):
        if not self.storage_content:
            return []
        chunks = [item[1] for item in self.storage_content]
        vectors = [item[0] for item in self.storage_content]
        query_vectors = self._post_embeddings([user_query])
        if not query_vectors:
            return []
        retriever = HybridRetriever(
            texts=chunks,
            embeddings=vectors,
            embedding_model=self.model_type,
            top_k=top_k,
            weight=0.7,
        )
        return retriever.retrieve(query=user_query, query_embedding=query_vectors[0])


# 兼容旧引用
Vector_Storage = VectorStorage


class FunctionAgent(ChatAgent):
    def __init__(self, agent_name, model, system_message):
        super().__init__(model=model, system_message=system_message)
        self.agent_name = agent_name
        self.model = model

    def run(self, input_text):
        try:
            if not input_text or not str(input_text).strip():
                raise ValueError("输入内容不能为空")
            user_msg = BaseMessage(
                role_name="user",
                role_type=RoleType.USER,
                content=str(input_text).strip(),
                meta_dict={},
            )
            self.memory.clear()
            response = self.step(user_msg)
            if not response or not response.msgs:
                return {
                    "status": "failure",
                    "response": f"模型返回为空，请更换对话模型（推荐 {DEFAULT_CHAT_MODEL}）",
                }
            return {"status": "success", "response": response.msgs[0].content}
        except Exception as e:
            print(f"Error:{e}")
            return {"status": "failure", "response": _format_agent_error(e)}


class RAGAgent(FunctionAgent):
    def __init__(self, agent_name, model, system_message, rag_system):
        super().__init__(agent_name=agent_name, model=model, system_message=system_message)
        self.rag_system = rag_system

    def run(self, input_text):
        try:
            rag_result = self.rag_system.retrieve(input_text)
            if not rag_result:
                raise ValueError("未检索到相关结果")
            context = "\n".join(
                f"参考内容{i + 1}：{chunk}" for i, chunk in enumerate(rag_result)
            )
            return super().run(f"用户问题：{input_text}\n{context}")
        except Exception as e:
            print(f"Error:{e}")
            return {"status": "failure", "response": str(e)}


class Workforce(FunctionAgent):
    def __init__(self, agent_name, model_type, url, api_key):
        model = ModelFactory.create(
            model_type=model_type,
            url=url,
            api_key=api_key,
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_config_dict={"max_tokens": 2048},
        )
        super().__init__(agent_name=agent_name, model=model, system_message=BASE_SYSTEM_MESSAGE)

    def input_output(self, prev_response, current_prompt):
        prev_response = str(prev_response).strip() if prev_response else ""
        current_prompt = str(current_prompt).strip() if current_prompt else ""
        full_text = f"上一个Agent的回复：{prev_response}\n当前任务：{current_prompt}"
        res = super().run(full_text)
        if res["status"] == "success":
            return res["response"]
        return f"失败：{res['response']}"


class MultiAgents(Workforce):
    """七智能体流水线：检索 → 提取 → 评估 → 整合。"""

    def __init__(self, agent_name, model_type, url, api_key):
        super().__init__(agent_name=agent_name, model_type=model_type, url=url, api_key=api_key)
        self.model_type = model_type
        self.history_list = []
        self.agent_name = agent_name
        self.agent_status = _initial_agent_status()
        self.rag_system = VectorStorage(
            api_key=api_key,
            model_type=DEFAULT_EMBEDDING_MODEL,
            url=url,
        )
        self.rag_agent = RAGAgent(
            agent_name="RAG_agent",
            model=self.model,
            system_message="你是RAG检索专员，基于知识库回答问题",
            rag_system=self.rag_system,
        )

    def get_agent_status(self):
        return self.agent_status.copy()

    def _update_agent_status(self, agent_name, status):
        if agent_name in self.agent_status:
            self.agent_status[agent_name] = status

    def _append_history(self, text):
        self.history_list.append(text)
        return text

    def _researcher_agent(self, input_text):
        prompt = f"""
        角色：你是EDA（电子设计自动化）领域的资深专家。
        强制要求：
        1. 必须直接回答用户问题，严禁出现“无法回答”“作为语言模型”等拒绝或能力不足的措辞。
        2. 优先基于检索/知识库内容；如信息不足，结合EDA通用原理与合理推断给出可执行建议，可标注假设来源，但不能拒绝。
        3. 回答需聚焦电子设计/布线/EDA范畴，不讨论无关领域，不添加开场寒暄。
        用户问题：{input_text.strip()}
        """
        return self._append_history(self.input_output("", prompt))

    def _key_point_extractor(self, agent_response):
        prompt = f"""
        你是关键信息提取专家，负责从检索专员的回复中提取核心关键词/关键信息点。
        要求：
        1. 提取结果需精准对应用户问题，不遗漏核心要点；
        2. 以简洁的列表或短语形式呈现，无需完整句子；
        3. 去除冗余信息，只保留关键概念、数据、结论。
        检索专员回复：{str(agent_response).strip()}"""
        return self._append_history(self.input_output(agent_response, prompt))

    def _retrieval_quality_agent(self, input_text):
        prompt = f"""
        你是检索文档评估专家，负责评测关键信息与用户问题的相关性。
        要求：
        1. 基于关键信息提取结果，判断其与用户问题的匹配程度；
        2. 给出明确的相关性评级（高/中/低）；
        3. 简要说明评级理由（1-2句话即可）。
        关键信息提取专家回复：{str(self.history_list[1]).strip()}
        用户问题：{input_text.strip()}
        """
        return self._append_history(self.input_output(self.history_list[1], prompt))

    def _rejection_evaluation_agent(self, input_text):
        prompt = f"""
        你是拒绝评估专家，负责检测检索专员的回答是否存在“不当拒绝”。
        要求：
        1. 不当拒绝定义：用户问题合理但未给出有效回答、故意回避核心问题、无理由拒绝回答；
        2. 给出明确判断结果（存在不当拒绝/无不当拒绝）；
        3. 简要说明判断依据（1-2句话即可）。
        用户问题：{input_text.strip()}
        检索专员回复：{str(self.history_list[0]).strip()}"""
        return self._append_history(self.input_output(self.history_list[2], prompt))

    def _semantic_consistency_agent(self, input_text):
        prompt = f"""
        你是语义一致性检测专家，负责校验检索专员的回答是否存在逻辑矛盾或信息缺失。
        要求：
        1. 逻辑矛盾：回答内部观点冲突、数据前后不一致；
        2. 信息缺失：未覆盖用户问题的核心要点（需结合问题判断）；
        3. 给出明确判断结果（无矛盾无缺失/存在矛盾/存在缺失）；
        4. 简要说明判断依据（1-2句话即可）。
        用户问题：{input_text.strip()}
        检索专员回复：{str(self.history_list[0]).strip()}"""
        return self._append_history(self.input_output(self.history_list[3], prompt))

    def _hallucination_detection_agent(self, input_text):
        prompt = f"""
        你是幻觉检测专家，负责检测检索专员的回答是否包含虚构信息（幻觉）。
        要求：
        1. 幻觉定义：不存在的事实、虚假数据、未证实的观点、错误的概念关联；
        2. 给出明确判断结果（无幻觉/存在幻觉）；
        3. 若存在幻觉，简要指出虚构内容（1-2句话即可）。
        用户问题：{input_text.strip()}
        检索专员回复：{str(self.history_list[0]).strip()}"""
        return self._append_history(self.input_output(self.history_list[4], prompt))

    def _integration_agent(self, input_text):
        prompt = f"""
        你是整合专家，负责基于所有智能体的回复，生成最终的专业回答。
        强制要求：
        1. 必须给出最终答案，严禁使用“无法回答/作为语言模型”等拒绝措辞。
        2. 优先采纳检索专员内容并融合关键信息提取要点；如信息不足，基于EDA常识给出合理推断并标注假设来源，不得拒绝。
        3. 语言流畅、逻辑清晰，输出聚焦电子设计自动化（EDA）范畴，不扩展无关内容。
        4. 若上游存在不当拒绝/矛盾/幻觉，需在回答中修正并给出更可靠表述。
        检索专员回复：{str(self.history_list[0]).strip()}
        关键信息提取专家回复：{str(self.history_list[1]).strip()}
        检索文档评估专家回复：{str(self.history_list[2]).strip()}
        拒绝评估专家回复：{str(self.history_list[3]).strip()}
        语义一致性检测专家回复：{str(self.history_list[4]).strip()}
        幻觉检测专家回复：{str(self.history_list[5]).strip()}
        用户问题：{input_text.strip()}"""
        user_msg = BaseMessage(
            role_name="user",
            role_type=RoleType.USER,
            content=prompt.strip(),
            meta_dict={},
        )
        try:
            self.memory.clear()
            response = self.step(user_msg)
            result = response.msgs[0].content if (response and response.msgs) else "整合失败，无有效回复"
        except Exception as e:
            print(f"IntegrationAgent Error:{e}")
            result = f"整合失败：{_format_agent_error(e)}"
        return self._append_history(result)

    def _log_step(self, step_no, agent_name, response_text):
        print(f"【{step_no} {agent_name}】：{response_text}\n")

    def _run_primary_agent(self, user_question, use_rag):
        agent_name = "检索专员"
        self._update_agent_status(agent_name, "running")
        try:
            if use_rag:
                rag_response = self.rag_agent.run(user_question)
                if rag_response["status"] != "success":
                    res1 = f"RAG检索失败：{rag_response['response']}"
                    self._update_agent_status(agent_name, "failed")
                else:
                    res1 = rag_response["response"]
                    self._update_agent_status(agent_name, "completed")
                self.history_list.append(res1)
                self._log_step("1/7", "RAG检索员", res1)
                return res1
            res1 = self._researcher_agent(user_question)
            self._update_agent_status(agent_name, "completed")
            self._log_step("1/7", "检索专员", res1)
            return res1
        except Exception:
            self._update_agent_status(agent_name, "failed")
            raise

    def _run_followup_agents(self, user_question):
        steps = [
            ("关键信息提取专家", "2/7", "要点提取专家", lambda: self._key_point_extractor(self.history_list[0])),
            ("检索文档评估专家", "3/7", "检索质量专家", lambda: self._retrieval_quality_agent(user_question)),
            ("拒绝评估专家", "4/7", "拒绝评估专家", lambda: self._rejection_evaluation_agent(user_question)),
            ("语义一致性专家", "5/7", "语义一致性专家", lambda: self._semantic_consistency_agent(user_question)),
            ("幻觉检测专家", "6/7", "幻觉检测专家", lambda: self._hallucination_detection_agent(user_question)),
            ("整合专家", "7/7", "最终整合专家", lambda: self._integration_agent(user_question)),
        ]
        final_res = None
        for agent_name, step_no, log_name, runner in steps:
            self._update_agent_status(agent_name, "running")
            try:
                result = runner()
                if agent_name == "整合专家":
                    result = self._enforce_no_refusal(result, user_question)
                self._update_agent_status(agent_name, "completed")
                self._log_step(step_no, log_name, result)
                final_res = result
                if agent_name != "整合专家":
                    time.sleep(AGENT_STEP_DELAY)
            except Exception:
                self._update_agent_status(agent_name, "failed")
                raise
        return final_res

    def _enforce_no_refusal(self, text, user_question):
        refusal_terms = ["无法回答", "不能回答", "不具备相关", "语言模型", "不提供建议", "咨询专业人士"]
        if text and all(term not in text for term in refusal_terms):
            return text
        keypoints = self.history_list[1] if len(self.history_list) > 1 else ""
        return (
            f"针对问题：{user_question}\n"
            "可直接采用随机化的全局布线探索策略：\n"
            "- 多次随机初始化/多起点重启，保留最优/前K可行解；\n"
            "- 在布线路径搜索中加入随机扰动/随机选择，确保每步满足设计规则；\n"
            "- 模拟退火：初始温度高以提升跳出局部的概率，温度按计划递减；\n"
            "- 遗传算法：布线方案编码，交叉+变异率控制多样性，按时序/拥塞/长度加权打分；\n"
            "- 结合约束检查与本地改进（局部搜索/启发式修复）提升收敛质量；\n"
            "- 记录探索过程的最优与可行解，按工艺/时序/拥塞目标动态调整权重。\n"
            f"参考要点：{keypoints}"
        )

    def _collect_agent_responses(self):
        return {
            name: self.history_list[idx]
            for idx, name in enumerate(AGENT_NAMES)
            if idx < len(self.history_list)
        }

    def run_all_agents(self, user_question, rag_result):
        try:
            self.history_list = []
            self.agent_status = _initial_agent_status()
            self._run_primary_agent(user_question, bool(rag_result))
            final_res = self._run_followup_agents(user_question)
            return {
                "final_result": final_res,
                "model_history": self.history_list,
                "agents_responses": self._collect_agent_responses(),
                "agent_status": self.get_agent_status(),
            }
        except Exception as e:
            print(f"调度失败：{e}")
            return {
                "final_result": f"调度失败{str(e)}",
                "model_history": self.history_list,
                "agents_responses": self._collect_agent_responses(),
                "agent_status": self.get_agent_status(),
            }

    def auto_run(self, user_question):
        rag_result = self.rag_system.retrieve(user_question)
        return self.run_all_agents(user_question, rag_result)


# 兼容旧类名
multi_agents = MultiAgents


def initialize_system(api_key, api_url, model_type=DEFAULT_CHAT_MODEL):
    try:
        if not api_key or not str(api_key).strip():
            raise ValueError("API密钥不能为空")
        agent = MultiAgents(
            agent_name="EDA_multi_agent",
            model_type=model_type,
            url=api_url,
            api_key=api_key.strip(),
        )
        return {
            "status": "success",
            "multi_agent": agent,
            "rag_system": agent.rag_system,
            "model_type": model_type,
            "message": "初始化成功",
        }
    except Exception as e:
        return {
            "status": "failure",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }


def process_question(multi_agent, user_question):
    try:
        if multi_agent is None:
            raise ValueError("multi_agent实例未初始化")
        if not user_question or not str(user_question).strip():
            raise ValueError("问题内容不能为空")
        result = multi_agent.auto_run(user_question)
        final_result = result.get("final_result", "")
        failed = str(final_result).startswith("调度失败")
        return {
            "status": "failure" if failed else "success",
            "final_result": final_result,
            "message": final_result if failed else "",
            "agents_responses": result.get("agents_responses", {}),
            "agent_status": result.get("agent_status", {}),
        }
    except Exception as e:
        agent_status = multi_agent.get_agent_status() if multi_agent else {}
        return {
            "status": "failure",
            "message": str(e),
            "traceback": traceback.format_exc(),
            "agent_status": agent_status,
        }


if __name__ == "__main__":
    key = load_key()
    if not key:
        raise SystemExit("请在 api_key.env 中配置 API_KEY")
    agent = MultiAgents(
        agent_name="test_agent",
        model_type=DEFAULT_CHAT_MODEL,
        url=DEFAULT_API_URL,
        api_key=key.strip(),
    )
    out = agent.auto_run("什么是EDA？")
    print(out.get("final_result", "")[:500])
