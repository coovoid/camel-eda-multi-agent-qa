import multiprocessing
from camel.agents import ChatAgent
from camel.embeddings import OpenAICompatibleEmbedding
from camel.storages import QdrantStorage
from camel.retrievers import HybridRetriever
from camel.models import ModelFactory
from camel.types import ModelPlatformType, RoleType
import requests
import json
import os
import time
import pickle
import tempfile
import traceback
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from camel.messages import BaseMessage


#设置处理api_key函数
def load_key():
    load_dotenv(dotenv_path = "api_key.env")
    api_key = os.getenv('API_KEY')
    return api_key

api_key = load_key()
#初始化环境
def create_environment():
    api_key = load_key()
    try:
        if api_key is None:
            raise ValueError(f"api_key不能为空")
        elif len(api_key) < 10:
            raise ValueError(f"api_key长度有误")
    except ValueError as e:
        print(f"Error:{e}")
        return
    print("api_key检验通过")


#初始化RAG模型
class Init_Model:          
    def __init__(self,
                 api_key,
                 model_type ,
                 url,
                 output_dim
                 ):
                                        #headers：请求头，格式：{“Authorization”：f“Bearer {}”}
        self.url = url
        self.api_key = api_key
        self.output_dim = output_dim
        self.model_type = model_type
        try:
            self.embedding_instance = OpenAICompatibleEmbedding(
                model_type=model_type,
                api_key=f"Bearer {api_key}",
                url=url
            )
            print("连接成功")

        except Exception as e:
            print(f"{e},尝试重新连接")
            raise
#存储向量
class Vector_Storage(Init_Model):
    def __init__(self,storage_path,api_key,model_type,url,output_dim):
        super().__init__(api_key = api_key,model_type = model_type,url = url,output_dim = output_dim)
        self.api_key = api_key
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = os.path.join(self.temp_dir.name, storage_path)
    
        self.storage_content = []
        with open(self.storage_path,'wb')as f:
            pickle.dump(self.storage_content,f)
    def reset_storage(self):
        """清空向量存储。"""
        self.storage_content = []
        with open(self.storage_path,'wb') as f:
            pickle.dump(self.storage_content,f)
#检索内容分块
    def Content_Chunking(self,user_content,chunk_size):
        splitter = RecursiveCharacterTextSplitter(chunk_size) 
        text_chunks = splitter.split_text(user_content)
        return text_chunks

    def Post_Embeddings(self,model_type,text_chunks,url,api_key,):
        headers = {"Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json" }
        json_data = {
            "model": model_type,
            "input": text_chunks,
        }

        response = requests.post(url = url,
                      headers = headers,
                      json = json_data,
                      timeout = 30
        )
        status_code = response.status_code
        try:
            if status_code == 200:
                result = response.json()
                if not isinstance(result, dict):
                    raise ValueError(f"返回内容非JSON对象：{result}")
                if "output" in result:
                    output = result.get("output", {})
                    if not isinstance(output, dict):
                        raise ValueError(f"output字段格式异常：{output}")
                    embeds = output.get("embeddings", [])
                    if not isinstance(embeds, list):
                        raise ValueError(f"embeddings字段格式异常：{embeds}")
                    vectors = [item.get("embedding") for item in embeds if isinstance(item, dict) and "embedding" in item]
                elif "data" in result:
                    data = result.get("data", [])
                    if not isinstance(data, list):
                        raise ValueError(f"data字段格式异常：{data}")
                    vectors = [item.get("embedding") for item in data if isinstance(item, dict) and "embedding" in item]
                else:
                    raise ValueError(f"响应缺少embeddings字段：{result}")
                if not vectors:
                    raise ValueError(f"未提取到任何embedding：{result}")
                print("连接成功，文件已转化为向量")
                return vectors
            elif status_code in[400,404,500]:
                raise ConnectionError(f"连接错误，状态码{status_code}")
            elif status_code is None:
                raise ConnectionError("错误，无响应")
        except Exception as e:
            try:
                detail = response.text
            except Exception:
                detail = ""
            print(f"失败:{e}, 详情:{detail}")
            return[]
    def ingest_texts(self, texts, chunk_size=300):
        """批量写入文本，完成分块、向量化与存储，返回新增统计和错误信息。"""
        summary = {"added": 0, "errors": []}
        try:
            if not texts:
                return summary
            for idx, text in enumerate(texts):
                if not text or not str(text).strip():
                    summary["errors"].append(f"第{idx+1}条文本为空")
                    continue
                chunks = self.Content_Chunking(str(text), chunk_size=chunk_size)
                embeddings = self.Post_Embeddings(
                    model_type=self.model_type,
                    text_chunks=chunks,
                    url=self.url,
                    api_key=self.api_key
                )
                if embeddings:
                    self.Vectors_Save(embeddings, chunks)
                    summary["added"] += len(chunks)
                else:
                    summary["errors"].append(f"第{idx+1}条向量生成失败，可能是API Key/额度/模型不可用")
            return summary
        except Exception as e:
            print(f"ingest_texts失败：{e}")
            summary["errors"].append(str(e))
            return summary
    #存储向量
    def Vectors_Save(self,vectors,text_chunks):
        vector_num = len(vectors)
        for i in range(vector_num):
            vector = vectors[i]
            chunk = text_chunks[i]
            self.storage_content.append((vector,chunk))

        with open(self.storage_path,'wb') as f:
            pickle.dump(self.storage_content,f)
            return self.storage_content
    #RAG检索
    def RAG_Retriever(self,user_query):
        chunks = [item[1] for item in self.storage_content]
        vectors = [item[0] for item in self.storage_content]
        if not chunks or not vectors:
            print("向量库为空")
            return []
    
        retriever = HybridRetriever(
            texts =  chunks,
            embeddings = vectors,
            embedding_model = self.model_type,
            top_k = 3,
            weight =0.7
        )

        query_vector = self.Post_Embeddings(        
            model_type=self.model_type,
            text_chunks=[user_query],                    
            url=self.url,
            api_key=self.api_key
        )
        if not query_vector or len(query_vector) == 0:
            print("查询向量生成失败")
            return []
        query_vector = query_vector[0]
        similar_chunks = retriever.retrieve(query = user_query,
                                            query_embedding = query_vector)
        return similar_chunks      

    #启动RAG
    def start_RAG(self,user_input):
        try:
            if not user_input.strip():  
                print("用户输入为空")
                return []
            chunks = self.Content_Chunking(user_input,chunk_size=100)
            embeddings = self.Post_Embeddings(
            model_type=self.model_type,
            text_chunks=chunks,
            url=self.url,
            api_key=self.api_key
                        )
            storage_content = self.Vectors_Save(embeddings,chunks)
            similar_chunks = self.RAG_Retriever(user_input)

            return similar_chunks
        except Exception as e:
            print(f"start_RAG失败：{e}")
            return []


#创建一个Function类，初始化agent
class FunctionAgent(ChatAgent):
    def __init__(self,agent_name,model,system_message):
        super().__init__(model = model,system_message=system_message)
        self.agent_name = agent_name
        self.model = model
        self.set_system_message = system_message
#运行输入内容
    def run(self,input_text):
        try:
            if input_text is not None and input_text.strip() != "":
                user_msg = BaseMessage(
                role_name="user",
                role_type=RoleType.USER, 
                content=input_text.strip(),
                meta_dict={}
                )
                self.memory.clear()
                response = self.step(user_msg)
                response_text = response.msgs[0].content if response.msgs else ""
                return {"status":"success","response":response_text}
            else:
                raise ValueError("输入内容不能为空")

        except Exception as e:
                print(f"Error:{e}")
                return {"status":"failure","response":str(e)}


#创建一个RAGAgent类，负责RAG检索
class RAGAgent(FunctionAgent):
    def __init__(self,agent_name,model,system_message,rag_system):
        super().__init__(agent_name= agent_name,
                         model = model,
                         system_message =system_message
                         )
        self.rag_system = rag_system
#启动检索
    def run(self,input_text):
        try:
            rag_result = self.rag_system.RAG_Retriever(input_text)
            if not rag_result:
                raise ValueError("未检索到相关结果")
            else:
                context = "\n".join([f"参考内容{num+1}：{chunk}" for num, chunk in enumerate(rag_result)])
                # 注意：join后面是括号，再接列表推导式
                prompt = f"用户问题：{input_text}\n{context}"
                response = super().run(prompt)
                return response
        except Exception as e:
            print(f"Error:{e}")
            return {"status": "failure", "response": str(e)}
#创建工作组类
class Workforce(FunctionAgent):
    def __init__(self,agent_name,model_type,url,api_key):
        self.model = ModelFactory.create(
            model_type = model_type,
            url = url,
            api_key = api_key,
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_config_dict={"max_tokens": 2048}
        )
        super().__init__(   model=self.model,
                            agent_name=agent_name,
                            system_message="你是多Agent协作系统的基础Agent，必须直接回答用户问题，不得使用任何拒绝或能力不足的措辞；信息不足时基于通用原理给出合理推断并标明假设。")
    def input_output(self, prev_response, current_prompt):
        try:
            prev_response = str(prev_response).strip() if prev_response else ""
            current_prompt = str(current_prompt).strip() if current_prompt else ""
            full_text = f"上一个Agent的回复：{prev_response.strip()}\n当前任务：{current_prompt.strip()}"
            #将上一个agent的回复和现在的prompt拼接在一起
            res = super().run(full_text)
            self.agent_response = res["response"] if res["status"] == "success" else f"失败：{res['response']}"
            return self.agent_response
        except Exception as e:
            print(f"Error:{e}")
            return f"处理失败：{str(e)}"
#创建所有agent，继承workforce
class multi_agents(Workforce):
    """所有agents
    ResearcherAgent	先检索再生成初始回答 ->这个是在用户没有提供文档时检索自己学习到的内容的
    KeyPointExtractorAgent	提取关键信息点（文档 / 真实答案 / 生成答案）
    RetrievalQualityAgent	评估检索文档的相关性（高 / 中 / 低）
    RejectionEvaluationAgent	检测是否 “不当拒绝”（该答不答 / 不该答却答）
    SemanticConsistencyAgent	校验语义一致性（无矛盾 / 无缺失）
    HallucinationDetectionAgent	检测 “幻觉”（虚构信息）
    IntegrationAgent	汇总所有专家结果，生成最终专业回答"""


    def __init__(self,agent_name,model_type,url,api_key):
        super().__init__(agent_name = agent_name,model_type=model_type, url=url, api_key=api_key)
        self.model_type = model_type
        self.history_list = []
        self.agent_name = agent_name
        #添加agent的几个状态状态: "pending", "running", "completed", "failed"
        self.agent_status = {
            "检索专员": "pending",
            "关键信息提取专家": "pending",
            "检索文档评估专家": "pending",
            "拒绝评估专家": "pending",
            "语义一致性专家": "pending",
            "幻觉检测专家": "pending",
            "整合专家": "pending"
        }
    #初始化RAG系统
        self.RAG_system = Vector_Storage(storage_path = "RAG_storage_path",
                                         api_key = api_key,
                                         model_type = "Qwen/Qwen3-Embedding-0.6B",
                                         url = url,
                                         output_dim = 1024)
    #写入agent
        self.RAG_agent_instance = RAGAgent(
            agent_name="RAG_agent",
            model=self.model,
            system_message="你是RAG检索专员，基于知识库回答问题",
            rag_system=self.RAG_system
        )
    
    def get_agent_status(self):
        """获取当前所有agent的状态"""
        return self.agent_status.copy()
    
    def _update_agent_status(self, agent_name, status):
        """更新agent状态: pending, running, completed, failed"""
        if agent_name in self.agent_status:
            self.agent_status[agent_name] = status
#检索专员
    def ResearcherAgent(self,input_text):
        prompt =f"""
        角色：你是EDA（电子设计自动化）领域的资深专家。
        强制要求：
        1. 必须直接回答用户问题，严禁出现“无法回答”“作为语言模型”等拒绝或能力不足的措辞。
        2. 优先基于检索/知识库内容；如信息不足，结合EDA通用原理与合理推断给出可执行建议，可标注假设来源，但不能拒绝。
        3. 回答需聚焦电子设计/布线/EDA范畴，不讨论无关领域，不添加开场寒暄。
        用户问题：{input_text.strip()}
        """
        self.agent_response = super().input_output("",prompt)
        self.history_list.append(self.agent_response)
        return self.agent_response
#关键信息提取专家
    def KeyPointExtractorAgent(self,agent_response):
        agent_response_str = str(agent_response).strip()
        prompt = f"""
        你是关键信息提取专家，负责从检索专员的回复中提取核心关键词/关键信息点。
        要求：
        1. 提取结果需精准对应用户问题，不遗漏核心要点；
        2. 以简洁的列表或短语形式呈现，无需完整句子；
        3. 去除冗余信息，只保留关键概念、数据、结论。
        检索专员回复：{agent_response_str.strip()}"""
        self.agent_response = super().input_output(agent_response,prompt)
        self.history_list.append(self.agent_response)
        return self.agent_response
#检索文档评估专家
    def RetrievalQualityAgent(self,input_text):
        prompt = f"""
        你是检索文档评估专家，负责评测关键信息与用户问题的相关性。
        要求：
        1. 基于关键信息提取结果，判断其与用户问题的匹配程度；
        2. 给出明确的相关性评级（高/中/低）；
        3. 简要说明评级理由（1-2句话即可）。
        关键信息提取专家回复：{str(self.history_list[1]).strip()}
        用户问题：{input_text.strip()}
        """
        self.agent_response = super().input_output(self.history_list[1],prompt)
        self.history_list.append(self.agent_response)
        return self.agent_response
#拒绝评估专家
    def RejectionEvaluationAgent(self,input_text):
        prompt = f"""
        你是拒绝评估专家，负责检测检索专员的回答是否存在“不当拒绝”。
        要求：
        1. 不当拒绝定义：用户问题合理但未给出有效回答、故意回避核心问题、无理由拒绝回答；
        2. 给出明确判断结果（存在不当拒绝/无不当拒绝）；
        3. 简要说明判断依据（1-2句话即可）。
        用户问题：{input_text.strip()}
        检索专员回复：{str(self.history_list[0]).strip()}"""
        self.agent_response = super().input_output(self.history_list[2], prompt)
        self.history_list.append(self.agent_response)
        return self.agent_response
#语义一致性检测专家
    def SemanticConsistencyAgent(self,input_text):
        prompt = f"""
        你是语义一致性检测专家，负责校验检索专员的回答是否存在逻辑矛盾或信息缺失。
        要求：
        1. 逻辑矛盾：回答内部观点冲突、数据前后不一致；
        2. 信息缺失：未覆盖用户问题的核心要点（需结合问题判断）；
        3. 给出明确判断结果（无矛盾无缺失/存在矛盾/存在缺失）；
        4. 简要说明判断依据（1-2句话即可）。
        用户问题：{input_text.strip()}
        检索专员回复：{str(self.history_list[0]).strip()}"""
        self.agent_response = super().input_output(self.history_list[3], prompt)
        self.history_list.append(self.agent_response)
        return self.agent_response
#幻觉检测专家
    def HallucinationDetectionAgent(self,input_text):
        prompt = f"""
        你是幻觉检测专家，负责检测检索专员的回答是否包含虚构信息（幻觉）。
        要求：
        1. 幻觉定义：不存在的事实、虚假数据、未证实的观点、错误的概念关联；
        2. 给出明确判断结果（无幻觉/存在幻觉）；
        3. 若存在幻觉，简要指出虚构内容（1-2句话即可）。
        用户问题：{input_text.strip()}
        检索专员回复：{str(self.history_list[0]).strip()}"""
        self.agent_response = super().input_output(self.history_list[4], prompt)
        self.history_list.append(self.agent_response)
        return self.agent_response
#整合专家
    def IntegrationAgent(self,input_text):
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
            meta_dict={}
        )
        self.memory.clear()          #清除记忆，保证智能体的单轮对话独立性
        response = self.step(user_msg)
        self.agent_response = response.msgs[0].content if (response and response.msgs) else "整合失败，无有效回复"
        self.agent_response = self._enforce_no_refusal(self.agent_response, input_text)
        self.history_list.append(self.agent_response)
        return self.agent_response

#定义启动所有agent的函数
    def _log_step(self, step_no, agent_name, response_text):
        """统一打印日志，避免重复代码。"""
        print(f"【{step_no} {agent_name}】：{response_text}\n")

    def _run_primary_agent(self, user_question, use_rag):
        """根据是否启用RAG决定首个Agent的执行，并写入历史。"""
        agent_name = "检索专员"
        self._update_agent_status(agent_name, "running")
        try:
            if use_rag:
                rag_response = self.RAG_agent_instance.run(user_question)
                if rag_response["status"] != "success":
                    res1 = f"RAG检索失败：{rag_response['response']}"
                    self._update_agent_status(agent_name, "failed")
                else:
                    res1 = rag_response["response"]
                    self._update_agent_status(agent_name, "completed")
                self.history_list.append(res1)
                self._log_step("1/7", "RAG检索员", res1)
                return res1
            res1 = self.ResearcherAgent(user_question)
            self._update_agent_status(agent_name, "completed")
            self._log_step("1/7", "检索专员", res1)
            return res1
        except Exception as e:
            self._update_agent_status(agent_name, "failed")
            raise

    def _run_followup_agents(self, user_question):
        """执行固定顺序的后续六个智能体。"""
        # 关键信息提取专家
        agent_name = "关键信息提取专家"
        self._update_agent_status(agent_name, "running")
        try:
            res2 = self.KeyPointExtractorAgent(self.history_list[0])
            self._update_agent_status(agent_name, "completed")
            self._log_step("2/7", "要点提取专家", res2)
        except Exception as e:
            self._update_agent_status(agent_name, "failed")
            raise
        
        # 检索文档评估专家
        agent_name = "检索文档评估专家"
        self._update_agent_status(agent_name, "running")
        try:
            res3 = self.RetrievalQualityAgent(user_question)
            self._update_agent_status(agent_name, "completed")
            self._log_step("3/7", "检索质量专家", res3)
        except Exception as e:
            self._update_agent_status(agent_name, "failed")
            raise
        
        # 拒绝评估专家
        agent_name = "拒绝评估专家"
        self._update_agent_status(agent_name, "running")
        try:
            res4 = self.RejectionEvaluationAgent(user_question)
            self._update_agent_status(agent_name, "completed")
            self._log_step("4/7", "拒绝评估专家", res4)
        except Exception as e:
            self._update_agent_status(agent_name, "failed")
            raise
        
        # 语义一致性专家
        agent_name = "语义一致性专家"
        self._update_agent_status(agent_name, "running")
        try:
            res5 = self.SemanticConsistencyAgent(user_question)
            self._update_agent_status(agent_name, "completed")
            self._log_step("5/7", "语义一致性专家", res5)
        except Exception as e:
            self._update_agent_status(agent_name, "failed")
            raise
        
        # 幻觉检测专家
        agent_name = "幻觉检测专家"
        self._update_agent_status(agent_name, "running")
        try:
            res6 = self.HallucinationDetectionAgent(user_question)
            self._update_agent_status(agent_name, "completed")
            self._log_step("6/7", "幻觉检测专家", res6)
        except Exception as e:
            self._update_agent_status(agent_name, "failed")
            raise
        
        # 整合专家
        agent_name = "整合专家"
        self._update_agent_status(agent_name, "running")
        try:
            final_res = self.IntegrationAgent(user_question)
            final_res = self._enforce_no_refusal(final_res, user_question)
            self._update_agent_status(agent_name, "completed")
            self._log_step("7/7", "最终整合专家", final_res)
            return final_res
        except Exception as e:
            self._update_agent_status(agent_name, "failed")
            raise

    def _enforce_no_refusal(self, text, user_question):
        """若回答出现拒绝措辞，生成兜底技术建议，确保有输出。"""
        refusal_terms = ["无法回答", "不能回答", "不具备相关", "语言模型", "不提供建议", "咨询专业人士"]
        if text and all(term not in text for term in refusal_terms):
            return text
        keypoints = self.history_list[1] if len(self.history_list) > 1 else ""
        fallback = (
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
        return fallback

    def _collect_agent_responses(self):
        """构建智能体名称到回复的映射，供前端展示。"""
        agent_order = [
            "检索专员",
            "关键信息提取专家",
            "检索文档评估专家",
            "拒绝评估专家",
            "语义一致性专家",
            "幻觉检测专家",
            "整合专家",
        ]
        responses = {}
        for idx, name in enumerate(agent_order):
            if idx < len(self.history_list):
                responses[name] = self.history_list[idx]
        return responses

    def run_all_agents(self,user_question,rag_result):
        try:
            self.history_list = []
            # 重置所有agent状态为pending
            for agent_name in self.agent_status.keys():
                self.agent_status[agent_name] = "pending"
            use_rag = bool(rag_result)
            self._run_primary_agent(user_question, use_rag)
            final_res = self._run_followup_agents(user_question)
            return {
                "final_result": final_res,
                "model_history": self.history_list,
                "agents_responses": self._collect_agent_responses(),
                "agent_status": self.get_agent_status()
            }

        except Exception as e:
            print(f"调度失败：{e}")
            return {
                "final_result":f"调度失败{str(e)}",
                "model_history": self.history_list,
                "agents_responses": self._collect_agent_responses(),
                "agent_status": self.get_agent_status()
            }
#定义自动转换模式（有rag检索内容就转rag模式，无就转普通对话模式）
    def auto_run(self, user_question):
        rag_result = self.RAG_system.RAG_Retriever(user_question)
        if rag_result:
            return self.run_all_agents(user_question, rag_result=rag_result)
        else:
            return self.run_all_agents(user_question, rag_result=None)
# 初始化系统，供前端调用
def initialize_system(api_key, api_url, 
                      model_type="Qwen/QVQ-72B-Preview",
                      embedding_model="Qwen/Qwen3-Embedding-0.6B"):
    """根据前端配置初始化多智能体与RAG实例。"""
    try:
        if not api_key or not str(api_key).strip():
            raise ValueError("API密钥不能为空")
        multi_agent = multi_agents(
            agent_name="EDA_multi_agent",
            model_type=model_type,
            url=api_url,
            api_key=api_key,
        )
        return {
            "status": "success",
            "multi_agent": multi_agent,
            "rag_system": multi_agent.RAG_system,
            "message": "初始化成功"
        }
    except Exception as e:
        return {
            "status": "failure",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

# 处理问题入口，供前端调用
def process_question(multi_agent, user_question):
    """统一封装问题处理逻辑，返回前端需要的结构。"""
    try:
        if multi_agent is None:
            raise ValueError("multi_agent实例未初始化")
        if not user_question or not str(user_question).strip():
            raise ValueError("问题内容不能为空")

        result = multi_agent.auto_run(user_question)
        return {
            "status": "success",
            "final_result": result.get("final_result", ""),
            "agents_responses": result.get("agents_responses", {}),
            "agent_status": result.get("agent_status", {})
        }
    except Exception as e:
        # 获取当前状态（即使失败也可能有部分agent完成了）
        agent_status = multi_agent.get_agent_status() if multi_agent else {}
        return {
            "status": "failure",
            "message": str(e),
            "traceback": traceback.format_exc(),
            "agent_status": agent_status
        }
#主程序入口
if __name__ == "__main__":
    #初始化环境
    create_environment()
    
    multi_agent = multi_agents(
        agent_name="test_agent",
        model_type="Qwen/QVQ-72B-Preview",
        url="https://api-inference.modelscope.cn/v1",
        api_key=api_key,
    )

    RAG = Vector_Storage(
        storage_path = "RAG_storage_path",
        api_key = api_key,
        model_type = "Qwen/Qwen3-Embedding-0.6B",
        url = "https://api-inference.modelscope.cn/v1",
        output_dim = 1024,
    )
    #这里暂时没有，之后要替换成用户实际上传的PDF文档
    knowledge_text = """
   
    """
    
    RAG.start_RAG(user_input=knowledge_text)
    result = multi_agent.auto_run(user_question="什么是EDA？")


















































