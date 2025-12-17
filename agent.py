import streamlit as st
import time as ts
from datetime import datetime
import base64
import json
import sys
import os
from typing import Dict, List, Any
import requests
import asyncio
import concurrent.futures
from contextlib import contextmanager
import threading
import traceback
from io import BytesIO

def extract_file_content(uploaded_file):
    """将上传文件转为纯文本，用于RAG索引。失败返回(None, error_msg)。"""
    name = uploaded_file.name.lower()
    suffix = name.split(".")[-1] if "." in name else ""
    try:
        data = uploaded_file.getvalue()
        if suffix in ["txt", "md"]:
            return data.decode("utf-8", errors="ignore"), None
        if suffix == "json":
            try:
                obj = json.loads(data.decode("utf-8", errors="ignore"))
                return json.dumps(obj, ensure_ascii=False, indent=2), None
            except Exception:
                return data.decode("utf-8", errors="ignore"), None
        if suffix == "pdf":
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(BytesIO(data))
                text = "\n".join([page.extract_text() or "" for page in reader.pages])
                return text, None
            except Exception as e:
                return None, f"PDF解析失败: {e}"
        if suffix == "docx":
            try:
                import docx
                doc = docx.Document(BytesIO(data))
                text = "\n".join([p.text for p in doc.paragraphs])
                return text, None
            except Exception as e:
                return None, f"DOCX解析失败: {e}"
        if suffix in ["xlsx", "xls"]:
            try:
                import pandas as pd
                sheets = pd.read_excel(BytesIO(data), sheet_name=None)
                text = "\n".join([df.to_csv(index=False) for df in sheets.values()])
                return text, None
            except Exception as e:
                return None, f"Excel解析失败: {e}"
        return None, "不支持的文件类型"
    except Exception as e:
        return None, f"读取失败: {e}"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multi_agent_backend import initialize_system, process_question, Vector_Storage


if 'multi_agent' not in st.session_state:
    st.session_state.multi_agent = None
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    
st.set_page_config(
    page_title="EDA多智能体整合问答系统",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .stButton button {
        width: 100%;
    }
    .agent-card {
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
    }
    .status-active { background-color: #d1fae5; color: #065f46; }
    .status-inactive { background-color: #f3f4f6; color: #6b7280; }
    .uploaded-file {
        background-color: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .chat-user {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 1rem 1rem 0 1rem;
        margin: 0.5rem 0;
    }
    .chat-assistant {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 1rem 1rem 1rem 0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'agents_activated' not in st.session_state:
    st.session_state.agents_activated = ["检索专员", "关键信息提取专家", "检索文档评估专家", "拒绝评估专家", "语义一致性专家", "幻觉检测专家", "整合专家"]
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'rag_knowledge' not in st.session_state:
    st.session_state.rag_knowledge = ""
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'api_config' not in st.session_state:
    st.session_state.api_config = {
        "selected_api": "deepseek",
        "custom_api": False,
        "api_key": "",
        "api_url": "https://api-inference.modelscope.cn/v1"
    }

#侧边栏
with st.sidebar:
# 系统配置
    st.subheader("系统配置")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("初始化系统", use_container_width=False, type="primary", 
                    help="初始化多智能体系统和RAG引擎"):
            with st.spinner("正在初始化系统..."):
                # 获取API配置
                api_key = st.session_state.api_config.get("api_key", "")
                api_url = st.session_state.api_config.get("api_url", "https://api-inference.modelscope.cn/v1")
                
                # 如果界面没有输入API密钥，尝试从文件加载
                if not api_key:
                    env_file_path = "api_key.env"
                    if os.path.exists(env_file_path):
                        with open(env_file_path, "r") as f:
                            content = f.read()
                            if "API_KEY=" in content:
                                api_key = content.split("API_KEY=")[1].strip()
                        st.info("从api_key.env文件加载API密钥")
                
                if not api_key:
                    st.error("请先配置API密钥")
                else:
                    # 初始化后端系统
                    init_result = initialize_system(api_key, api_url)
                    if init_result["status"] == "success":
                        st.session_state.multi_agent = init_result["multi_agent"]
                        st.session_state.rag_system = init_result["rag_system"]
                        st.session_state.system_initialized = True
                        st.success("系统初始化完成！")
                    else:
                        st.error(f"系统初始化失败: {init_result['message']}")
                        # 特别处理API认证错误
                        if "阿里云账户" in init_result['message']:
                            st.warning("解决方案：请访问 ModelScope 官网绑定您的阿里云账户，然后重新生成API密钥")
                        # 显示详细的错误信息
                        with st.expander("详细错误信息"):
                            st.code(init_result.get("traceback", "无详细追踪信息"))
                    ts.sleep(2)
                    st.rerun()
    
    with col2:
        if st.button("重置系统", use_container_width=True, type="secondary",
                    help="清空所有会话和数据"):
            st.session_state.clear()
            st.success("系统已重置")
            st.rerun()
    
    st.divider()
     # 系统状态显示
    st.subheader("系统状态:")
    # status_color = "✅" if st.session_state.system_initialized else "⭕"
    if st.session_state.system_initialized :    
        st.title("已就绪")
    else:
        st.title("未初始化")
    st.divider()
    
   
    
    
    # 智能体团队管理
    st.subheader("智能体团队管理")
    agents_options = ["检索专员", "关键信息提取专家", "检索文档评估专家", "拒绝评估专家", "语义一致性专家", "幻觉检测专家", "整合专家"]
    selected_agents = st.multiselect(
        "启用智能体",
        options=agents_options,
        default=st.session_state.agents_activated,
        placeholder="选择要启用的智能体...",
        help="选择参与问答流程的智能体成员"
    )
    st.session_state.agents_activated = selected_agents
    
    st.divider()
    
    
    # 系统信息
    st.subheader("系统信息")
    st.info(f"""
    - 对话记录: {len(st.session_state.chat_history)} 条
    - 知识库文档: {len(st.session_state.uploaded_files)} 个
    - 智能体数: {len(st.session_state.agents_activated)} 个
    - 最后更新: {datetime.now().strftime("%H:%M:%S")}
    """)


# 主界面

st.title("EDA多智能体整合问答系统")
st.caption("基于RAG技术的多智能体协作问答平台")

col_main, col_agents = st.columns([2, 1])

with col_main:
    # 聊天界面
    with st.container(border=True):
        st.subheader("对话")
        
        # 显示聊天历史
        chat_container = st.container(height=400, border=False)
        with chat_container:
            for i, chat in enumerate(st.session_state.chat_history):
                if chat["role"] == "user":
                    with st.chat_message("user"):
                        st.write(f"**您**: {chat['content']}")
                        if "timestamp" in chat:
                            st.caption(f"时间: {chat['timestamp']}")
                else:  
                    with st.chat_message("assistant"):
                        st.write(f"**系统回答**: {chat['content']}")
                        if "timestamp" in chat:
                            st.caption(f"时间: {chat['timestamp']}")
                        
                      
                        if "agents" in chat:
                            with st.expander("查看智能体分析详情", expanded=False):
                                for agent_name, response in chat["agents"].items():
                                    if agent_name in st.session_state.agents_activated:
                                        st.markdown(f"**{agent_name}**:")
                                        st.info(response)
                                        st.divider()
        
        # 输入区域
        with st.form(key="chat_form", clear_on_submit=True):
            col_input, col_submit = st.columns([5, 1])
            with col_input:
                user_input = st.text_area(
                    "请输入您的问题",
                    placeholder="例如：什么是EDA？EDA工具的主要功能有哪些？...",
                    label_visibility="collapsed",
                    height=100,
                    key="user_input"
                )
            
            with col_submit:
                st.write("")
                st.write("")
                submitted = st.form_submit_button(
                    "发送",
                    use_container_width=True,
                    disabled=not st.session_state.system_initialized or st.session_state.processing,
                    help="发送问题给智能体团队处理" if st.session_state.system_initialized else "请先初始化系统"
                )
            
            # 文件上传
            uploaded_file = st.file_uploader(
                "",
                type=["pdf", "txt", "md", "docx", "xlsx", "json"],
                accept_multiple_files=True,
                help="上传文档将添加到RAG知识库中"
            )
            
            if uploaded_file:
                new_texts = []
                for file in uploaded_file:
                    if file.name not in [f["name"] for f in st.session_state.uploaded_files]:
                        content, err = extract_file_content(file)
                        if content is None:
                            st.warning(f"{file.name} 解析失败: {err}")
                            continue
                        st.session_state.uploaded_files.append({
                            "name": file.name,
                            "size": file.size,
                            "type": file.type,
                            "upload_time": datetime.now().strftime("%H:%M"),
                            "content": content
                        })
                        new_texts.append(content)
                        st.success(f"已上传: {file.name}")
                # 写入向量库
                if new_texts:
                    if st.session_state.rag_system is not None:
                        summary = st.session_state.rag_system.ingest_texts(new_texts)
                        if isinstance(summary, dict):
                            added = summary.get("added", 0) or 0
                            errors = summary.get("errors", [])
                        else:
                            added = int(summary) if summary is not None else 0
                            errors = []
                        if added > 0:
                            st.info(f"已索引 {added} 条文本片段")
                        if errors:
                            st.warning("部分文件未成功索引：\n" + "\n".join(errors))
                    else:
                        st.warning("系统未初始化，无法索引文档")
                      
        
        # 处理用户提交
        if submitted and user_input.strip() and st.session_state.system_initialized:
            # 添加用户消息到历史
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input.strip(),
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            # 显示处理状态
            processing_placeholder = st.empty()
            with processing_placeholder:
                with st.spinner("多智能体协作处理中...(响应可能需要几分钟，请耐心等待)"):
                    st.session_state.processing = True
                   
                    if st.session_state.multi_agent is not None:
                        try:
                            # 处理问题
                            result = process_question(st.session_state.multi_agent, user_input.strip())
                            
                            if result["status"] == "success":
                                # 获取智能体响应
                                agents_responses = result.get("agents_responses", {})
                                
                                st.session_state.processing = False
                                
                                # 添加助手消息到历史
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": result["final_result"],
                                    "agents": {k: v for k, v in agents_responses.items() 
                                              if k in st.session_state.agents_activated},
                                    "timestamp": datetime.now().strftime("%H:%M")
                                })
                            else:
                                st.session_state.processing = False
                                # 显示详细的错误信息
                                st.error(f"处理失败: {result['message']}")
                                # 特别处理API认证错误
                                if "阿里云账户" in result['message']:
                                    st.warning("解决方案：请访问 ModelScope 官网绑定您的阿里云账户，然后重新生成API密钥")
                                # 在开发阶段，显示更多调试信息
                                with st.expander("详细错误信息"):
                                    st.code(str(result), language="json")
                        except Exception as e:
                            st.session_state.processing = False
                            st.error(f"处理过程中发生异常: {str(e)}")
                            # 特别处理API认证错误
                            if "阿里云账户" in str(e):
                                st.warning("解决方案：请访问 ModelScope 官网绑定您的阿里云账户，然后重新生成API密钥")
                            # 显示详细的异常信息
                            with st.expander("异常详情"):
                                st.code(traceback.format_exc(), language="python")
                    else:
                        st.session_state.processing = False
                        st.error("系统未正确初始化，请重新初始化系统")
                    
            processing_placeholder.empty()
            st.rerun()

with col_agents:
    with st.expander("使用说明", expanded=False):
        st.markdown("""
    ### 系统使用指南
    
    **1. 初始化系统**
    - 点击侧边栏"初始化系统"按钮
    - 等待系统初始化完成（约2-3秒）
    
    **2. 配置API（可选）**
    - 选择API服务商
    - 如需自定义API，勾选"使用自定义API"
    - 填写API密钥和端点地址
    
    **3. 管理智能体团队**
    - 在侧边栏选择要启用的智能体
    - 系统默认启用全部7个智能体
    
    **4. 上传知识文档（RAG功能）**
    - 支持PDF、TXT、MD、Word等格式
    - 上传文档将自动添加到知识库
    - 可点击"重新索引"更新向量索引
    
    **5. 开始对话**
    - 在输入框输入问题
    - 可同时上传相关文档
    - 点击"发送"开始处理
    
    **6. 查看结果**
    - 查看最终回答
    - 点击"查看智能体分析详情"查看各智能体分析
    - 使用导出功能保存结果
    
    ### 智能体功能
    
    | 智能体 | 功能说明 |
    |--------|----------|
    | 检索专员 | 从知识库检索相关信息 |
    | 关键信息提取 | 提取检索结果核心要点 |
    | 检索质量评估 | 评估检索相关性 |
    | 拒绝行为检测 | 检查不当拒绝回答 |
    | 语义一致性 | 验证逻辑一致性 |
    | 幻觉检测 | 检测虚构/错误信息 |
    | 整合专家 | 生成最终专业回答 |
    
    ### 使用建议
    1. 对于专业问题，建议上传相关技术文档
    2. 复杂问题建议启用全部智能体
    3. 简单问题可只启用核心智能体
    4. 定期重新索引以保证检索质量
    """)
     # API配置
    with st.container(border=True):
        st.subheader("API配置")
        selected_api = st.selectbox(
            "选择API服务商",
            options=["硅基流动", "讯飞星火", "智谱AI", "其他"],
            index=0,
            help="选择要使用的模型API服务商"
        )
        st.session_state.api_config["selected_api"] = selected_api
        
        use_custom_api = st.checkbox("使用自定义API", value=False, 
                                    help="启用自定义API端点配置")
        st.session_state.api_config["custom_api"] = use_custom_api
        
        if use_custom_api:
            api_key = st.text_input(
                "API密钥",
                type="password",
                placeholder="请输入您的API密钥",
                help="请输入您的API访问密钥",
                value=st.session_state.api_config.get("api_key", "")
            )
            api_url = st.text_input(
                "API端点",
                value=st.session_state.api_config.get("api_url", "https://api-inference.modelscope.cn/v1"),
                placeholder="请输入API端点URL",
                help="API服务端点地址"
            )
            st.session_state.api_config["api_key"] = api_key
            st.session_state.api_config["api_url"] = api_url
            
            # 添加API配置说明
            st.info("提示：使用ModelScope API需要绑定阿里云账户")
        
    # 知识库管理
    with st.container(border=True):
        st.subheader(" 知识库管理")
        
        if st.session_state.uploaded_files:
            st.write("已上传文档:")
            for file in st.session_state.uploaded_files:
                file_size = f"{file['size']:,} bytes"
                if file['size'] > 1024:
                    file_size = f"{file['size']/1024:.1f} KB"
                if file['size'] > 1024 * 1024:
                    file_size = f"{file['size']/(1024 * 1024):.1f} MB"
                
                st.markdown(f"""
                <div class="uploaded-file">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{file['name']}</strong><br>
                            <small>类型: {file['type']} | 大小: {file_size}</small>
                        </div>
                        <button style="background: none; border: none; color: #ff4444; cursor: pointer;" 
                                onclick="alert('删除功能需在后端实现')">删除</button>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("重新索引", use_container_width=True, 
                           help="重新构建向量索引"):
                    with st.spinner("正在重新索引..."):
                        # 如果有RAG系统实例，重新索引
                        if st.session_state.rag_system is not None:
                            texts = [f.get("content","") for f in st.session_state.uploaded_files if f.get("content")]
                            if texts:
                                st.session_state.rag_system.reset_storage()
                                summary = st.session_state.rag_system.ingest_texts(texts)
                                if isinstance(summary, dict):
                                    added = summary.get("added", 0) or 0
                                    errors = summary.get("errors", [])
                                else:
                                    added = int(summary) if summary is not None else 0
                                    errors = []
                                if added > 0:
                                    st.success(f"知识库索引更新完成，共 {added} 条片段")
                                if errors:
                                    st.warning("部分文本未成功索引：\n" + "\n".join(errors))
                            else:
                                st.warning("未找到可索引的文本内容")
                        else:
                            st.warning("系统未初始化，无法重新索引")
            
            with col_btn2:
                if st.button("清空知识库", use_container_width=True, type="secondary",
                           help="清空所有上传的文档"):
                    st.session_state.uploaded_files = []
                    st.success("知识库已清空")
                    st.rerun()
        else:
            st.info("暂无上传文档")
            st.caption("上传文档以启用RAG检索功能")
            
    with st.container(border=True):
        st.subheader("导出工具")
        if st.button("导出对话JSON", use_container_width=True, 
                    help="导出完整的对话记录为JSON格式"):
            if st.session_state.chat_history:
                chat_json = json.dumps(st.session_state.chat_history, 
                                    ensure_ascii=False, indent=2)
                
                st.download_button(
                    label="下载JSON文件",
                    data=chat_json,
                    file_name=f"eda_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )


        if st.button("导出回答文本", use_container_width=True,
                    help="导出所有系统回答为文本格式"):
            if st.session_state.chat_history:
                final_responses = [chat["content"] for chat in st.session_state.chat_history 
                                if chat["role"] == "assistant"]
                if final_responses:
                    txt_content = "\n\n" + "="*50 + "\n\n".join(
                        [f"回答 {i+1}:\n{resp}" for i, resp in enumerate(final_responses)]
                    )
                    
                    st.download_button(
                        label="下载TXT文件",
                        data=txt_content,
                        file_name=f"eda_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )


        if st.button("导出分析报告", use_container_width=True,
                    help="导出智能体分析报告"):
            if st.session_state.chat_history:
                # 创建分析报告
                report = {
                    "系统信息": {
                        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "活跃智能体": st.session_state.agents_activated,
                        "对话总数": len(st.session_state.chat_history),
                        "知识文档数": len(st.session_state.uploaded_files)
                    },
                    "对话记录": st.session_state.chat_history
                }
                
                report_json = json.dumps(report, ensure_ascii=False, indent=2)
                
                st.download_button(
                    label="下载报告",
                    data=report_json,
                    file_name=f"eda_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )


        if st.button("清空对话", use_container_width=True, type="secondary",
                    help="清空当前对话记录"):
            if st.session_state.chat_history:
                st.session_state.chat_history = []
                st.success("对话记录已清空")
                st.rerun()

st.caption("EDA多智能体整合问答系统 v1.0 | 基于RAG技术的智能问答平台")
