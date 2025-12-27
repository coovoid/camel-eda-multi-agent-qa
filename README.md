# camel-eda-multi-agent-qa
本多智能体RAG 代码基于camel框架，运行前先依照如下步骤：  

请确保你的系统已安装Python 3.10+  

1、使用win+R打开Windows 命令提示符，输入`pip install uv`  
2、输入`git clone -b v0.2.38 https://github.com/camel-ai/camel.git` 克隆Github仓库  
3、切换到项目目录：`cd camel`  
4、`uv venv .venv --python=3.10` # （可选）  
5、激活 camel 虚拟环境，出现类似(camel-ai-py3.10) C:\camel>中左侧的（虚拟环境）代表激活成功  
   mac/linux `source .venv/bin/activate`  
   windows `.venv\Scripts\activate`  
6、从源代码安装依赖环境，大约需要 90 秒 `uv pip install -e ".[all]"`  

# 调用模型
1、注册魔搭社区 https://modelscope.cn/my/overview  
2、绑定阿里云账号（必需）  
3、打开魔搭左侧栏的访问令牌，复制自己的令牌，并把它粘贴到api_key.env中
