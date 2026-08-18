# -*- coding: utf-8 -*-
# @Time    : 2026/8/18  13:03
# @Author  : loveuav02
# @File    : LLM_sdk.py
# @Software: jupyter notebook
# @Describe:
# -*- encoding:utf-8 -*-
import os #导入python标准库，用于操作系统相关功能，在这里读取APIKEY
from openai import OpenAI #导入 OpenAI 官方 Python SDK 的客户端类，为了调用大模型，pip install --upgrade "openai>=1.0"
import re #Python 正则表达式库。用于文本匹配、提取、替换。
import airsim_wrapper #无人机sdk封装

#火山云引擎，其他云平台也可以
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_API_KEY = "" # your api key，visit https://volcengine.com/L/GDhZ-EE4RrY/ 点击api接入
MODEL = "doubao-seed-2-1-turbo-260628" #接入doubao-seed2.1turbo

aw = airsim_wrapper.AirSimWrapper()

class AirSimAgent:
    #引入system输入的prompt和knowledge输入的prompt
    def __init__(self, system_prompts="system_prompts/system_prompt_cn.txt", knowledge_prompt="kg_prompts/kg_prompt_cn.txt", chat_history=[]):
        #llm client
        self.client = OpenAI(
            base_url = BASE_URL,
            api_key = ARK_API_KEY,
        )

        # 先对话列表，全局变量
        self.chat_history = []

        # 系统提示词，读取并加入对话记录
        sys_prompt = open(system_prompts, "r", encoding="utf8").read()
        self.chat_history.append(
            {
                "role": "system",
                "content": sys_prompt,
            }
        )

        # 知识库，并加入对话记录，通过聊天函数，加入知识库
        kg_prompt = open(knowledge_prompt, "r", encoding="utf8").read()
        self.ask(kg_prompt) #输入知识prompt

    # 调用chat api，包含历史记录，多轮对话
    def ask(self, prompt):
        # 加入用户输入的prompt
        self.chat_history.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        completion = self.client.chat.completions.create(
            model=MODEL,
            messages=self.chat_history,  # chat_history[-10:0]
            temperature=0.1,
        )

        #h回复
        content = completion.choices[0].message.content

        # 加入机器人回复，相当于保存全部的历史记录，多轮对话
        self.chat_history.append(
            {
                "role": "assistant",
                "content": content,
            }
        )

        return content

    # 从llm回答的文本中提取python代码
    def extract_python_code(self, content):
        """
        Extracts the python code from a response.
        :param content:
        :return:
        """
        code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)
        code_blocks = code_block_regex.findall(content)
        if code_blocks:
            full_code = "\n".join(code_blocks)

            if full_code.startswith("python"):
                full_code = full_code[7:]

            return full_code
        else:
            return None

    def process(self, command,run_python_code=False):
        #step 1,  用户输入指令到LLLM
        response = self.ask(command)

        #step 2, 从LLM生成的文本中提取python代码
        python_code = self.extract_python_code(response)

        #step 3, 执行提取到的python代码
        if run_python_code and python_code:
            exec(python_code)
        return python_code



