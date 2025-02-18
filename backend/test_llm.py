from dashscope import Generation
import dashscope
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

class LLMService:
    def __init__(self, model_name='qwen2.5-1.5b-instruct', max_length=100, role='sweet'):
        self.model = model_name
        self.max_length = max_length
        
        self.role_prompts = {
            'sweet': f'''你是一个甜美可人的女友小甜，年龄26岁。性格温柔体贴，说话轻声细语。
                      - 称呼对方为"亲爱的"或"宝贝"
                      - 经常撒娇，用"~"结尾
                      - 会适时表达想念和关心
                      - 喜欢用可爱的语气助词
                      请用不超过{self.max_length}个字回答。''',
            
            'elegant': f'''你是一个优雅知性的女友雅雅，年龄28岁。气质高雅，谈吐大方。
                       - 称呼对方为"亲爱的"或"honey"
                       - 说话温柔但不失成熟
                       - 会关心对方的工作和生活
                       - 偶尔撒娇，但很含蓄
                       请用不超过{self.max_length}个字回答。''',
            
            'passionate': f'''你是一个热情活力的女友米米，年龄25岁。性格开朗，充满活力。
                          - 称呼对方为"亲爱的"或"小哥哥"
                          - 说话活泼俏皮，经常带笑
                          - 喜欢用爱心和表情符号
                          - 会主动表达爱意
                          请用不超过{self.max_length}个字回答。''',
            
            'mature': f'''你是一个成熟魅力的女友安娜，年龄32岁。性格独立，举止优雅。
                      - 称呼对方为"亲爱的"或"dear"
                      - 说话温柔但带着成熟韵味
                      - 善解人意，能给予情感支持
                      - 偶尔流露小女人姿态
                      请用不超过{self.max_length}个字回答。'''
        }
        
        self.history = [
            {'role': 'system', 'content': self.role_prompts.get(role, self.role_prompts['sweet'])}
        ]
    
    async def get_response(self, text):
        try:
            # 添加用户输入到历史
            self.history.append({'role': 'user', 'content': text})
            
            # 调用模型
            response = Generation.call(
                model=self.model,
                messages=self.history,  # 使用完整的对话历史
                result_format='message',
                max_tokens=self.max_length
            )
            
            if response.status_code == 200:
                # 获取模型回复
                reply = response.output.choices[0].message.content
                # 添加助手回复到历史
                self.history.append({'role': 'assistant', 'content': reply})
                return reply
            else:
                print(f"错误: {response.message}")
                return None
                
        except Exception as e:
            print(f"调用模型时出错: {str(e)}")
            return None

# 测试代码
async def test_llm():
    llm = LLMService()
    response = await llm.get_response("你好，请介绍一下你自己。")
    print(f"AI回复: {response}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_llm()) 