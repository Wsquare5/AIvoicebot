from test_realtime_asr import ASRCallback, Recognition
from test_llm import LLMService
from test_tts import text_to_speech
import asyncio
import dashscope
from dotenv import load_dotenv
import os
import sys

# 加载环境变量
load_dotenv()
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

def select_role():
    while True:
        print("\n请选择你想要交谈的对象：")
        print("1. 小甜 - 26岁 甜美可人的女友")
        print("2. 雅雅 - 28岁 优雅知性的女友")
        print("3. 米米 - 25岁 热情活力的女友")
        print("4. 安娜 - 32岁 成熟魅力的女友")
        
        choice = input("请输入数字(1-4)选择: ").strip()
        
        role_map = {
            '1': 'sweet',
            '2': 'elegant',
            '3': 'passionate',
            '4': 'mature'
        }
        
        if choice in role_map:
            return role_map[choice]
        else:
            print("无效的选择，请重新输入...")

async def voice_chat():
    # 通过输入选择角色
    role = select_role()
    llm = LLMService(role=role)
    
    try:
        print('\n开始对话，说"再见"结束对话...')
        
        while True:  # 主对话循环
            try:  # 添加内部 try-except 来处理单次对话的错误
                callback = ASRCallback()
                print("\n请说话...")
                callback.on_open()
                
                recognition = Recognition(
                    model='paraformer-realtime-v2',
                    format='pcm',
                    sample_rate=16000,
                    callback=callback
                )
                
                recognition.start()
                await asyncio.sleep(1)
                
                while callback.is_running:
                    if callback and callback.stream:
                        try:
                            audio_data = callback.stream.read(3200, exception_on_overflow=False)
                            recognition.send_audio_frame(audio_data)
                            callback.check_silence()
                            await asyncio.sleep(0.1)
                        except Exception as e:
                            print(f'读取音频数据时出错: {str(e)}')
                            break
                
                if callback.current_text:
                    user_input = callback.current_text
                    print(f"\n用户说: {user_input}")
                    
                    if "再见" in user_input:
                        print("\n对话结束，谢谢使用！")
                        recognition.stop()
                        callback.cleanup()
                        return  # 使用 return 代替 sys.exit(0)
                    
                    response = await llm.get_response(user_input)
                    if response:
                        print(f"\nAI回复: {response}")
                        text_to_speech(response)
                        await asyncio.sleep(0.5)
                
                recognition.stop()
                callback.cleanup()
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"单轮对话出错: {str(e)}")
                # 继续下一轮对话，而不是退出程序
                await asyncio.sleep(1)
                continue
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生严重错误: {str(e)}")
    finally:
        if 'recognition' in locals():
            recognition.stop()
        if 'callback' in locals():
            callback.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(voice_chat())
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序异常: {str(e)}")
    finally:
        sys.exit(0)