import pyaudio
import time
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
import dashscope
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')  # 从环境变量获取API key

# 确保设置了API key
if not dashscope.api_key:
    raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")


class ASRCallback(RecognitionCallback):
    def __init__(self):
        self.mic = None
        self.stream = None
        self.is_running = False
        self.last_speech_time = time.time()
        self.silence_threshold = 3.5  # 增加到3.5秒
        self.current_text = ""
        self.last_length = 0
        self.final_text = ""
        self.last_update_time = time.time()
        print("ASRCallback 初始化完成")

    def on_open(self) -> None:
        try:
            print('初始化音频设备...')
            self.mic = pyaudio.PyAudio()
            
            # 列出所有可用的音频输入设备
            info = self.mic.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            device_id = None
            
            # 查找默认输入设备
            default_device = self.mic.get_default_input_device_info()
            device_id = default_device['index']
            print(f"使用默认输入设备: {default_device['name']}")
            
            print('打开音频流...')
            self.stream = self.mic.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=device_id,
                frames_per_buffer=3200  # 调回较大的缓冲区
            )
            print('音频流打开成功')
            self.is_running = True
            
        except Exception as e:
            print(f'初始化音频设备时出错: {str(e)}')
            self.cleanup()
            raise

    def on_event(self, result: RecognitionResult) -> None:
        try:
            sentence = result.get_sentence()
            if sentence and 'text' in sentence:
                # 更新最后一次语音时间
                current_time = time.time()
                self.last_speech_time = current_time
                
                # 获取新识别的文本
                new_text = sentence["text"]
                
                # 更新文本，并确保有足够的处理时间
                if new_text:
                    self.current_text = new_text
                    self.final_text = new_text
                    # 只有在距离上次更新超过0.5秒时才打印
                    if current_time - self.last_update_time > 0.5:
                        print(f'\r当前识别: {new_text}', end='', flush=True)
                        self.last_update_time = current_time

        except Exception as e:
            print(f'处理识别结果时出错: {str(e)}')

    def check_silence(self):
        """检查是否超过沉默阈值"""
        current_time = time.time()
        # 只有在最后一次更新后超过1秒才检查静默
        if (current_time - self.last_update_time > 1.0 and 
            current_time - self.last_speech_time > self.silence_threshold):
            if self.final_text:
                print(f'\n最终识别结果: {self.final_text}')
            self.is_running = False

    def on_close(self) -> None:
        # 确保在关闭前输出最终结果
        if self.final_text and self.final_text != self.current_text:
            print(f'\n最终识别结果: {self.final_text}')
        print('WebSocket连接关闭，清理音频设备...')
        self.cleanup()
        print('音频设备已关闭')

    def cleanup(self):
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f'关闭音频流时出错: {str(e)}')
        if self.mic:
            try:
                self.mic.terminate()
            except Exception as e:
                print(f'终止PyAudio时出错: {str(e)}')
        self.stream = None
        self.mic = None

def start_realtime_asr():
    callback = None
    recognition = None
    
    try:
        print("创建回调处理器...")
        callback = ASRCallback()
        
        print("初始化识别器...")
        recognition = Recognition(
            model='paraformer-realtime-v2',
            format='pcm',
            sample_rate=16000,
            callback=callback
        )
        
        print("启动识别服务...")
        recognition.start()
        
        print("开始录音(3秒无语音将自动停止)...")  # 更新提示信息
        
        # 等待WebSocket连接建立
        time.sleep(2)
        
        while callback.is_running:
            if callback and callback.stream:
                try:
                    audio_data = callback.stream.read(3200, exception_on_overflow=False)
                    recognition.send_audio_frame(audio_data)
                    callback.check_silence()  # 检查是否需要自动停止
                    time.sleep(0.1)
                except Exception as e:
                    print(f'读取音频数据时出错: {str(e)}')
                    break
            else:
                print("音频流未初始化")
                break
                
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        print("清理资源...")
        if recognition:
            try:
                recognition.stop()
            except Exception as e:
                print(f'停止识别服务时出错: {str(e)}')
        if callback:
            callback.cleanup()

if __name__ == "__main__":
    start_realtime_asr() 