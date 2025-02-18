import dashscope
from dashscope.audio.tts import SpeechSynthesizer
import pyaudio
import sys
from dotenv import load_dotenv
import os
import time

# 加载环境变量
load_dotenv()
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

class TTSCallback:
    def __init__(self):
        self._player = None
        self._stream = None
        self._is_finished = False

    def on_open(self):
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=48000,
            output=True)

    def on_complete(self):
        self._is_finished = True

    def on_error(self, response):
        print(f'语音合成失败: {str(response)}')
        self._is_finished = True

    def on_close(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._player:
            self._player.terminate()

    def on_event(self, result):
        if result.get_audio_frame() is not None:
            self._stream.write(result.get_audio_frame())

def text_to_speech(text):
    callback = TTSCallback()
    callback.on_open()
    
    try:
        result = SpeechSynthesizer.call(
            model='sambert-zhiting-v1',
            text=text,
            sample_rate=48000,
            format='pcm',
            callback=callback
        )
        
        # 等待合成完成
        while not callback._is_finished:
            time.sleep(0.1)
            
    except Exception as e:
        print(f'语音合成出错: {str(e)}')
    finally:
        # 等待一下确保音频播放完成
        time.sleep(1)
        callback.on_close()

if __name__ == "__main__":
    # 测试文本
    test_text = "你好，我是语音助手。很高兴为你服务。"
    text_to_speech(test_text) 