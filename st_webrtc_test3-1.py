import threading
import time
from collections import deque
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoTransformerBase
from typing import List
import io
import wave
import asyncio
import cv2
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere.chat_models import ChatCohere
import base64
import keyboard
from gtts import gTTS
import os
#import speech_recognition as sr
from pydub import AudioSegment
import queue, pydub, tempfile
import whisper
import torch
import torchaudio
import torchvision
import re
from pydub.effects import low_pass_filter, high_pass_filter
from io import BytesIO

def init_page():
    st.set_page_config(
        page_title="Yas Chatbot(Webカメの画像、音声を表示)",
        page_icon="🤖"
    )
    st.header("Yas Chatbot 🤖")
    st.write("""Webカメラに移した画像についての問合せ、音声での入出力ができます。\n
             Webブラウザのカメラ,マイクのアクセスを許可する設定にしてください。""") 
    st.sidebar.title("Options")

def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    # clear_button が押された場合や message_history がまだ存在しない場合に初期化
    if clear_button or "message_history" not in st.session_state:
        st.session_state.message_history = [
            ("system", "You are a helpful assistant.")
        ]    
def select_model():
    # スライダーを追加し、temperatureを0から2までの範囲で選択可能にする
    # 初期値は0.0、刻み幅は0.01とする
    temperature = 0.0
    models = ( "GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro")
    model = st.sidebar.radio("大規模言語モデルを選択:", models)
       
    if model == "GPT-4o":  #"gpt-4o 'gpt-4o-2024-08-06'" 有料？、Best
        st.session_state.model_name = "gpt-4o"
        return ChatOpenAI(
            temperature=temperature,
            model=st.session_state.model_name,
            api_key= st.secrets.key.OPENAI_API_KEY,
            max_tokens=512,  #指定しないと短い回答になったり、途切れたりする。
            streaming=True,
        )
    elif model == "Claude 3.5 Sonnet": #コードがGood！！
        st.session_state.model_name = "claude-3-5-sonnet-20240620"
        return ChatAnthropic(
            temperature=temperature,
            model_name=st.session_state.model_name, 
            api_key= st.secrets.key.ANTHROPIC_API_KEY,
            max_tokens_to_sample=2048,  
            timeout=None,  
            max_retries=2,
            stop=None,  
        )
    elif model == "Gemini 1.5 Pro":
        st.session_state.model_name = "gemini-1.5-pro-latest"
        return ChatGoogleGenerativeAI(
            temperature=temperature,
            model=st.session_state.model_name,
            api_key= st.secrets.key.GOOGLE_API_KEY,
        )
#音声出力関数
def streaming_text_speak(llm_response):
    # 末尾の空白の数を確認
    trailing_spaces = len(llm_response) - len(llm_response.rstrip())
    print(f"末尾の空白の数: {trailing_spaces}")
    # 末尾の空白を削除
    cleaned_response = llm_response.rstrip()
    print(f"空白を除去した文字列: '{cleaned_response}'")
    # 句読点やスペースを基準に分割
    #復帰文字（\r）は、**キャリッジリターン（Carriage Return）**と呼ばれる特殊文字で、
    # ASCIIコード13（10進数）に対応します。主に改行の一部として使用される制御文字です。
    split_response = re.split(r'([\r\n!-;=:、。 \?]+)', llm_response) 
    #split_response = re.split(r'([;:、。 ]+😊🌟🚀🎉)', llm_response)  #?はなくてもOK
    split_response = [segment for segment in split_response if segment.strip()]  # 空要素を削除
    print(split_response)
    # AIメッセージ表示
    with st.chat_message("ai"):
        response_placeholder = st.empty()
        # ストリーミング応答と音声出力処理
        partial_text = ""
        
        for segment in split_response:
            if segment.strip():  # 空文字列でない場合のみ処理
                partial_text += segment
                response_placeholder.markdown(f"**{partial_text}**")  # 応答のストリーミング表示
                # gTTSで音声生成（部分テキスト）
                try:
                    # アスタリスクやその他の発音に不要な文字を削除
                    cleaned_segment = re.sub(r'[\*!-]', '', segment)
                    tts = gTTS(cleaned_segment, lang="ja")  # 音声化
                    audio_buffer = BytesIO()
                    tts.write_to_fp(audio_buffer)  # バッファに書き込み
                    audio_buffer.seek(0)

                    # pydubで再生速度を変更
                    audio = AudioSegment.from_file(audio_buffer, format="mp3")
                    audio = audio._spawn(audio.raw_data, overrides={
                        "frame_rate": int(audio.frame_rate * 1.3)  # 1.5倍速
                    }).set_frame_rate(audio.frame_rate)
                    audio_buffer.close()

                    # 音質調整
                    audio = audio.set_frame_rate(44100)  # サンプリングレート
                    audio = audio + 5  # 音量を5dB増加
                    audio = audio.fade_in(500).fade_out(500)  # フェードイン・アウト
                    #audio = audio.low_pass_filter(3000)  # 高音域をカット
                    audio = low_pass_filter(audio, cutoff=900)  # 高音域をカット
                    # ベースブースト（低音域を強調）
                    low_boost = low_pass_filter(audio,1000).apply_gain(10)
                    audio = audio.overlay(low_boost)

                    # バッファに再エクスポート
                    output_buffer = BytesIO()
                    audio.export(output_buffer, format="mp3")
                    output_buffer.seek(0)

                    # 音声の再生
                    # チェックする文字列
                    if re.search(r"\n\n", segment):
                        print("文字列に '\\n\\n' が含まれています。")
                        #time.sleep(1) 
                    #else:
                        #print("文字列に '\\n\\n' は含まれていません。")
                    #st.audio(audio_buffer, format="audio/mp3",autoplay = True)
                    # 音声データをBase64にエンコード
                    audio_base64 = base64.b64encode(output_buffer.read()).decode()
                    audio_buffer.close()  # バッファをクローズ
                    a=len(audio_base64)
                    #print(a)
                     # HTMLタグで音声を自動再生（プレイヤー非表示、再生速度調整）
                    audio_html = f"""
                   <audio id="audio-player" autoplay style="display:none;">
                        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)

                except Exception as e:
                    #print(f"音声生成エラー: {e}")
                    pass
                try:
                    time.sleep(a*0.00004)  # テキストストリーミング速度に同期
                except Exception as e:
                  time.sleep(2) 
                    
def speak(text):
    #st.write("音声ファイルを作成します。")
    # テキストを音声に変換
    tts = gTTS(text=text, lang='ja')
    output_file = "output.mp3"
    tts.save(output_file)
    st.write("音声ファイルに保存しました。")
    # 音声ファイルを提供
    audio_file = open(output_file, "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3", start_time=0,autoplay=True)
    #st.write("音声再生が完了しました。")
    # 音声ファイルを削除
    audio_file.close()
    os.remove(output_file)
    #st.write("音声再生が完了し、ファイルは削除されました。")

#  LLM問答関数   
async def query_llm(user_input,frame):
    try:
        if st.session_state.input_img == "有":    
            # 画像を適切な形式に変換（例：base64エンコードなど）
            # 画像をエンコード
            encoded_image = cv2.imencode('.jpg', frame)[1]
            # 画像をBase64に変換
            base64_image = base64.b64encode(encoded_image).decode('utf-8')  
            #image = f"data:image/jpeg;base64,{base64_image}"
        
        if st.session_state.model_name ==  "keep_gpt-4o":
            llm = st.session_state.llm  
            stream = llm.stream([
                    *st.session_state.message_history,
                    (
                        "user",
                        [
                            {
                                "type": "text",
                                "text": user_input
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "auto"
                                },
                            }
                        ]
                    )
                ])
            #response = chain.invoke(user_input)
         
        if st.session_state.model_name ==  "command-r-plus":
            print("st.session_state.model_name=",st.session_state.model_name)
            print(user_input)
            prompt = ChatPromptTemplate.from_messages(
                [
                    *st.session_state.message_history,
                     #("user", f"{user_input}:{base64_image}"),  #やっぱりだめ
                     ("user", f"{user_input}")
                ]
            )
            
            output_parser = StrOutputParser()
            chain = prompt | st.session_state.llm | output_parser
            #stream = chain.stream(user_input,base64_image)
            
            stream = chain.stream({"user_input":user_input,"base64_image": base64_image})
            print("stream=",stream)
            #response = chain.invoke(user_input)
            
        else:
            print("st.session_state.model_name=",st.session_state.model_name)
            if st.session_state.input_img == "有":
                prompt = ChatPromptTemplate.from_messages(
                    [
                        *st.session_state.message_history,
                        (
                            "user",
                            [
                                {
                                    "type": "text",
                                    "text": user_input
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                }
                            ],
                        ),
                    ]
                )
              
                output_parser = StrOutputParser()
                chain = prompt | st.session_state.llm | output_parser
                #stream = chain.stream(user_input,base64_image)
                stream = chain.stream({"user_input":user_input,"base64_image": base64_image})
                #print("stream=",stream)
                #response = chain.invoke(user_input)
            else:
                prompt = ChatPromptTemplate.from_messages(
                    [
                        *st.session_state.message_history,
                        (
                            "user",
                            [
                                {
                                    "type": "text",
                                    "text": user_input
                                }
                            ],
                        ),
                    ]
                )
                
            output_parser = StrOutputParser()
            chain = prompt | st.session_state.llm | output_parser
            if st.session_state.output_method == "音声":
                response = chain.invoke({"user_input":user_input})
                #speak(response)   #st.audio ok
                streaming_text_speak(response)
            else:    
                stream = chain.stream({"user_input":user_input})
            # LLMの返答を表示する  Streaming
                with st.chat_message('ai'):  
                    response =st.write_stream(stream) 
                           
            print(f"{st.session_state.model_name}=",response)
 
            # チャット履歴に追加
            st.session_state.message_history.append(("user", user_input))
            st.session_state.message_history.append(("ai", response))
        
            return response
    except StopIteration:
        # StopIterationの処理
        print("StopIterationが発生")
        pass

    user_input = ""
    base64_image = ""
    frame = ""   

def main():
    #st.header("Real Time Speech-to-Text with_video")
    #画面表示
    init_page()
    init_messages()
    #stで使う変数初期設定
    st.session_state.llm = select_model()
    st.session_state.input_method = ""
    st.session_state.user_input = ""
    st.session_state.result = ""
    st.session_state.frame = "" 
    st.session_state.text_output = ""
    st.session_state.audio_receiver_size = 4096

    col1, col2 ,col3= st.sidebar.columns(3)
     # 各列にボタンを配置
    with col1:
        # 入力方法の選択
        input_method = st.sidebar.radio("入力方法", ("テキスト", "音声"))
        st.session_state.input_method = input_method
    with col2:
        # 画像についての問合せ有無の選択
        input_img = st.sidebar.radio("  カメラ画像問合せ", ("有", "無"))
        st.session_state.input_img = input_img
    with col3:
        # 出力方法の選択
        output_method = st.sidebar.radio("出力方法", ("テキスト", "音声"))
        st.session_state.output_method = output_method
    

    
    # チャット履歴の表示 (第2章から少し位置が変更になっているので注意)
    for role, message in st.session_state.get("message_history", []):
        st.chat_message(role).markdown(message)
    #データ初期値
    user_input = ""
    base64_image = ""
    frame = ""    
    app_sst_with_video() 

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None
    def recv(self, frame):   
        self.frame = frame.to_ndarray(format="bgr24")
        return frame
        
async def process_audio(audio_data_bytes, sample_rate):
    temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(temp_audio_file, 'wb') as wf:
        wf.setnchannels(2)
        #wf.setnchannels(1)  # モノラル音声として記録　NG
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data_bytes)
    # 一時ファイルのパスを transcribe に渡す
    temp_audio_file_path = temp_audio_file.name 
    temp_audio_file.close()  
    # Whisperのモデルをロード
    model = whisper.load_model("small")  # モデルのサイズは適宜選択
    #base:74M,small:244M,medium,large
    # 音声をデコード
    try:
        # Whisperで音声をテキストに変換
        result = model.transcribe(temp_audio_file_path, language="ja")  # 日本語指定
        answer = result['text']
    finally:
        # 一時ファイルを削除
        os.remove(temp_audio_file_path)
    
        
    # テキスト出力が空、または空白である場合もチェック
    if answer == "" :
        print("テキスト出力が空")
        #return None 
    elif "ご視聴" in answer or "お疲れ様" in answer:
        print("テキスト出力が「ご視聴」、または「お疲れ様」を含む")
        #return None 
    else:
        print(answer)
        return answer

def save_audio(audio_segment: AudioSegment, base_filename: str) -> None:
    filename = f"{base_filename}_{int(time.time())}.wav"
    audio_segment.export(filename, format="wav")

def transcribe(audio_segment: AudioSegment, debug: bool = False) -> str:
    """
    OpenAIのWhisper ASRシステムを使用して音声セグメントを文字起こしします。
    引数:
        audio_segment (AudioSegment): 文字起こしする音声セグメント。
        debug (bool): Trueの場合、デバッグ目的で音声セグメントを保存します。
    戻り値:
        str: 文字起こしされたテキスト。
    """
    if debug:
        save_audio(audio_segment, "debug_audio")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        audio_segment.export(tmpfile.name, format="wav")
        # 一時ファイルのパスを指定
        audio = whisper.load_audio(tmpfile.name)
        audio = whisper.pad_or_trim(audio)
        # Whisperのモデルをロード
        model = whisper.load_model("small")  # モデルのサイズは適宜選択
        #base:74M,small:244M,medium,large
        # 音声をデコード
        result = model.transcribe(audio, language="ja")  # 日本語を指定
        answer = result['text']
        
        # テキスト出力が空、または空白である場合もチェック
        if len(answer) < 5 or "ご視聴" in answer or "お疲れ様" in answer:
            #print("テキスト出力が空")
            #print("transcribeルーチンのtext(answer)=",answer)
            return ""
        elif "見てくれてありがとう" in answer or "はっはっは" in answer:
            #print("テキスト出力が「ご視聴」、または「お疲れ様」を含む")
            return "" 
        elif "んんんんんん" in answer :
            #print("テキスト出力が「ご視聴」、または「お疲れ様」を含む")
            return "" 
    tmpfile.close()  
    os.remove(tmpfile.name)
    print("transcribeルーチンのtext(answer)=",answer)
    st.session_state.text_output = answer
    return answer

def frame_energy(frame):
    samples=[]
    # フレームのデータをnumpy配列として読み込み
    samples = np.frombuffer(frame.to_ndarray().tobytes(), dtype=np.int16)
    # NaN、正の無限大、負の無限大を0に置換
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    # 配列の長さが0の場合はエネルギーを0として返す
    if len(samples) == 0: 
        return 0.0
    # 負の値を絶対値に変換して処理 
    samples = np.abs(samples)
    try:
        #print("samples=",samples)
        energy = np.sqrt(np.mean(samples**2))
        #print("energy=",energy)  #50-90
        return energy  
    except Exception as e:
        #print(f"Error exporting audio: {e}")
        return 0.0

def frame_amplitude(audio_frame):
    samples = np.frombuffer(audio_frame.to_ndarray().tobytes(), dtype=np.int16)
    max_amplitude = np.max(np.abs(samples))
    #print("max_amplitude=",max_amplitude)
    return max_amplitude 

def process_audio_frames(audio_frames, sound_chunk, silence_frames, energy_threshold, amp_threshold):
    """
    音声フレームを順次処理し、無音フレームの数をカウント。
    無音フレームが一定数以上続いた場合、無音区間として処理し、後続の処理（例えば、音声認識のトリガー）に役立てます。
    この処理により、無音や音声の有無を正確に検出することができます。

    音声フレームのリストを処理します。 
    引数：
        audio_frames (list[VideoTransformerBase.Frame]): 処理する音声フレームのリスト。
        sound_chunk (AudioSegment): 現在のサウンドチャンク。
        silence_frames (int): 現在の無音フレームの数。
        energy_threshold (int): 無音検出に使用するエネルギーしきい値。
        amp_threshold:無音検出に使用する最大振幅しきい値。
    戻り値：
        tuple[AudioSegment, int]: 更新されたサウンドチャンクと無音フレームの数。
        
    """
        
    for audio_frame in audio_frames:
        sound_chunk = add_frame_to_chunk(audio_frame, sound_chunk)

        energy = frame_energy(audio_frame)
        amplitude = frame_amplitude(audio_frame)

        if energy < energy_threshold or amplitude < amp_threshold:
            silence_frames += 1 
            #無音のエネルギー又は最大振幅がしきい値以下である場合、無音フレームの数を1つ増やします。
        else:
            silence_frames = 0 
            #エネルギー又は最大振幅がしきい値を超える場合、無音フレームをリセットして0にします。

    return sound_chunk, silence_frames,energy,amplitude

def add_frame_to_chunk(audio_frame, sound_chunk):
    """
    オーディオフレームをサウンドチャンクに追加します。 
    引数：
        audio_frame (VideoTransformerBase.Frame): 追加するオーディオフレーム。
        sound_chunk (AudioSegment): 現在のサウンドチャンク。
    戻り値：
        AudioSegment: 更新されたサウンドチャンク。
    """
    sound = pydub.AudioSegment(
        data=audio_frame.to_ndarray().tobytes(),
        sample_width=audio_frame.format.bytes,
        frame_rate=audio_frame.sample_rate,
        channels=len(audio_frame.layout.channels),
    )
    sound_chunk += sound
    return sound_chunk

def handle_silence(sound_chunk, silence_frames, silence_frames_threshold):
    """
    オーディオストリーム内の無音を処理します。 
    引数：
        sound_chunk (AudioSegment): 現在のサウンドチャンク。
        silence_frames (int): 現在の無音フレームの数。
        silence_frames_threshold (int): 無音フレームのしきい値。
        text_output (st.empty): Streamlitのテキスト出力オブジェクト。
    戻り値：
        tuple[AudioSegment, int]: 更新されたサウンドチャンクと無音フレームの数。
   
    """
    if silence_frames >= silence_frames_threshold: 
        #無音フレーム数が100以上の時、音声の途切れ（間隔）として扱う
        if len(sound_chunk) > 0:
            #無音フレーム数が連続したら、音声の途切れとして、そこまでの音声データをテキストに変換している
            text = transcribe(sound_chunk)
            #text_output.write(text)
            #print("handle_silenceルーチンのtext=",text)
            #print("オーディオストリーム内の無音時の応答=",text)
            
            sound_chunk = pydub.AudioSegment.empty()
            silence_frames = 0
    return sound_chunk, silence_frames

def handle_queue_empty(sound_chunk, text_output):
    """
    オーディオフレームキューが空の場合の処理を行います。
    引数:
        sound_chunk (AudioSegment): 現在のサウンドチャンク。
        text_output (st.empty): Streamlitのテキスト出力オブジェクト。
    戻り値:
        AudioSegment: 更新されたサウンドチャンク。
    """
    if len(sound_chunk) > 0:
        text = transcribe(sound_chunk)
        #text_output.write(text)
        #st.session_state.text_output = text
        sound_chunk = pydub.AudioSegment.empty()
    return sound_chunk

# カスタムキューの作成 
# 固定サイズのキューを実現するためのもの
class CustomQueue: 
    def __init__(self, maxsize): 
        self.queue = deque(maxlen=maxsize)
        #maxsize という引数を受け取り、そのサイズのdeque（双方向キュー）を作成します。
        # dequeはPythonのコレクションの一種で、リストのようなもので、
        # maxlenを指定することでキューの最大サイズを設定できます。自動的に古いアイテムを削除します。 
    def put(self, item): 
        #print("len(self.queue)=",len(self.queue))
        # maxlenがNoneでないことを確認してから比較
        if self.queue.maxlen is not None and len(self.queue) >= self.queue.maxlen:
            print("Queue overflow. Oldest item will be discarded.") 
        self.queue.append(item)
        #キューに新しいアイテムを追加 
    def get(self): 
        #キューから最も古いアイテムを取り出して返します。
        #キューが空でない場合はpopleftメソッドを使って最も古いアイテムを取り出し、
        # キューが空の場合はqueue.Empty例外を発生させます。
        if len(self.queue) > 0: 
            return self.queue.popleft() 
        else: 
            raise queue.Empty 
    def qsize(self): 
        return len(self.queue)         


def app_sst_with_video():
    text_input = ""
    frames_deque_lock = threading.Lock()
    frames_deque: deque = deque([])

    async def queued_audio_frames_callback(
        frames: List[av.AudioFrame],
    ) -> av.AudioFrame:
        with frames_deque_lock:
            frames_deque.extend(frames)

        # Return empty frames to be silent.
        new_frames = []
        for frame in frames:
            input_array = frame.to_ndarray()
            new_frame = av.AudioFrame.from_ndarray(
                np.zeros(input_array.shape, dtype=input_array.dtype),
                layout=frame.layout.name,
            )
            new_frame.sample_rate = frame.sample_rate
            new_frames.append(new_frame)

        return new_frames
    
    # サイドバーにWebRTCストリームを表示
    with st.sidebar:
        st.header("Webcam Stream")
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text-w-video",
            desired_playing_state=True, 
            mode=WebRtcMode.SENDRECV, #.SENDONLY,  #
            #audio_receiver_size=2048,  #1024　#512 #デフォルトは4
            #小さいとQueue overflow. Consider to set receiver size bigger. Current size is 1024.
            queued_audio_frames_callback=queued_audio_frames_callback,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": True},
            video_processor_factory=VideoTransformer,  
        )
    
    #status_indicator.write("Loading...")
    cap_title = st.sidebar.empty()    
    cap_image = st.sidebar.empty() # プレースホルダーを作成 
    status_indicator = st.sidebar.empty() # プレースホルダーを作成
    
    ###################################################################
    #音声入力（テキストに変換した入力）の対話ループ
    #print("Before_st.session_state.input_method=",st.session_state.input_method)
    if st.session_state.input_method == "音声": 

        if webrtc_ctx is None:
            st.error("WebRTC context is not initialized. Please check your setup.")
        if webrtc_ctx and webrtc_ctx.audio_receiver is None:
            st.error("Audio receiver is not available. Please check the WebRTC mode.")
        if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.audio_receiver:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=st.session_state.timeout)
            #st.write(f"Received {len(audio_frames)} audio frames.")
        else:
            st.info("Waiting for audio stream to initialize...")
        if not webrtc_ctx.state.playing:
            return    
        audio_receiver_size = st.sidebar.slider(
        "音声受信容量。デフォルト4096:", 
        min_value=512, max_value=4096, value=4096, step=512
        )
        st.session_state.audio_receiver_size =audio_receiver_size
        # 無音を検出するための閾値    
        energy_threshold = st.sidebar.slider(
        "無音最大エネルギー。デフォルト400:", 
        min_value=100, max_value=900, value=400, step=100
        )
        energy_indicator = st.sidebar.empty() 

        amp_threshold = st.sidebar.slider(
            "無音最大振幅。デフォルト800:", 
            min_value=200, max_value=1600, value=800, step=100
            )
        amp_indicator = st.sidebar.empty() 

        silence_frames_threshold = st.sidebar.slider(
            "音声途切れ検出幅。デフォルト100:", 
            min_value=0, max_value=200, value=100, step=10
            )
        #60がBest,デフォルト100
        st.session_state.timeout = st.sidebar.slider(
            "音声検出タイムアウト。デフォルト3秒:", 
            min_value=1, max_value=3, value=1, step=1
            )
        st.write("🤖何か話して!")
        status_indicator = st.empty() # プレースホルダーを作成
        status_indicator.write("音声認識動作中...")
        text_output = st.empty() # プレースホルダーを作成
        #st.sidebar.header("Capture Image")
        #cap_image = st.sidebar.empty() # プレースホルダーを作成
        
        #stで使う変数初期設定
        st.session_state.energy_threshold = energy_threshold
        st.session_state.amp_threshold = amp_threshold
        st.session_state.silence_frames_threshold = silence_frames_threshold
        sound_chunk = pydub.AudioSegment.empty()
        silence_frames = 0
        #CustomQueue クラスを使ってキューを作成し、
        # オーバーフローが発生した場合には古いアイテムを削除するようにしています。
        # これにより、キューがいっぱいになっても新しいデータを受け取ることができます。
        # CustomQueueのインスタンス作成 
        #audio_queue = CustomQueue(MAX_QUEUE_SIZE)
        audio_queue = CustomQueue(st.session_state.audio_receiver_size)
        while True:
            if webrtc_ctx.audio_receiver:

                #handle_queue_overflow(webrtc_ctx.audio_receiver,st.session_state.audio_receiver_size)
                
                try:
                    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=st.session_state.timeout)
                    #print("len(audio_frames)=",len(audio_frames))
                    
                    for frame in audio_frames:
                        audio_queue.put(frame)    
                except queue.Empty:
                    status_indicator.write("No frame arrived.")
                    sound_chunk = handle_queue_empty(sound_chunk, text_output)
                    continue
                #print("len(audio_frames)=",len(audio_frames))    
                #sound_chunk, silence_frames ,energy,amplitude= process_audio_frames(audio_frames, sound_chunk, silence_frames, st.session_state.energy_threshold,st.session_state.amp_threshold)
                #sound_chunk, silence_frames = handle_silence(sound_chunk, silence_frames, st.session_state.silence_frames_threshold)
                try: 
                    while audio_queue.qsize() > 0: 
                        frame = audio_queue.get() 
                        sound_chunk, silence_frames, energy, amplitude = process_audio_frames([frame], sound_chunk, silence_frames, st.session_state.energy_threshold, st.session_state.amp_threshold) 
                        sound_chunk, silence_frames = handle_silence(sound_chunk, silence_frames, st.session_state.silence_frames_threshold) 
                        #print("len(sound_chunk) =",len(sound_chunk),"audio_queue.qsize() =",audio_queue.qsize() )
                        #print("audio_queue.qsize() =",audio_queue.qsize())    
                        # キューからデータを取得して処理 
                        try:
                            energy = 0.0 if np.isnan(energy) else energy
                            energy = round(energy)
                        except Exception as e:
                            #print(f"Error exporting round(energy): {e}")
                            energy = 0

                        energy_indicator.write(f"音声エネルギー={energy}")
                        amp_indicator.write(f"音声振幅={amplitude}")
                        if len(st.session_state.text_output) > 4 :
                            print("st.session_state.text_output=",st.session_state.text_output)    
                            text_input =  st.session_state.text_output 
                            st.session_state.text_output = ""
                            st.write(f"Received {len(audio_frames)} audio frames.")
                            #これ以降は、音声入力、テキスト入力共通の処理へ
                            qa(text_input,webrtc_ctx,cap_title,cap_image)
                            st.write(f"🤖何か話して!")  
                            text_input = ""
                except queue.Empty: 
                    status_indicator.write("Queue is empty.")
               
            else:
                status_indicator.write("音声認識停止")
                break

    ################################################################### 
    # テキスト入力の場合
    # テキスト入力フォーム
    if st.session_state.input_method == "テキスト":
        button_input = ""
        # 4つの列を作成
        col1, col2, col3, col4 = st.columns(4)
        # 各列にボタンを配置
        with col1:
            if st.button("画像の内容を説明して"):
                button_input = "画像の内容を説明して"
        with col2:
            if st.button("前の画像と何が変わりましたか？"):
                button_input = "前の画像と何が変わりましたか？"
        with col3:
            if st.button("この画像の文を翻訳して"):
                button_input = "この画像の文を翻訳して"
        with col4:
            if st.button("人生の意義は？"):
                button_input = "人生の意義？"
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            if st.button("日本の悪いところは？"):
                button_input = "日本の悪いところは？"
        with col6:
            if st.button("善悪は何で決まりますか？"):
                button_input = "善悪は何で決まりますか？"
        with col7:
            if st.button("小松市のおいしい料理店は？"):
                button_input = "小松市のおいしい料理店は？"
        with col8:
            if st.button("今日の料理はなにがいいかな"):
                button_input = "今日の料理はなにがいいかな"
        if button_input !="":
            st.session_state.user_input=button_input

        text_input =st.chat_input("🤗テキストで問い合わせる場合、ここに入力してね！") #,key=st.session_state.text_input)
        #text_input = st.text_input("テキストで問い合わせる場合、以下のフィールドに入力してください:", key=st.session_state.text_input) 
        if button_input:
            text_input = button_input
        if text_input:
            qa(text_input,webrtc_ctx,cap_title,cap_image)

################################################################### 
def qa(text_input,webrtc_ctx,cap_title,cap_image):
     # 末尾の空白の数を確認
    trailing_spaces = len(text_input) - len(text_input.rstrip())
    print(f"入力テキスト末尾の空白の数: {trailing_spaces}")
    # 末尾の空白を削除
    cleaned_text = text_input.rstrip()
    #print(f"入力テキスト末尾の空白を除去した文字列: '{cleaned_text}'")
    with st.chat_message('user'):   
        st.write(cleaned_text) 
    # 画像と問い合わせ入力があったときの処理
    cap = None 
    if st.session_state.input_img == "有":
        # 画像と問い合わせ入力があったときの処理
        #現在の画像をキャプチャする
        #キャプチャー画像入力
        if webrtc_ctx.video_transformer:  
            cap = webrtc_ctx.video_transformer.frame
        if cap is not None :
            #st.sidebar.header("Capture Image") 
            cap_title.header("Capture Image")     
            cap_image.image(cap, channels="BGR")
            # if st.button("Query LLM : 画像の内容を説明して"):
    # if st.button("Query LLM : 画像の内容を説明して"):
    with st.spinner("Querying LLM..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        st.session_state.result= ""
        result = loop.run_until_complete(query_llm(cleaned_text,cap))
        st.session_state.result = result
    result = ""
    text_input="" 
###################################################################      
if __name__ == "__main__":
    
    main()
