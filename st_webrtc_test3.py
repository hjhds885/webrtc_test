import threading
import time
from collections import deque
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoTransformerBase
import speech_recognition as sr
from typing import List
import io
import wave
import asyncio
import time

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

def init_page():
    st.set_page_config(
        page_title="Mr.Yas Chatbot",
        page_icon="🤖"
    )
    st.header("Mr.Yas Chatbot 🤖")
    st.write("""カメラとマイクのアクセスを許可するようブラウザを設定してください。
         \nChrome・・・support.google.com参照。「カメラやマイクのアクセスを許可する」を入力して検索,
         \nSafari・・・support.apple.com参照。設定⇒Safari⇒カメラ及びマイク⇒確認又は許可,
         \nFirefox・・・support.mozilla.org参照。         
         """) 
    
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
    temperature = st.sidebar.slider(
        "Temperature(回答バラツキ度合):", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
    models = ( "GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro")
    model = st.sidebar.radio("Choose a model（大規模言語モデルを選択）:", models)

       
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
            #model=st.session_state.model_name,
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
    print("user_input=",user_input)
    
    try:
            
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
            # LLMの返答を表示する  Streaming
            with st.chat_message('ai'):   
                #st.write(response)  
                response = st.write_stream(stream) 
            #full = next(stream)
            #for chunk in stream:
                #full += chunk
            #response = full    
            print(response)
          
        
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
            # LLMの返答を表示する  Streaming
            with st.chat_message('ai'):   
                #st.write(response)  
                response =st.write_stream(stream) 
            print("response=",response)

        elif st.session_state.model_name ==  "keep_command-r-plus":
            print("st.session_state.model_name=",st.session_state.model_name)
            prompt = ChatPromptTemplate.from_messages([
                    *st.session_state.message_history,
                    ("user", "{user_input}")  # ここにあとでユーザーの入力が入る
                ])
            output_parser = StrOutputParser()
            chain = prompt | st.session_state.llm | output_parser
            stream = chain.stream(user_input)
            
            #response = chain.invoke(user_input)
            # LLMの返答を表示する  Streaming
            with st.chat_message('ai'):   
                #st.write(response)  
                response =st.write_stream(stream) 
            print("response=",response)
        else:
            print("st.session_state.model_name=",st.session_state.model_name)
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
            # LLMの返答を表示する  Streaming
            with st.chat_message('ai'):   
                #st.write(response)  
                response =st.write_stream(stream) 
            #print("response=",response)            
        

        print(f"{st.session_state.model_name}=",response)
        

        # 音声出力処理                
        if st.session_state.output_method == "音声":
            #st.write("音声出力を開始します。")
            speak(response)   #st.audio ok
            #speak1(response) pygame NG
            #speak_thread = speak_async(response)
            # 必要に応じて音声合成の完了を待つ
            #speak_thread.join() 
            #print("音声再生が完了しました。次の処理を実行します。")
            #st.write("音声再生が完了しました。次の処理を実行できます。")
            
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
    col, col2 = st.sidebar.columns(2)
     # 各列にボタンを配置
    with col:
        # 入力方法の選択
        input_method = st.sidebar.radio("入力方法", ("テキスト", "音声"))
        st.session_state.input_method = input_method
     # 各列にボタンを配置
    with col2:
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
async def process_audio(audio_data_bytes, sample_rate, text_output):
    audio_data_io = io.BytesIO()
    with wave.open(audio_data_io, 'wb') as wf:
        wf.setnchannels(2)
        #wf.setnchannels(1)  # モノラル音声として記録　NG
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data_bytes)
    audio_data_io.seek(0)

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_data_io) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language="ja-JP")
            print("認識されたテキスト:", text)
            text_output.write(f"変換されたテキスト：{text}")  # プレースホルダーにテキストを表示
            st.session_state.user_input = text
        except sr.UnknownValueError:
            text = "音声を認識できませんでした。"
            #text_output.write(f"変換されたテキスト：{text}")  # プレースホルダーにエラーメッセージを表示
        except sr.RequestError as e:
            text = f"サービスにアクセスできませんでした; {e}"
            #text_output.write(f"変換されたテキスト：{text}") 
    return text
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
            video_processor_factory=VideoTransformer,  #機能している？
        )

    
    
    if not webrtc_ctx.state.playing:
        return
    #status_indicator.write("Loading...")

    ###################################################################
    #音声入力（テキストに変換した入力）の対話ループ
    #print("Before_st.session_state.input_method=",st.session_state.input_method)
    if st.session_state.input_method == "音声": 
        
        status_indicator = st.empty() # プレースホルダーを作成
        status_indicator.write("Loading...")
        text_output = st.empty() # プレースホルダーを作成
        
        st.sidebar.header("Capture Image")
        cap_image = st.sidebar.empty() # プレースホルダーを作成
        
        while True:
            if webrtc_ctx.state.playing:
                audio_frames = []
                with frames_deque_lock:
                    while len(frames_deque) > 0:
                        frame = frames_deque.popleft()
                        audio_frames.append(frame)

                if len(audio_frames) == 0:
                    time.sleep(0.1)
                    status_indicator.write("No frame arrived.")
                    continue

                status_indicator.write("🤗何か話して!")
                audio_buffer = []  # バッファを初期化
                for audio_frame in audio_frames:
                    
                    # フレームを numpy 配列として取得（s16 フォーマットを int16 として解釈）
                    audio = audio_frame.to_ndarray().astype(np.int16)
                    audio_buffer.append(audio)  # バッファにフレームデータを追加

                    # 正規化して -1.0 から 1.0 の範囲に収める
                    #max_val = np.max(np.abs(audio_buffer))
                    #if max_val > 0:
                        #audio_buffer = audio_buffer / max_val

                if len(audio_buffer) >0:  # 100: # 
                    # 複数フレームをまとめる
                    audio_data = np.concatenate(audio_buffer)
                    audio_data_bytes= audio_data.tobytes()
                    st.session_state.user_input=""
                    # 非同期で音声データを処理
                    text_input=asyncio.run(process_audio(audio_data_bytes, frame.sample_rate, text_output))
                    # バッファをクリア
                    audio_buffer.clear() 
                    #print("ここを通過E2") #ここまでOK
                    #print("text_input=",text_input)
                    if st.session_state.user_input !="":
                        print("st.session_state.user_input=",st.session_state.user_input)  
                        #llm_in()
                        print("user_input=",st.session_state.user_input)
                        with text_output.chat_message('user'):   
                            st.write(st.session_state.user_input) 
                        # 画像と問い合わせ入力があったときの処理
                        #現在の画像をキャプチャする
                        cap = None    
                        #キャプチャー画像入力
                        if webrtc_ctx.video_transformer:  
                            cap = webrtc_ctx.video_transformer.frame
                        if cap is not None and st.session_state.user_input !="":
                            #st.sidebar.header("Capture Image")
                            cap_image.image(cap, channels="BGR")
                            # if st.button("Query LLM : 画像の内容を説明して"):
                            with st.spinner("Querying LLM..."):
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                st.session_state.result= ""
                                result = loop.run_until_complete(query_llm(st.session_state.user_input,cap))
                                st.session_state.result = result
                                result = ""
                                #result = await query_llm(text,frame)
                                st.session_state.user_input=""
                                        
            else:
                status_indicator.write("Stopped.")
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
            if st.button("日本語に翻訳してください。"):
                button_input = "日本語に翻訳してください。"
        with col6:
            if st.button("善悪は何で決まりますか？"):
                button_input = "善悪は何で決まりますか？"
        with col7:
            if st.button("日本の観光地を教えてください。"):
                button_input = "日本の観光地を教えてください。"
        with col8:
            if st.button("今日の料理はなにがいいかな"):
                button_input = "今日の料理はなにがいいかな"
        if button_input !="":
            st.session_state.user_input=button_input

        text_input =st.chat_input("🤗テキストで問い合わせる場合、ここに入力してね！") #,key=st.session_state.text_input)
        #text_input = st.text_input("テキストで問い合わせる場合、以下のフィールドに入力してください:", key=st.session_state.text_input) 
        if text_input:
            st.session_state.user_input=text_input
            text_input=""
            #llm_in()
            print("user_input=",st.session_state.user_input)
        with st.chat_message('user'):   
            st.write(st.session_state.user_input) 
        # 画像と問い合わせ入力があったときの処理
        #現在の画像をキャプチャする
        cap = None    
        #キャプチャー画像入力
        if webrtc_ctx.video_transformer:  
            cap = webrtc_ctx.video_transformer.frame
        if cap is not None and st.session_state.user_input !="":
            st.sidebar.header("Capture Image")
            st.sidebar.image(cap, channels="BGR")
            # if st.button("Query LLM : 画像の内容を説明して"):
            with st.spinner("Querying LLM..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                st.session_state.result= ""
                result = loop.run_until_complete(query_llm(st.session_state.user_input,cap))
                st.session_state.result = result
                result = ""
                #result = await query_llm(text,frame)
                st.session_state.user_input=""

################################################################### 
 
###################################################################      
if __name__ == "__main__":
    
    main()
