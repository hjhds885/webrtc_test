import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere.chat_models import ChatCohere
import base64
import speech_recognition as sr
#import pyttsx3
import asyncio
import nest_asyncio
import threading
import keyboard
from gtts import gTTS
import pygame
import os
#from torch import res

r = sr.Recognizer()
nest_asyncio.apply()
#######################################################################
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
            st.write("音声出力を開始します。")
            speak_thread = speak_async(response)
            # 必要に応じて音声合成の完了を待つ
            speak_thread.join() 
                   
            print("音声再生が完了しました。次の処理を実行します。")
            st.write("音声再生が完了しました。次の処理を実行します。")
            
        #if engine._inLoop:
            #print("音声出力がLOOPになっています。")
            #engine.endLoop()
            #print("音声再生LOOPを解除しました。次の処理を実行できます")

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
#######################################################################

class VideoProcessor:
    def __init__(self) -> None:
        #self.threshold1 = 100
        #self.threshold2 = 200
        self.frame = None
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #img = cv2.cvtColor(cv2.Canny(img, self.threshold1, self.threshold2), cv2.COLOR_GRAY2BGR)
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        #return av.VideoFrame.from_ndarray(img, format="bgr24")
        #return img
        return frame
    
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):   
        self.frame = frame.to_ndarray(format="bgr24")
        return frame
def init_page():
    st.set_page_config(
        page_title="Mr.Yas Chat",
        page_icon="🤗"
    )
    st.header("Mr.Yas Chat 🤗")
    st.write("Safari,Chrome,Edge,Firefoxなどブラウザのカメラ、マイク、スピーカーの使用許可設定が必要です。")
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
 #######################################################################
# 音声入力（認識）関数
def speech_to_text():
    with sr.Microphone() as source:
        audio = r.listen(source)
        try:
            return r.recognize_google(audio, language="ja-JP")
        except:
            return ""
#######################################################################
#音声出力関数
#engine = pyttsx3.init()
def speak_async1(text):
    def run():
        engine.say(text)
        engine.startLoop(False)
        engine.iterate()
        engine.endLoop()
        if engine._inLoop:
            print("音声出力がLOOPになっています。")
            engine.endLoop()
            print("音声再生LOOPを解除しました。次の処理を実行できます")

    
    thread = threading.Thread(target=run)
    thread.start()
    return thread
#######################################################################
#音声出力関数

def speak_async2(text):
    def run():
        st.write("音声ファイルを作成します。")
        # 初期設定
        pygame.mixer.init()
        # Pygameを終了してファイルを解放
        #pygame.mixer.quit()
        # テキストを音声に変換
        tts = gTTS(text=text, lang='ja')
        output_file="output.mp3"
        tts.save(output_file)
        st.write("音声ファイルが保存されました。")
        # 音声ファイルを提供
        pygame.mixer.init()
        # Pygameを使って音声を再生
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()
        # 再生が終了するまで待機
        while pygame.mixer.music.get_busy():
            continue
        # Pygameを終了してファイルを解放
        pygame.mixer.quit()
        st.write("st.audioでの音声出力が完了しました。")
        # 音声ファイルを削除
        os.remove(output_file)
        st.write("音声再生が完了し、ファイルは削除されました。")    
    thread = threading.Thread(target=run)
    thread.start()
    return thread
def speak_async(text):
    def run():
        st.write("音声ファイルを作成します。")
        # テキストを音声に変換
        tts = gTTS(text=text, lang='ja')
        output_file="output.mp3"
        tts.save(output_file)
        st.write("音声ファイルが保存されました。")
        # 音声ファイルを提供
        audio_file = open(output_file, "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3", start_time=0)
    
        st.write("st.audioでの音声出力が完了しました。")
        # 音声ファイルを削除
        os.remove(output_file)
        st.write("音声再生が完了し、ファイルは削除されました。")    
    thread = threading.Thread(target=run)
    thread.start()
    return thread
#######################################################################

 #async 
def main(): 
    ###################################################################    
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
    
    
    # サイドバーにWebRTCストリームを表示
    with st.sidebar:
        st.header("Webcam Stream")
        webrtc_ctx=webrtc_streamer(
                key="speech-to-text",
                desired_playing_state=True, 
                #mode=WebRtcMode.SENDRECV,
                #audio_receiver_size=1024,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
                #audio_processor_factory=AudioTransformer,
                video_processor_factory=VideoTransformer,
            )   
    #if webrtc_ctx.state.playing:
        #st.write("WebRTC is playing")
    #else:
        #st.write("WebRTC is not playing")


    user_input = ""
    base64_image = ""
    frame = ""    
    
    #キャプチャー画像入力
    if webrtc_ctx.video_transformer:  
        frame = webrtc_ctx.video_transformer.frame
        ##st.sidebar.image(frame)
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
            if st.button("CIDPとは？"):
                button_input = "CIDPとは？"

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
        
        
        text_input =st.chat_input("テキストで問い合わせる場合、ここに入力してね！") #,key=st.session_state.text_input)
        #text_input = st.text_input("テキストで問い合わせる場合、以下のフィールドに入力してください:", key=st.session_state.text_input) 
        if text_input:
            st.session_state.user_input=text_input
            text_input=""

        if st.session_state.user_input:
            print("user_input=",st.session_state.user_input)
            with st.chat_message('user'):   
                st.write(st.session_state.user_input) 
        # 対話ループ 
        # 画像と問い合わせ入力があったときの処理
            if frame is not None and st.session_state.user_input !="":
                st.sidebar.header("Capture Image")
                st.sidebar.image(frame, channels="BGR")
                # if st.button("Query LLM : 画像の内容を説明して"):
                with st.spinner("Querying LLM..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    st.session_state.result= ""
                    result = loop.run_until_complete(query_llm(st.session_state.user_input,frame))
                    st.session_state.result = result
                    result = ""
                    #result = await query_llm(text,frame)
                    st.session_state.user_input=""
    ###############################################################################
    #音声入力（テキストに変換した入力）の対話ループ
    #print("Before_st.session_state.input_method=",st.session_state.input_method)
    if st.session_state.input_method == "音声": 
        already_displayed = False
        st.sidebar.header("Capture Image") 
        image_placeholder = st.sidebar.empty()
         
        while True:
            if not already_displayed:
                print("話しかけてください...")
                st.write("🤗話しかけてください...")
                already_displayed = True
            st.session_state.user_input = ""
            st.session_state.user_input = speech_to_text()
            if keyboard.is_pressed('1') :st.session_state.user_input ="こんばんは"
            if keyboard.is_pressed('2') :st.session_state.user_input ="画像の内容を説明して"
            if keyboard.is_pressed('3') :st.session_state.user_input ="石川県小松市の観光地は？"
            if keyboard.is_pressed('4') :st.session_state.user_input ="有名な道の駅は？"
            if keyboard.is_pressed('5') :st.session_state.user_input ="CIDPとは？"
            if keyboard.is_pressed('6') :st.session_state.user_input ="きょうの料理はなにがいいかな"
            if keyboard.is_pressed('7') :st.session_state.user_input ="宇宙人はいますか？"
            if keyboard.is_pressed('8') :st.session_state.user_input ="私の名前は誠です。"
            if keyboard.is_pressed('9') :st.session_state.user_input ="私の名前は？"
            if keyboard.is_pressed('0') :st.session_state.user_input ="善悪は何で決まりますか？"
            if keyboard.is_pressed('esc') :
                print("音声での問い合わせを終了しました。")
                with st.chat_message('assistant'):   
                    st.write("音声での問い合わせを終了しました。") 
                #break   
            # 対話ループ 
            # 画像と問い合わせ入力があったときの処理
            if webrtc_ctx.video_transformer: #VideoProcessor
                frame = webrtc_ctx.video_transformer.frame  #VideoProcessor.frame 
            if frame is not None and st.session_state.user_input !="":
                #サイドバーに画像を表示
                image_placeholder.image(frame, channels="BGR")
                #ユーザーの音声入力を表示
                with st.chat_message('user'):   
                    st.write(st.session_state.user_input) 
                #LMMの回答を表示 
                with st.spinner("Querying LLM..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    st.session_state.result= ""
                    result = loop.run_until_complete(query_llm(st.session_state.user_input,frame))
                    #result = await query_llm(st.session_state.user_input,frame)
                st.session_state.result = result
                result = ""
                st.session_state.user_input=""
                already_displayed = False
                    
    ###############################################################################  
    ###############################################################################
    #await text_input =st.chat_input("テキストで問い合わせる場合、ここに入力してね！") #,key=st.session_state.text_input)
    #text_input = st.text_input("テキストで問い合わせる場合、以下のフィールドに入力してください:", key=st.session_state.text_input) 
    
if __name__ == "__main__":
    main()
    #asyncio.run(main())
    #loop = asyncio.get_event_loop()
    #loop.run_until_complete(main())


