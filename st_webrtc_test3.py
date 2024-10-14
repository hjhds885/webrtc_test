import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2

st.title("My first Streamlit app")
st.write("Hello, world")


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




#async 
def main():  
    ctx = webrtc_streamer(
            key="example",
            video_processor_factory=VideoProcessor,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            #video_frame_callback=Callback, NG画面画止まる

        )
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




    if webrtc_ctx.state.playing:
        st.write("WebRTC is playing")
    else:
        st.write("WebRTC is not playing")


    user_input = ""
    base64_image = ""
    frame = ""    
    #stで使う変数初期設定
    #st.session_state.llm = select_model()
    st.session_state.input_method = ""
    st.session_state.user_input = ""
    st.session_state.result = ""
    st.session_state.frame = "" 

    #キャプチャー画像入力
    if webrtc_ctx.video_transformer:  
        frame = webrtc_ctx.video_transformer.frame
        ##st.sidebar.image(frame)

    

    #await text_input =st.chat_input("テキストで問い合わせる場合、ここに入力してね！") #,key=st.session_state.text_input)
    #text_input = st.text_input("テキストで問い合わせる場合、以下のフィールドに入力してください:", key=st.session_state.text_input) 
    """
    if text_input:
        st.session_state.user_input=text_input
        text_input=""
        #キャプチャー画像入力
        if ctx.video_transformer:  
            frame = ctx.video_transformer.frame  #VideoProcessor.frame 
            st.sidebar.header("Capture Image")
            st.sidebar.image(frame, channels="BGR")

        print("user_input=",st.session_state.user_input)
        with st.chat_message('user'):   
            st.write(st.session_state.user_input) 

        
        
    """
if __name__ == "__main__":
    main()
    #asyncio.run(main())
    #loop = asyncio.get_event_loop()
    #loop.run_until_complete(main())


