import streamlit as st
import requests
import argparse
import os


# how to run: streamlit run streamlit_app.py --ip localhost --port 8000
def main(args):
    
    SERVER_URL = f"{args.ip}:{args.port}"
    print(f'api server: {SERVER_URL}')
    
    st.title("파일 업로드")

    # 사용 가능한 모델 목록을 요청하는 함수
    def get_available_models():
        response = requests.get(f"http://{SERVER_URL}/models")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"모델 목록을 가져오는 데 실패했습니다: {response.status_code}")
            return []
    models = get_available_models()
    # 모델 선택 위젯
    if models:
        selected_model = st.selectbox("사용할 모델을 선택하세요", models)
    else:
        selected_model = None
        st.warning("사용 가능한 모델 목록을 불러올 수 없습니다.")
        
        
    # 파일 업로드 위젯
    uploaded_file = st.file_uploader("예측할 CSV 파일을 업로드하세요", type="csv")
    # 업로드된 파일이 있으면 처리
    if uploaded_file is not None:
        # FastAPI 서버로 파일 전송
        server_url = f"http://{SERVER_URL}/uploadfile/predict"
        files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
        data = {"model": selected_model} # 선택된 모델을 함께 전송
        
        response = requests.post(server_url, files=files, data=data)

        if response.status_code == 200:
            st.success("파일이 성공적으로 FastAPI 서버로 전송되었습니다.")
            # 점수 범위가 0~1 이므로 ProgressBar 사용
            score = response.json().get("Score", 0)
            score_percentage = score * 100
            energy_bar = st.progress(0)  # 초기값 0으로 설정
            energy_bar.progress(score)  # 받은 점수로 ProgressBar 업데이트
            st.write(f"고객 이탈 확률: {score_percentage:.2f}%")
        else:
            st.error(f"파일 전송 실패: {response.status_code}")

    # 파일 업로드 위젯
    uploaded_file = st.file_uploader("학습할 CSV 파일을 업로드하세요", type="csv")
    # 업로드된 파일이 있으면 처리
    if uploaded_file is not None:
        # FastAPI 서버로 파일 전송
        server_url = f"http://{SERVER_URL}/uploadfile/train"
        files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
        
        response = requests.post(server_url, files=files, )

        if response.status_code == 200:
            st.success("파일이 성공적으로 FastAPI 서버로 전송되었습니다.")
            st.write(response.json())  # 서버 응답 내용 표시
            # 파일 전송 후 로컬에서 업로드된 파일 삭제
            uploaded_file_path = uploaded_file.name
            if os.path.exists(uploaded_file_path):
                os.remove(uploaded_file_path)
                st.info(f"{uploaded_file_path}가 삭제되었습니다.")
        else:
            st.error(f"파일 전송 실패: {response.status_code}")
    
    # PDF 다운로드 버튼
    if st.button("PDF 다운로드"):
        pdf_url = f"http://{SERVER_URL}/download_pdf"  # FastAPI 서버의 PDF 다운로드 URL
        response = requests.get(pdf_url)
        
        if response.status_code == 200:
            # PDF 다운로드를 위한 링크 제공
            st.download_button(
                label="PDF 파일 다운로드",
                data=response.content,
                file_name="document.pdf",
                mime="application/pdf"
            )
            st.success("PDF 파일을 다운로드할 수 있습니다.")
        else:
            st.error("PDF 다운로드 실패")
        
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--ip', type=str, default='localhost', help="api IP address")
    parser.add_argument('--port', type=int, default=8000, help="api Port number")
    parser.add_argument('--verbose', action='store_true', help="verbose mode")
    args = parser.parse_args()

    main(args)