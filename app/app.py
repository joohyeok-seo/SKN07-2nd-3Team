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
        datap = {"model": selected_model} # 선택된 모델을 함께 전송
        
        response = requests.post(server_url, files=files, data=datap)

        if response.status_code == 200:
            st.success("파일이 성공적으로 전송되었습니다.")
            # 점수 범위가 0~1 이므로 ProgressBar 사용
            score = response.json().get("prob", 0)
            score_percentage = score * 100
            energy_bar = st.progress(0)  # 초기값 0으로 설정
            energy_bar.progress(score)  # 받은 점수로 ProgressBar 업데이트
            num = response.json().get("total", 0)
            
            if num[0] > 1:
                st.write(f"고객 이탈률 에측: {score_percentage:.2f}% / 총 {num[0]}명 중 {(num[0] * score):.0f}명 이탈 예정으로 보여요.")
            else:
                prob = response.json().get("ratio", 0) * 100
                st.write(f"고객 이탈률 예측: {prob:.2f}%")
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
            st.success("파일이 성공적으로 전송되었습니다.")
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
    

    
    # Streamlit UI 구성
    st.title("Customer Churn Prediction")
    st.write("Enter the customer details to predict churn.")

    # 각 컬럼에 대한 입력 받기
    state = st.text_input("State (e.g., CA, TX, NY)", "")
    account_length = st.number_input("Account length", min_value=1, max_value=500, value=100)
    area_code = st.number_input("Area code", min_value=0, max_value=999, value=415)
    international_plan = st.selectbox("International plan", ["yes", "no"])
    voice_mail_plan = st.selectbox("Voice mail plan", ["yes", "no"])
    number_vmail_messages = st.number_input("Number of voice mail messages", min_value=0, max_value=100, value=0)
    total_day_minutes = st.number_input("Total day minutes", min_value=0.0, max_value=500.0, value=120.0)
    total_day_calls = st.number_input("Total day calls", min_value=0, max_value=100, value=60)
    total_day_charge = st.number_input("Total day charge", min_value=0.0, max_value=100.0, value=20.0)
    total_eve_minutes = st.number_input("Total evening minutes", min_value=0.0, max_value=500.0, value=100.0)
    total_eve_calls = st.number_input("Total evening calls", min_value=0, max_value=100, value=60)
    total_eve_charge = st.number_input("Total evening charge", min_value=0.0, max_value=100.0, value=15.0)
    total_night_minutes = st.number_input("Total night minutes", min_value=0.0, max_value=500.0, value=100.0)
    total_night_calls = st.number_input("Total night calls", min_value=0, max_value=100, value=60)
    total_night_charge = st.number_input("Total night charge", min_value=0.0, max_value=100.0, value=10.0)
    total_intl_minutes = st.number_input("Total international minutes", min_value=0.0, max_value=500.0, value=10.0)
    total_intl_calls = st.number_input("Total international calls", min_value=0, max_value=100, value=5)
    total_intl_charge = st.number_input("Total international charge", min_value=0.0, max_value=20.0, value=5.0)
    customer_service_calls = st.number_input("Customer service calls", min_value=0, max_value=10, value=2)

    # 입력받은 데이터
    input_data = {
        "State": state,
        "Account length": account_length,
        "Area code": area_code,
        "International plan": 1 if international_plan == "yes" else 0,
        "Voice mail plan": 1 if voice_mail_plan == "yes" else 0,
        "Number vmail messages": number_vmail_messages,
        "Total day minutes": total_day_minutes,
        "Total day calls": total_day_calls,
        "Total day charge": total_day_charge,
        "Total eve minutes": total_eve_minutes,
        "Total eve calls": total_eve_calls,
        "Total eve charge": total_eve_charge,
        "Total night minutes": total_night_minutes,
        "Total night calls": total_night_calls,
        "Total night charge": total_night_charge,
        "Total intl minutes": total_intl_minutes,
        "Total intl calls": total_intl_calls,
        "Total intl charge": total_intl_charge,
        "Customer service calls": customer_service_calls
    }


    if st.button("Predict Churn"):
        # 예측 수행
        server_url = f"http://{SERVER_URL}/uploadfile/single"
        data = {"model": selected_model,
                "input_data": input_data} # 선택된 모델을 함께 전송
        response = requests.post(server_url, data=data, )
        
        prediction = response.json().get("prob", 0)
        c = response.json().get("class", 0)

        if c == 1:
            st.write(f"The customer will churn. ({prediction:.2f}%)")
        else:
            st.write(f"The customer will not churn.")
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--ip', type=str, default='localhost', help="api IP address")
    parser.add_argument('--port', type=int, default=8000, help="api Port number")
    parser.add_argument('--verbose', action='store_true', help="verbose mode")
    args = parser.parse_args()

    main(args)