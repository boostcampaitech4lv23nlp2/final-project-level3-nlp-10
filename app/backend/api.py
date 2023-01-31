import requests


def CSR(data):
    Lang = "Kor"
    URL = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=" + Lang

    ID = ""  # 인증 정보의 Client ID
    Secret = ""  # 인증 정보의 Client Secret

    headers = {
        "Content-Type": "application/octet-stream",  # Fix
        "X-NCP-APIGW-API-KEY-ID": ID,
        "X-NCP-APIGW-API-KEY": Secret,
    }
    response = requests.post(URL, data=data, headers=headers)
    print(type(response))
    rescode = response.status_code

    if rescode == 200:
        return response.json()
    else:
        print("Error : " + response.text)
