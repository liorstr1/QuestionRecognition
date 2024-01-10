import json
import os
import time
from selenium.webdriver import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from entities import Status, WHATSAPP_WEB, UserData, ClientData, CLIENT_PATH
from selenium.webdriver.support import expected_conditions as ec
from selenium import webdriver
from services.prompts_and_texts import INIT_JSON_STRUCT
from selenium.webdriver.chrome.options import Options


def init_data(user_name):
    chrome_options = Options()
    # Now, pass chrome_options when initializing the driver
    driver = webdriver.Chrome(options=chrome_options)
    # WebDriver setup
    # s = Service(SERVICE_SETUP)
    status = Status()
    user_data = UserData(
        user_name=user_name,
        user_id=1
    )
    client_data = ClientData(
        client_name="david",
        client_id=1
    )

    # Access WhatsApp Web
    driver.get(WHATSAPP_WEB)
    input("Scan the QR code and then press Enter")

    # Search for specific contact
    search_box = WebDriverWait(driver, 20).until(
        ec.presence_of_element_located((By.XPATH, '//div[@contenteditable="true"][@data-tab="3"]'))
    )
    search_box.send_keys(user_name, Keys.ENTER)
    time.sleep(2)
    return driver, status, user_data, client_data


def check_latest_message(latest_message, user_messages):
    if 'message-in' in latest_message.get_attribute('class') and not user_messages:
        return True
    if 'message-in' in latest_message.get_attribute('class') and latest_message != user_messages[-1]:
        return True
    return False


def analyze_new_message(latest_message):
    timestamp = latest_message.find_element(By.CSS_SELECTOR, 'div.copyable-text').get_attribute('data-pre-plain-text')
    message_content = latest_message.find_element(By.CSS_SELECTOR, 'span.selectable-text').text
    print(f"New incoming message at {timestamp}: {message_content}")
    return message_content


def get_updated_json(client_data: ClientData, user_data: UserData):
    main_client_path = CLIENT_PATH
    if not os.path.exists(main_client_path):
        os.makedirs(main_client_path)
    client_path = os.path.join(main_client_path, str(client_data.client_id))
    if not os.path.exists(client_path):
        os.makedirs(client_path)
    user_path = os.path.join(client_path, str(user_data.user_id))
    if not os.path.exists(user_path):
        os.makedirs(user_path)
    json_path = os.path.join(user_path, "my_question_path.json")

    if not os.path.exists(json_path):
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(INIT_JSON_STRUCT)

    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    return json_path, json_data


def check_json_struct(json_struct):
    if None in list(json_struct.values()):
        return True
    return False


def get_next_question(json_struct):
    return [k for k, v in json_struct.items() if v is None][0]





