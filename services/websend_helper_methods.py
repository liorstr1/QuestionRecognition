import json
import os
import time
from datetime import datetime
from selenium.webdriver import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from entities import SERVICE_SETUP, Status, WHATSAPP_WEB, UserData, ClientData
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as ec

from services.prompts_and_texts import INIT_JSON_STRUCT


def init_data(user_name):
    print('Hey Elior, I\'m on = ', datetime.now().strftime("%H:%M:%S"))
    # WebDriver setup
    s = Service(SERVICE_SETUP)
    driver = webdriver.Chrome(service=s)
    status = Status()
    user_data = UserData(
        name=user_name,
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


def check_latest_message(latest_message, last_incoming_message):
    if 'message-in' in latest_message.get_attribute('class') and latest_message != last_incoming_message:
        return True
    return False


def analyze_new_message(latest_message):
    timestamp = latest_message.find_element(By.CSS_SELECTOR, 'div.copyable-text').get_attribute('data-pre-plain-text')
    message_content = latest_message.find_element(By.CSS_SELECTOR, 'span.selectable-text').text
    print(f"New incoming message at {timestamp}: {message_content}")
    return message_content


def get_updated_json(client_data: ClientData, user_data: UserData):
    json_path = './jsons/'
    if not os.path.exists(json_path):
        os.makedirs(json_path)
    client_path = os.path.join(json_path, client_data.client_id)
    if not os.path.exists(client_path):
        os.makedirs(client_path)
    user_path = os.path.join(client_path, user_data.user_id)
    if not os.path.exists(user_path):
        os.makedirs(user_path)
    json_path = os.path.join(user_path, "my_question_path.json")
    with open(json_path, 'r') as f:
        json.dump(INIT_JSON_STRUCT, f)
    return json_path, json.loads(INIT_JSON_STRUCT)


def check_json_struct(json_struct):
    if None in list(json_struct.values()):
        return True
    return False


def get_next_question(json_struct):
    return [k for k, v in json_struct.items() if v is None][0]





