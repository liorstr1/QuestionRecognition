from selenium.common.exceptions import NoSuchElementException
from chat_with_gpt.main_gpt_process import ChatWithGPT
from services.api import analyze_init_user_answer, update_json_struct
from services.prompts_and_texts import INITIAL_CONTACT, START_ASK_QUESTIONS
from services.websend_helper_methods import *

IS_CALL_API_FLAG = False  # Global flag


def run_main_process():
    driver, status, user_data, client_data = init_data(
        user_name='אליאור שמש'
    )
    send_outgoing_message(driver, INITIAL_CONTACT)
    status.update_init_asked()
    user_messages = []

    # Main loop to continuously check for new messages and respond
    while status.keep_running:
        incoming_message = continuously_check_for_new_messages(driver, user_messages)
        if incoming_message:
            call_api(driver, incoming_message, status, user_data, client_data, user_messages)
    driver.quit()


def process_existing_messages(driver):
    time.sleep(5)
    # Locate all messages in the chat
    messages = driver.find_elements(By.XPATH, "//div[contains(@class, '_1AOLJ _2UtSC _1jHIY')]")

    last_incoming = None
    for message in messages:
        # Determine if the message is incoming or outgoing
        if 'message-in' in message.get_attribute('class'):
            last_incoming = message  # Update last incoming message

        # Check for voice message
        voice_message_elements = message.find_elements(By.XPATH,
                                                       ".//button[contains(@aria-label, 'השמעה של הודעה קולית')]")
        if len(voice_message_elements) > 0:
            timestamp = "Not available"
            message_content = "Voice message"
        else:
            try:
                # Extract the timestamp
                timestamp = message.find_element(By.CSS_SELECTOR, 'div.copyable-text').get_attribute(
                    'data-pre-plain-text')
                # Extract the text content of the message
                message_content = message.find_element(By.CSS_SELECTOR, 'span.selectable-text').text
            except NoSuchElementException:
                # Handle the case where the message might be another non-text message
                timestamp = "Not available"
                message_content = "Non-text message (e.g., image, video, etc.)"

        print(
            f"Type: {'incoming' if last_incoming == message else 'outgoing'}, Timestamp: {timestamp}, "
            f"Message: {message_content}")

    return last_incoming


def continuously_check_for_new_messages(driver, user_messages):
    global IS_CALL_API_FLAG
    while True:
        if IS_CALL_API_FLAG:
            print("Waiting for API processing to complete...")
            time.sleep(5)  # Wait before checking again
            continue

        print("Checking for messages...")
        time.sleep(5)  # Wait for 5 seconds

        current_messages = driver.find_elements(By.XPATH, "//div[contains(@class, '_1AOLJ _2UtSC _1jHIY')]")
        if current_messages:
            latest_message = current_messages[-1]
            if check_latest_message(latest_message, user_messages):
                user_messages.append(latest_message)
                message_content = analyze_new_message(latest_message)
                time.sleep(2)
                return message_content


def send_outgoing_message(driver, message_to_send):
    try:
        xpath_expression = "//div[contains(@class, 'lexical-rich-text-input')]//div[@contenteditable='true' and @title='הקלדת הודעה']"
        message_box = WebDriverWait(driver, 20).until(
            ec.presence_of_element_located((By.XPATH, xpath_expression))
        )
        time.sleep(2)
        message_box.click()
        time.sleep(4)
        message_box.send_keys(message_to_send, Keys.ENTER)
    except Exception as e:
        print(e.args)


def send_response(driver, response):
    message_box = WebDriverWait(driver, 20).until(
        ec.presence_of_element_located((By.XPATH,
                                        "//div[contains(@class, 'lexical-rich-text-input')]//div[@contenteditable='true' and @title='הקלדת הודעה']"))
    )
    message_box.click()
    time.sleep(2)
    message_box.send_keys(response, Keys.ENTER)


def call_api(driver, message_content, status: Status, user_data, client_data, user_messages):
    global IS_CALL_API_FLAG
    IS_CALL_API_FLAG = True
    print("start")

    if status.is_init_answer():
        analyze_init_user_answer(message_content, status)
    if status.is_fill_json():
        send_outgoing_message(driver, START_ASK_QUESTIONS)
        json_path, json_struct = get_updated_json(client_data, user_data)
        cwg = ChatWithGPT()
        # start to fill json with chatGPT
        while check_json_struct(json_struct):
            next_question = get_next_question(json_struct)
            check_next, next_message, messages = cwg.fill_init_message_with_struct_json(next_question)
            while not check_next:
                IS_CALL_API_FLAG = False
                # send user gpt message
                send_outgoing_message(driver, next_message)
                status.update_gpt_sent()

                # wait the user to answer
                while status.keep_running:
                    incoming_message = continuously_check_for_new_messages(driver, user_messages)
                    if incoming_message:
                        check_next, next_message, messages = cwg.analyze_next_user_answer(
                            messages,
                            incoming_message
                        )
                        status.update_gpt_got()
            update_json_struct(json_path, json_struct, next_question, next_message)
        status.update_finish_gpt()


def test_gpt():
    user_data = UserData(
        user_name="אליאור שמש",
        user_id=1
    )
    client_data = ClientData(
        client_name="david",
        client_id=1
    )

    json_path, json_struct = get_updated_json(client_data, user_data)
    cwg = ChatWithGPT()
    # start to fill json with chatGPT
    while check_json_struct(json_struct):
        next_question = get_next_question(json_struct)
        check_next, next_message, messages = cwg.fill_init_message_with_struct_json(next_question)
        while not check_next:
            # send user gpt message
            send_outgoing_message(driver, INITIAL_CONTACT)
            status.update_gpt_sent()

            # wait the user to answer
            while status.keep_running:
                incoming_message = continuously_check_for_new_messages(driver, user_messages)
                if incoming_message:
                    check_next, next_message, messages = cwg.analyze_next_user_answer(
                        messages,
                        incoming_message
                    )
                    status.update_gpt_got()
        update_json_struct(json_path, json_struct, next_question, next_message)
    status.update_finish_gpt()


if __name__ == "__main__":
    #test_gpt()
    run_main_process()
