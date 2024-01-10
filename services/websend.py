from selenium.common.exceptions import NoSuchElementException
import openai

from services.api import analyze_init_user_answer
from services.prompts_and_texts import INITIAL_CONTACT
from services.websend_helper_methods import *

IS_CALL_API_FLAG = False  # Global flag


def run_main_process():
    driver, status, last_incoming_message, user_data, client_data = init_data(
        user_name='ליאור שטראוס'
    )
    last_incoming_message = process_existing_messages(driver)
    send_outgoing_message(driver, INITIAL_CONTACT)
    status.update_init_asked()

    # Main loop to continuously check for new messages and respond
    while status.keep_running:
        incoming_message_content = continuously_check_for_new_messages(driver, last_incoming_message)
        if incoming_message_content:
            call_api(driver, incoming_message_content, status, user_data, client_data)
            last_incoming_message = incoming_message_content  # Update the last message content
    driver.quit()


def process_existing_messages(driver):
    time.sleep(5)
    # Locate all messages in the chat
    messages = driver.find_elements(By.XPATH, "//div[contains(@class, '_1AOLJ _2UtSC _1jHIY')]")

    last_incoming_message = None
    for message in messages:
        # Determine if the message is incoming or outgoing
        if 'message-in' in message.get_attribute('class'):
            last_incoming_message = message  # Update last incoming message

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
            f"Type: {'incoming' if last_incoming_message == message else 'outgoing'}, Timestamp: {timestamp}, "
            f"Message: {message_content}")

    # Optionally send an initial message (if this function is implemented)
    # send_initial_message(driver)

    # Return the last incoming message element for further processing
    return last_incoming_message


def continuously_check_for_new_messages(driver, last_incoming_message):
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
            if check_latest_message(latest_message, last_incoming_message):
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


def call_api(driver, message_content, status: Status, user_data, client_data):
    global IS_CALL_API_FLAG
    IS_CALL_API_FLAG = True
    print("start")

    if status.is_init_answer():
        analyze_init_user_answer(message_content, status)
    if status.is_fill_json():
        json_path, json_struct = get_updated_json(client_data, user_data)
        while check_json_struct(json_struct):
            next_question = get_next_question(json_struct)
            fill_next_question_using_gpt(next_question)




    def get_openai_response(conversation):
        print("start GPT")
        client = openai.OpenAI(api_key='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')
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation
        )
        return response.choices[0].message.content

    # Prepare the conversation history and add the initial system message
    conversation_history = [{"role": "system", "content": "System initialization message"}]

    # Add the incoming user message to the conversation history
    conversation_history.append({"role": "user", "content": message_content})

    # Get the response from OpenAI
    gpt_response = get_openai_response(conversation_history)

    # Set the flag to False to resume message checking
    is_call_api_active = False

    def is_response_related(question, response):
        print("start GPT2")
        response = response.strip()
        if "אתה יכול לספר לי על הנכס שלך מעט ?" in question:
            return len(response.split()) > 5
        elif "מה התקציב שלך לנכס?" in question:
            return any(char.isdigit() for char in response)
        elif "האם אתה גמיש בתקציב שלך לנכס?" in question:
            return response in ["כן", "לא", "אולי"]
        else:
            return False

    form_questions = [
        "אתה יכול לספר לי על הנכס שלך מעט ?",
        "מה התקציב שלך לנכס?",
        "האם אתה גמיש בתקציב שלך לנכס?"
    ]

    form_responses = {}
    conversation_history = [{"role": "system",
                             "content": "אתה מייצג סוכן נדלן ואתה אמור לשכנע את הלקוח להשתמש בשירותים שלך בצורה מכירתית. התשובות שלך צריכות להיות קצרות ומדוייקות בהתאם לשאלות הלקוח. עלייך לקדם את השיחה בהתאם. "}]
    current_question_index = 0
    gpt_response_count = 0
    print("the message for GPT is: ", message_content)

    while current_question_index < len(form_questions):
        user_input = message_content
        conversation_history.append({"role": "user", "content": user_input})

        if current_question_index not in form_responses and not is_response_related(
                form_questions[current_question_index], user_input):
            if gpt_response_count < 2:
                gpt_response = get_openai_response(conversation_history)
                conversation_history.append({"role": "assistant", "content": gpt_response})
                print(gpt_response)
                gpt_response_count += 1
                break
            else:
                question = form_questions[current_question_index]
                print(question)
                conversation_history.append({"role": "assistant", "content": question})
                gpt_response_count = 0
                break

        else:
            form_responses[current_question_index] = user_input
            current_question_index += 1
            gpt_response_count = 0
            break

    print("\nResponses:")
    for i, response in form_responses.items():
        print(f"{form_questions[i]} {response}")

    # Send the last response received from the conversation history
    if conversation_history:
        last_response = conversation_history[-1].get("content")
        if last_response:
            send_response(driver, last_response)

    is_call_api_active = False  # Reset flag when done
    last_incoming_message = gpt_response
    continuously_check_for_new_messages(driver, last_incoming_message)


if __name__ == "__main__":
    run_main_process()
