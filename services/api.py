from chat_with_gpt.main_gpt_process import ChatWithGPT
from entities import Status
from services.prompts_and_texts import INITIAL_CONTACT


def analyze_init_user_answer(user_answer, status: Status):
    cwg = ChatWithGPT()
    response = cwg.analyze_first_message(INITIAL_CONTACT, user_answer)
    print(f'user picked to:{response}')
    if response == 'continue':
        status.init_continue()
    elif response == 'stop':
        status.end_session()
    else:
        # TODO: add option for postpone/unclear
        status.end_session()


def fill_next_question_using_gpt(next_question):
    messages = []
    cwg = ChatWithGPT()
    cwg.fill_init_message_with_struct_json(messages, json_struct)

