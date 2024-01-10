from enum import Enum

SERVICE_SETUP = 'C:/chromedriver.exe'
WHATSAPP_WEB = 'https://web.whatsapp.com'


class MainStatus(Enum):
    MY_TURN_TO_ASK = "my_turn_to_ask"
    MY_TURN_TO_ANSWER = "my_turn_to_answer"
    USER_TURN_TO_ANSWER = "user_turn_to_answer"
    USER_TURN_TO_ASK = "user_turn_to_ask"
    SESSION_ENDED = "session ended"


class SecondaryStatus(Enum):
    INIT_QUESTION = "init_question"
    FILL_JSON = "fill_json"
    USER_TURN_TO_ANSWER = "user_turn_to_answer"
    USER_TURN_TO_ASK = "user_turn_to_ask"
    CHECKING_MODELS = "checking_models"
    ASKING_GPT = "asking_gpt"
    ASKING_CLIENT = "asking_client"


class Status:
    def __init__(self):
        self.main_status: MainStatus = MainStatus.MY_TURN_TO_ASK
        self.secondary_status: SecondaryStatus = SecondaryStatus.INIT_QUESTION
        self.keep_running = True

    def init_asked(self):
        self.main_status = MainStatus.USER_TURN_TO_ANSWER
        self.secondary_status = SecondaryStatus.INIT_QUESTION

    def update_init_asked(self):
        self.main_status = MainStatus.USER_TURN_TO_ANSWER
        self.secondary_status = SecondaryStatus.INIT_QUESTION

    def is_init_answer(self):
        if (
            self.main_status == MainStatus.USER_TURN_TO_ANSWER and
            self.secondary_status == SecondaryStatus.INIT_QUESTION
        ):
            return True
        return False

    def init_continue(self):
        self.main_status = MainStatus.MY_TURN_TO_ASK
        self.secondary_status = SecondaryStatus.FILL_JSON

    def end_session(self):
        self.main_status = MainStatus.SESSION_ENDED

    def is_fill_json(self):
        if (
                self.main_status == MainStatus.MY_TURN_TO_ASK and
                self.secondary_status == SecondaryStatus.FILL_JSON
        ):
            return True
        return False

    def update_gpt_sent(self):
        self.main_status = MainStatus.USER_TURN_TO_ANSWER
        self.secondary_status = SecondaryStatus.FILL_JSON

    def update_gpt_got(self):
        self.main_status = MainStatus.MY_TURN_TO_ASK
        self.secondary_status = SecondaryStatus.FILL_JSON

    def update_finish_gpt(self):
        self.main_status = MainStatus.USER_TURN_TO_ASK
        self.secondary_status = SecondaryStatus.USER_TURN_TO_ASK


class UserData:
    def __init__(self, name, user_id):
        self.user_name: str = name
        self.user_id: str = user_id
        self.next_data_dict: dict = {}


class ClientData:
    def __init__(self, client_name, client_id):
        self.client_name: str = name
        self.client_id: str = client_id
