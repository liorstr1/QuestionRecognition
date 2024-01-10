from entities import Status, MainStatus, SecondaryStatus
from api import *


def process_manager(status: Status, user_data):
    if status.main_status == MainStatus.MY_TURN_TO_ASK and status.secondary_status == SecondaryStatus.INIT_QUESTION:
        send_init_question_to_user()
