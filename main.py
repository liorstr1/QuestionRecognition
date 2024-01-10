from chat_with_gpt.main_gpt_process import ChatWithGPT
from helper_methods import get_label2answer
from main_processes import train_model_process, enrichment_process, prediction_process


def model_functions():
    question_path = "C:/Users/liors/Desktop/אליאור/questions.txt"
    saved_model_path = 'C:/Users/liors/Desktop/אליאור/models'
    answer_path = "C:/Users/liors/Desktop/אליאור/תשובות נדלן.txt"

    enrichment_process(question_path)
    label2answer = get_label2answer(answer_path)
    train_model_process(question_path, saved_model_path)
    docs = [
        'איזה סוג של נכס יענה על הצרכים שלך?',
        'באילו אזורים  בארץ אתה מתעניין לרכישת נכס?'
    ]
    prediction_results = prediction_process(docs, saved_model_path)
    for q, (label, conf) in prediction_results.items():
        print(f'question: {q}')
        print(f'answer: {label2answer[label]}')
        print(f'confidence: {round(conf,3)}')


def gpt_function():
    cwg = ChatWithGPT()
    check_next, next_message, messages = cwg.fill_init_message_with_struct_json('כמה זמן הנכס עומד למכירה')
    count_round = 0
    while not check_next:
        print(f'gpt: {next_message}')
        if count_round == 0:
            print(f'client: מזג האוויר לא נעים היום')
            check_next, next_message, messages = cwg.analyze_next_user_answer(messages, 'מזג האוויר לא נעים היום')
            count_round += 1
        elif count_round == 1:
            print(f'client: שלושה חודשים')
            check_next, next_message, messages = cwg.analyze_next_user_answer(messages, 'שלושה חודשים')
    print(f'gpt: client final answer: {next_message}')


if __name__ == '__main__':
    model_functions()
    #gpt_function()
