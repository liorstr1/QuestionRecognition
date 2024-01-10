import json
import os
import openai
from chat_with_gpt.prompts import INIT_MESSAGE_PROMPT, INIT_TEXT_PROMPTS, INTERVIEWER_PROMPT


class ChatWithGPT:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = 'gpt-4-1106-preview'
        openai.api_key = self.api_key

    def get_response(self, prompt):
        from openai import OpenAI
        client = OpenAI()

        completion = client.chat.completions.create(
            model=self.model,
            messages=prompt,
            response_format={"type": "json_object"}
        )
        return completion.choices[0].message

    def analyze_first_message(self, my_question, init_message):
        prompt = [
            {
                "role": "system",
                "content": INIT_MESSAGE_PROMPT
            },
            {
                "role": "user",
                "content": INIT_TEXT_PROMPTS.format(question=my_question, answer=init_message)
            }
        ]
        response = self.get_response(prompt)
        return json.loads(response.content)['response']

    def fill_init_message_with_struct_json(self, question):
        messages = [
            {
                'role': "system",
                'content': INTERVIEWER_PROMPT.format(question=question)
            },
            {
                'role': "user",
                'content': "create your next message"
            }
        ]
        response = self.get_response(messages)
        res = json.loads(response.content)
        messages.append({
            'role': 'assistant',
            'content': res['next']
        })
        return False,  res['next'], messages

    def analyze_next_user_answer(self, messages, user_answer):
        messages.append(
            {
                'role': "user",
                'content': f"this is the user answer: {user_answer}"
            }
        )
        response = self.get_response(messages)
        res = json.loads(response.content)
        if res['answered'] is True:
            return True, res['next'], messages
        else:
            messages.append({
                'role': 'assistant',
                'content': res['next']
            })
            return False, res['next'], messages






