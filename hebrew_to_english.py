import requests
from transformers import MarianTokenizer, MarianMTModel


class TranslateSentences:
    def __init__(self):
        self.helsinki_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-tc-big-he-en')
        self.helsinki_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-tc-big-he-en')

    def return_eng_to_heb(self, hebrew_sentence):
        res = [translate_libre(hebrew_sentence)]
        res.extend(translate_with_transformers(hebrew_sentence, self.helsinki_model, self.helsinki_tokenizer))
        return res


def translate_libre(text):
    try:
        url = 'https://libretranslate.de/translate'
        params = {
            'q': text,
            'source': "he",
            'target': "en",
            'format': 'text'
        }
        response = requests.post(url, data=params)
        return response.json()['translatedText']
    except Exception as e:
        print(e.args)
        return None


def translate_with_transformers(text, model, tokenizer):
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
