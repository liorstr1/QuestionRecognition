from parrot import Parrot
from transformers import (
    PegasusForConditionalGeneration, PegasusTokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM
)
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import pipeline

from helper_methods import save_dict_to_csv

NUM_OF_VARIANTS_PER_MODEL = 5


class CreateParaphrase:
    def __init__(self):
        self.parrot_model = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")
        self.vamsi_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        self.vamsi_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        self.paraphrase_generator = pipeline("text2text-generation", model="t5-base")
        self.bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.pegasus_model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
        self.pegasus_tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")
        self.results = {}

    def run_sentences(self, sentences, question_enrich_path=None):
        for idx, sentence in enumerate(sentences):
            if idx % 5 == 0:
                print(f'running {idx} out of {len(sentences)}')
            parrot_sentences = using_parrot_model(sentence, self.parrot_model)
            bart_res = using_bart_model(sentence, self.bart_model, self.bart_tokenizer)
            res_pegasus = using_pegasus(sentence, self.pegasus_model, self.pegasus_tokenizer)
            res_vesmi = using_vesmi(sentence, self.vamsi_model, self.vamsi_tokenizer)
            self.results[sentence] = list(set(parrot_sentences + bart_res + res_pegasus + res_vesmi))
            if question_enrich_path is not None:
                save_dict_to_csv(self.results, question_enrich_path)


def using_vesmi(sentence, model, tokenizer):
    try:
        # tokenize the text to be form of a list of token IDs
        inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
        # generate the paraphrased sentences
        outputs = model.generate(
            **inputs,
            num_beams=10,
            num_return_sequences=10,
        )
        # decode the generated sentences using the tokenizer to get them back to text
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)
    except Exception as e:
        print(e.args)
        return []


def using_pegasus(sentence, model, tokenizer):
    try:
        # tokenize the text to be form of a list of token IDs
        inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
        # generate the paraphrased sentences
        outputs = model.generate(
            **inputs,
            num_beams=10,
            num_return_sequences=10,
        )
        # decode the generated sentences using the tokenizer to get them back to text
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)
    except Exception as e:
        print(e.args)
        return []


def using_bart_model(sentence, bart_model, bart_tokenizer):
    try:
        inputs = bart_tokenizer(sentence, return_tensors='pt')
        # Generate paraphrases
        paraphrases = bart_model.generate(
            **inputs,
            max_length=128,
            num_beams=5,
            num_return_sequences=5,
            temperature=1.5
        )

        # Decode paraphrases
        return bart_tokenizer.batch_decode(paraphrases, skip_special_tokens=True)
    except Exception as e:
        print(e.args)
        return []


def using_parrot_model(sentence, model):
    try:
        paras = model.augment(input_phrase=sentence, use_gpu=False)
        if paras is None or not paras:
            return []
        return [para[0] for para in paras][:NUM_OF_VARIANTS_PER_MODEL]
    except Exception as e:
        print(e.args)
        return []
