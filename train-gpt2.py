import os
import pandas as pd
import requests

from tqdm import tqdm
tqdm.pandas()

from pandarallel import pandarallel

from gensim.summarization import keywords
from gpt2_client import GPT2Client
import gpt_2_simple as gpt2
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)
pandarallel.initialize(progress_bar=True)
def parse_csv():
    df = pd.read_csv("./train.csv")
    print(df.head(5))
    def kwp(text):
        line = ', '.join(keywords(text, words=13).split('\n'))
        return line

    df['keywords'] = df.abstract.parallel_apply(kwp)
    df.to_csv("parsed.csv")

def generate_text_corpus():
    df = pd.read_csv("./parsed.csv")
    print(df.head(5))
    text = ''
    i = 0

    for index, row in df.iterrows():
        line = ''
        #print(row['keywords'])
        #print(row['title'])
        line  += "<START> <TEXT:> " + str(row['abstract'])
        line  += "<KEYS:> " + str(row['keywords'])
        line += "; <TITLE:> " + str(row['title']) + " <END>"
        line  += "\n"
        text += line;
        i = i + 1
        if i % 1000 == 0:
            print("stage 2: " + str(i))
            print("sample: " + line)
    outF = open("parsed-train.txt", "w")
    outF.write(text)
    outF.close()


class GPT2EC(GPT2Client):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    def finetune(self, corpus, steps=1000, return_text=True):
        print("finetuning")
        sess = gpt2.start_tf_sess()
        print("starting session")
        gpt2.finetune(sess,
                corpus,
                model_name=self.model_name,
                steps=steps,
                multi_gpu=True)     # steps is max number of training steps

        if return_text:
            text = gpt2.generate(sess, return_as_list=True)
            return text
        else:
            gpt2.generate(sess)

    def load(self, sess):
        sess = gpt2.start_tf_sess()
        gpt2.load_gpt2(sess)

        gpt2.generate(sess)

def train_on_corpus():
    gpt2c = GPT2EC('774M', save_dir='models') # This could also be `345M`, `774M`, or `1558M`
    gpt2c.load_model()
    my_corpus = './parsed-train.txt' # path to corpus
    custom_text = gpt2c.finetune(my_corpus, 50000) # Load your custom dataset


parse_csv()
print("\n---\nparsed!\n---\n")
generate_text_corpus()
print("\n---\ngenerated corpys!\n---\n")
train_on_corpus()
