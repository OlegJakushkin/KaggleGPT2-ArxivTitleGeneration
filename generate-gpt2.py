import os
import pandas as pd
import requests

from tqdm import tqdm

tqdm.pandas()

from pandarallel import pandarallel

from gensim.summarization import keywords
from gpt2_client import *
import gpt_2_simple as gpt2
import tensorflow as tf
from itertools import groupby

from termcolor import colored, cprint
import sys
from tqdm import tqdm
import json
import regex as re

from tensorflow.contrib.training import HParams
import numpy as np
from utils import *
import multiprocessing
from math import *
from functools import partial
os.environ["CUDA_VISIBLE_DEVICES"]=str("")
def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]

pandarallel.initialize(progress_bar=True)


def parse_csv():
    df = pd.read_csv("./test.csv")
    print(df.head(5))

    def kwp(text):
        line = ', '.join(keywords(text, words=13).split('\n'))
        return line

    df['keywords'] = df.abstract.parallel_apply(kwp)
    df.to_csv("parsed-test.csv")


counter = multiprocessing.Value('i', 0)


class GPT2EC():
    def __init__(self, model_name='117M', save_dir='models'):
        assert save_dir != '', 'Please provide a save directory for the model weights and checkpoints. This cannot be empty.'

        self.model_name = model_name
        self.save_dir = save_dir
        self.enc = get_encoder(self.model_name, self.save_dir)


def encode(ctx, in_str):
    context_tokens = ctx.enc.encode(in_str)
    return context_tokens

def process_mt(ctx,  encoded_batch, gpus=2):
    p = multiprocessing.Pool(gpus)
    total = len(encoded_batch)
    chunk_size = ceil(total / gpus)
    slice = chunks(encoded_batch, chunk_size)
    func = partial(processBatch, ctx)
    r = p.map(func, slice)
    out = []
    for i in r:
        out += i
    return out


def processBatch(ctx, encoded_batch):
    global counter
    cvalue = 0
    with counter.get_lock():
        cvalue = counter.value
        counter.value += 1

    models_dir = models_dir = os.path.expanduser(os.path.expandvars(ctx.save_dir))
    hparams = default_hparams()

    with open(os.path.join(ctx.save_dir, ctx.model_name, 'hparams.json')) as f:
        data = json.load(f)
        hparams.override_from_dict(data)

    length = hparams.n_ctx
    clen = 0
    for context_tokens in encoded_batch:
        csize = len(context_tokens)
        if clen < csize:
            clen = csize

    if csize > 900:
        csize = 900
    print(cvalue)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(cvalue)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    tf.debugging.set_log_device_placement(True)
    #gpu_options = tf.GPUOptions(visible_device_list=str(cvalue))
    results = []
    with tf.Session(graph=tf.Graph()) as sess:
        batch_size = 1
        temperature = 1
        top_k = 2

        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(None)
        tf.set_random_seed(None)
        initialized = False

        output = sample_sequence(
            context=context,
            hparams=hparams,
            length=350,  # min(length, 1023 - csize),
            start_token=None,
            batch_size=batch_size,
            temperature=temperature,
            top_k=top_k
        )
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(ctx.save_dir, ctx.model_name))
        saver.restore(sess, ckpt)
        i = 0
        
        clen = len(encoded_batch)
        for context_tokens in encoded_batch:
            i = i + 1
            out = sess.run(output, feed_dict={
                context: [context_tokens]
            })[:, len(context_tokens):]
            results.append(out[0])
            if i % 2 == 0:
                print(str(cvalue) + " at " + str(i) + " from " + str(clen))
        
    final_generated_text = []
    for r in results:
        decoded = ctx.enc.decode(r)
        final_generated_text.append(decoded)

    return final_generated_text


def generate_titles():
    gpt2c = GPT2EC('run1', save_dir='checkpoint')  # This could also be `345M`, `774M`, or `1558M`
    df = pd.read_csv("./parsed-test.csv")#, nrows=10)
    print(df.head(5))

    def encodex(row):
        line = "<END>\n<START> <TEXT:> " + str(row.abstract) + " <KEYS:> " + str(row.keywords) + "; <TITLE:>"
        code = gpt2c.enc.encode(line)
        return code

    codes = df.parallel_apply(encodex, axis=1)
    titles_array = process_mt(gpt2c, codes, 2)
    df['title'] = pd.Series(titles_array, index=df.index)

    def filter(generated):
        garr = generated.split()
        res = [i[0] for i in groupby(garr)]
        s = ' '
        result = s.join(res)
        result = result.split('<END>', 1)[0]

        try:
            result = result.split('.', 1)[0]
        #    result = result.split('<TITLE:>',1)[1]
        except:
            pass
        result = re.sub(' +', ' ', result)
        result = re.sub(r"^\s+|\s+$", "", result)
        return result

    df['title'] = df.title.parallel_apply(filter)
    df.to_csv("titled.csv")


parse_csv()
print("\n---\nparsed!\n---\n")
generate_titles()
print("\n---\ngenerated corpys!\n---\n")
