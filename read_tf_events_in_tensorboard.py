import tensorflow as tf
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--add", type=str, required=True, help='The address of tf event file; output of tensorboard')
args = parser.parse_args()

add = args.add

tag_all = []
for e in tf.train.summary_iterator(add):
    for v in e.summary.value:
        if v.tag not in tag_all:
            tag_all.append(v.tag)

print(tag_all)  # list all available tags, such as loss, acc, etc


for e in tf.train.summary_iterator(add):
    for v in e.summary.value:
        if v.tag == 'loss':  # Choice the tag you want
            print(e.step, v.tag, v.simple_value)  
