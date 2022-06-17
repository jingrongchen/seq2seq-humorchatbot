#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.cost import cross_entropy_seq, cross_entropy_seq_with_mask
from tqdm import tqdm
from sklearn.utils import shuffle
from data.twitter import data
from tensorlayer.models.seq2seq import Seq2seq
from tensorlayer.models.seq2seq_with_attention import Seq2seqLuongAttention
import os


def initial_setup(data_corpus):
    metadata, idx_q, idx_a = data.load_data(PATH='data/{}/'.format(data_corpus))
    (trainX, trainY), (testX, testY), (validX, validY) = data.split_dataset(idx_q, idx_a)
    trainX = tl.prepro.remove_pad_sequences(trainX.tolist())
    trainY = tl.prepro.remove_pad_sequences(trainY.tolist())
    testX = tl.prepro.remove_pad_sequences(testX.tolist())
    testY = tl.prepro.remove_pad_sequences(testY.tolist())
    validX = tl.prepro.remove_pad_sequences(validX.tolist())
    validY = tl.prepro.remove_pad_sequences(validY.tolist())
    return metadata, trainX, trainY, testX, testY, validX, validY



if __name__ == "__main__":
    data_corpus = "reddit"

    #data preprocessing
    metadata, trainX, trainY, testX, testY, validX, validY = initial_setup(data_corpus)

    # Parameters
    src_len = len(trainX)
    tgt_len = len(trainY)

    assert src_len == tgt_len

    batch_size = 64
    n_step = src_len // batch_size
    src_vocab_size = len(metadata['idx2w']) # 8002 (0~8001)
    emb_dim = 1024

    word2idx = metadata['w2idx']   # dict  word 2 index
    idx2word = metadata['idx2w']   # list index 2 word

    unk_id = word2idx['unk']   # 1
    pad_id = word2idx['_']     # 0

    start_id = src_vocab_size  # 8002
    end_id = src_vocab_size + 1  # 8003

    word2idx.update({'start_id': start_id})
    word2idx.update({'end_id': end_id})
    idx2word = idx2word + ['start_id', 'end_id']

    src_vocab_size = tgt_vocab_size = src_vocab_size + 2

    num_epochs = 100
    vocabulary_size = src_vocab_size
    


    def inference(seed, top_n):
        model_.eval()
        seed_id = [word2idx.get(w, unk_id) for w in seed.split(" ")]
        sentence_id = model_(inputs=[[seed_id]], seq_length=20, start_token=start_id, top_n = top_n)
        sentence = []
        for w_id in sentence_id[0]:
            w = idx2word[w_id]
            if w == 'end_id':
                break
            sentence = sentence + [w]
        return sentence

    decoder_seq_length = 156
    model_ = Seq2seq(
        decoder_seq_length = decoder_seq_length,
        cell_enc=tf.keras.layers.GRUCell,
        cell_dec=tf.keras.layers.GRUCell,
        n_layer=4,
        n_units=256,
        embedding_layer=tl.layers.Embedding(vocabulary_size=vocabulary_size, embedding_size=emb_dim),
        )
    

    # Uncomment below statements if you have already saved the model

    # load_weights = tl.files.load_npz(name='model.npz')
    # tl.files.assign_weights(load_weights, model_)

    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    model_.train()


    for epoch in range(num_epochs):
        model_.train()
        trainX, trainY = shuffle(trainX, trainY, random_state=0)
        total_loss, n_iter = 0, 0
        for X, Y in tqdm(tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=batch_size, shuffle=False), 
                        total=n_step, desc='Epoch[{}/{}]'.format(epoch + 1, num_epochs), leave=False):

            X = tl.prepro.pad_sequences(X)
            _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=end_id)
            _target_seqs = tl.prepro.pad_sequences(_target_seqs, maxlen=decoder_seq_length)
            _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=start_id, remove_last=False)
            _decode_seqs = tl.prepro.pad_sequences(_decode_seqs, maxlen=decoder_seq_length)
            _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

            with tf.GradientTape() as tape:
                ## compute outputs
                output = model_(inputs = [X, _decode_seqs])
                
                output = tf.reshape(output, [-1, vocabulary_size])
                ## compute loss and update model
                loss = cross_entropy_seq_with_mask(logits=output, target_seqs=_target_seqs, input_mask=_target_mask)

                grad = tape.gradient(loss, model_.all_weights)
                optimizer.apply_gradients(zip(grad, model_.all_weights))
            
            total_loss += loss
            n_iter += 1

        # printing average loss after every epoch
        print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, num_epochs, total_loss / n_iter))


        seeds = [
                    "Why will a Tesla scandal be exciting?",
                 "I’m sorry” and “I apologize” mean the same thing...",
                 "Knock - Knock Who's there?",
                 "Did you hear about the new German microwave?",
                 "Where do robo-babies come from?",
                 "What did the doe say when it left the forest?",
                 "What did the indian child say to his mother before he left for school?",
                 "Why was the electrochemical cell arrested?",
                 "What kind of soda did moses drink?",
                 "What do you call a dead composer?",
                 "I hate people who make cancer jokes.",
                 "main difference between /news/ mods and north korea?",
                 "How do you foil a plan?",
                 "Did you hear about the french cheese factory explosion?",
                 "My boss is going to fire the employee with the worst posture.",
                 "My addiction to computer gaming started when my family bought a PC in the 90's...",
                 "Why didn't Darwin cut off his beard?",
                 "Why did the horse feel famous on reddit?",
                 "Where do pirates store their files?",
                 "What did they say about the atheist seminary?",
                 "What did the capitalist uncle say to his soviet nephew?",
                 "How can you tell the gender of a chromosome?",
                 "How does reddit feel about civil war jokes?",
                 "A German taught me how to crack eggs today.",
                 "Why did the man on LSD cross the road?",
                 "What do you get for a nun who wears men's clothes and likes outdated electronics?",
                 "What does a skeleton orders at a restaurant?",
                 "Plastics I recently gave up plastic straws and plastics in general.",
                 "What do you call a group of friends who happen to be Muslims?",
                 "The snow in the UK is pretty bad right now So I thought I’d check on my elderly 85 year old neighbour Valerie to see if she needed anything from the shops.",
                 "Why does jesus jaywalk?",
                 "Why i un-installed league of legends.",
                 "Why did princess leia cry at the end of return of the jedi?",
                 "Find a girl who's a good driver.",
                 "Why did the cat cross the road?",
                 "The other night me and my girlfriend had an argument..",
                 "Fruit by the foot, but no meat by the meter?",
                 "/r/jokes Must be full of insecure men...",
                 "What happens when two same pokemons meet eachother?",
                 "Know why Trump supporters are so obsessed with cuck?",
                 "What has a bottom at its top?"]

        for seed in seeds:
            print("Query >", seed)
            top_n = 1
            for i in range(top_n):
                sentence = inference(seed, top_n)
                print(" >", ' '.join(sentence))

        tl.files.save_npz(model_.all_weights, name='model.npz')


        
    
    
