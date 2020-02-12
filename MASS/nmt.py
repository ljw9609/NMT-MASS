#!/usr/bin/env python3 -u
from MASS.mass import xmasked_seq2seq
from collections import namedtuple

import os

import nltk
import jieba
import torch

from fairseq import checkpoint_utils, options, tasks, utils
from subword_nmt.apply_bpe import BPE

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')

cur_path = os.path.dirname(__file__)
_data_dir = os.path.join(cur_path, './model/data')
_user_dir = os.path.join(cur_path, './mass')
_model = os.path.join(cur_path, './model/zhen_mass_pre-training.pt')
_bpe_codes_en = os.path.join(cur_path, './model/bpe/all.en.bpe.codes')
_bpe_codes_zh = os.path.join(cur_path, './model/bpe/all.zh.bpe.codes')


class Translator:
    def __init__(self):
        self.parser = None
        self.args = None
        self.task = None
        self.models = None
        self.model = None
        self.src_dict, self.tgt_dict = None, None
        self.generator = None
        self.align_dict = None
        self.max_positions = None
        self.decoder = None
        self.encode_fn = None
        self.use_cuda = True
        self.src = 'en'
        self.tgt = 'zh'

        self.bpe = None
        self.tokenizer = True

    def initialize(self, data_dir=_data_dir, model_path=_model,
                   user_dir=_user_dir, task='xmasked_seq2seq',
                   s_lang='en', t_lang='zh', beam=5, cpu=False, align_dict=None,
                   bpe_codes=_bpe_codes_en, tokenizer=True):
        self.parser = options.get_generation_parser(interactive=True)
        self.src, self.tgt = s_lang, t_lang

        # generate args
        input_args = [data_dir, '--path', model_path]
        if cpu:
            input_args.append('--cpu')
        if user_dir:
            input_args.append('--user-dir')
            input_args.append(user_dir)
        if task:
            input_args.append('--task')
            input_args.append(task)
        if align_dict:
            input_args.append('--replace-unk')
            input_args.append(align_dict)
        input_args.append('--langs')
        input_args.append('{},{}'.format(s_lang, t_lang))
        input_args.append('--source-langs')
        input_args.append(s_lang)
        input_args.append('--target-langs')
        input_args.append(t_lang)
        input_args.append('-s')
        input_args.append(s_lang)
        input_args.append('-t')
        input_args.append(t_lang)
        input_args.append('--beam')
        input_args.append(str(beam))
        input_args.append('--remove-bpe')

        self.bpe = BPE(open(bpe_codes, 'r'))
        self.tokenizer = tokenizer

        self.args = options.parse_args_and_arch(self.parser, input_args=input_args)

        # initialize model
        utils.import_user_module(self.args)

        if self.args.buffer_size < 1:
            self.args.buffer_size = 1
        if self.args.max_tokens is None and self.args.max_sentences is None:
            self.args.max_sentences = 1

        assert not self.args.sampling or self.args.nbest == self.args.beam, \
            '--sampling requires --nbest to be equal to --beam'
        assert not self.args.max_sentences or self.args.max_sentences <= self.args.buffer_size, \
            '--max-sentences/--batch-size cannot be larger than --buffer-size'

        self.use_cuda = torch.cuda.is_available() and not self.args.cpu

        # Setup task, e.g., translation
        self.task = tasks.setup_task(self.args)

        # Load ensemble
        self.models, _model_args = checkpoint_utils.load_model_ensemble(
            self.args.path.split(':'),
            arg_overrides=eval(self.args.model_overrides),
            task=self.task,
        )

        # Set dictionaries
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        # Optimize ensemble for generation
        for model in self.models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if self.args.no_beamable_mm else self.args.beam,
                need_attn=self.args.print_alignment,
            )
            if self.args.fp16:
                model.half()
            if self.use_cuda:
                model.cuda()

        # Initialize generator
        self.generator = self.task.build_generator(self.args)

        def encode_fn(x):
            if tokenizer:
                x = tokenize(x, is_zh=(s_lang == 'zh'))
            if bpe_codes:
                x = self.bpe.process_line(x)
            return x

        # Hack to support GPT-2 BPE
        if self.args.remove_bpe == 'gpt2':
            pass
        else:
            self.decoder = None
            # self.encode_fn = lambda x: x
            self.encode_fn = encode_fn

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(self.args.replace_unk)

        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(),
            *[model.max_positions() for model in self.models]
        )

    def translate(self, text, verbose=False):
        start_id = 0
        inputs = [text]
        #inputs = [text.lower()]
        #inputs = [tokenize(text, is_zh=(self.src == 'zh'))]
        results = []
        outputs = []
        for batch in self.make_batches(inputs, self.args, self.task, self.max_positions, self.encode_fn):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if self.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            translations = self.task.inference_step(self.generator,self. models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))

        # sort output to match input order
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if self.src_dict is not None:
                src_str = self.src_dict.string(src_tokens, self.args.remove_bpe)
                if verbose:
                    print('S-{}\t{}'.format(id, src_str))

            # Process top predictions
            for hypo in hypos[:min(len(hypos), self.args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    align_dict=self.align_dict,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=self.args.remove_bpe,
                )
                if self.decoder is not None:
                    hypo_str = self.decoder.decode(map(int, hypo_str.strip().split()))
                outputs.append(hypo_str)
                if verbose:
                    print('H-{}\t{}\t{}'.format(id, hypo['score'], hypo_str))
                    print('P-{}\t{}'.format(
                        id,
                        ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
                    ))
                if self.args.print_alignment and verbose:
                    print('A-{}\t{}'.format(
                        id,
                        ' '.join(map(lambda x: str(utils.item(x)), alignment))
                    ))
        return ''.join(''.join(outputs).split(' ')) if self.src == 'en' else ' '.join(''.join(outputs).split(' '))

    def make_batches(self, lines, args, task, max_positions, encode_fn):
        tokens = [
            task.source_dictionary.encode_line(
                encode_fn(src_str), add_if_not_exist=False
            ).long()
            for src_str in lines
        ]
        lengths = torch.LongTensor([t.numel() for t in tokens])
        itr = task.get_batch_iterator(
            dataset=task.build_dataset_for_inference(tokens, lengths),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=max_positions,
        ).next_epoch_itr(shuffle=False)
        for batch in itr:
            yield Batch(
                ids=batch['id'],
                src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
            )


def tokenize(line, is_zh=False, lower_case=True, delim=' '):
    # replace non-breaking whitespace
    _line = line.replace("\xa0", " ").strip()
    # tokenize
    _tok = jieba.cut(_line.rstrip('\r\n')) if is_zh else nltk.word_tokenize(_line)
    _tokenized = delim.join(_tok)
    # lowercase. ignore if chinese.
    _tokenized = _tokenized.lower() if lower_case and not is_zh else _tokenized
    return _tokenized


def test_en2zh(src_file, data_dir, model, user_dir, bpe_codes, N=10, beam=5, verbose=False):
    t = Translator()
    t.initialize(data_dir, model, user_dir, task='xmasked_seq2seq', bpe_codes=bpe_codes, beam=beam)

    with open(src_file, 'r') as f:
        inputs = [next(f) for i in range(N)]

    for _input_ in inputs:
        _output_ = t.translate(_input_, verbose=verbose)
        print('Source: {}Target: {}\n--------------------'.format(_input_, _output_))


def test_zh2en(src_file, data_dir, model, user_dir, bpe_codes, N=10, beam=5, verbose=False):
    t = Translator()
    t.initialize(data_dir, model, user_dir, task='xmasked_seq2seq', s_lang='zh', t_lang='en', bpe_codes=bpe_codes, beam=beam)

    with open(src_file, 'r') as f:
        inputs = [next(f) for i in range(N)]

    for _input_ in inputs:
        _output_ = t.translate(_input_, verbose=verbose)
        print('Source: {}Target: {}\n--------------------'.format(_input_, _output_))


if __name__ == '__main__':
    data_dir = './model/data'
    user_dir = './mass'
    model = './model/zhen_mass_pre-training.pt'
    bpe_codes_en = './model/bpe/all.en.bpe.codes'
    bpe_codes_zh = './model/bpe/all.zh.bpe.codes'

    # test English to Chinese
    test_en2zh('./model/data/test/valid.en', data_dir, model, user_dir, bpe_codes_en, 5, beam=5, verbose=False)

    # test Chinese to English
    test_zh2en('./model/data/test/valid.zh', data_dir, model, user_dir, bpe_codes_zh, 5, beam=10, verbose=False)
