from collections import defaultdict

import regex
import editdistance
from ctcdecode import CTCBeamDecoder

import torch

import numpy as np


def calc_wer(predicted_text, gt_text):
    """ Compute wer.
        Inputs:
            predicted_text: str
            gt_text: str
        Returns:
            wer: int
    """
    # write your code here
    words_split_pattern = regex.compile(r'\b\w+\b')

    gt_words = regex.findall(words_split_pattern, gt_text)
    predicted_words = regex.findall(words_split_pattern, predicted_text)

    if len(gt_words) == len(predicted_words) == 0:
        return 0.0
    if len(gt_words) == 0:
        return 1.0
    wer = editdistance.eval(predicted_words, gt_words) / len(gt_words)
    return wer


def calc_wer_for_batch(list_of_predicted_text, list_of_gt_text):
    """ Compute mean wer for batch.
            Inputs:
                list_of_predicted_text: list
                list_of_gt_text: list
            Returns:int

    """

    # write your code here
    mean_wer = np.mean([calc_wer(_, __) for _, __ in zip(list_of_predicted_text, list_of_gt_text)])
    return mean_wer


def decode(alignment):
    """ Get text from alignment.
        Inputs:
            alignment: str
        Returns:
            text: srt
    """
    # write your code here
    alignment = regex.sub(r'(\p{L})\1+', r'\1', alignment)
    alignment = alignment.replace('<blank>', '')
    alignment = regex.sub(' +', ' ', alignment)
    text = alignment.strip()
    return text


def greedy_decoder(logprobs, logprobs_lens, vocab, **kwargs):
    predictions = []

    tokens = torch.argmax(logprobs, dim=-1)
    for idx in range(logprobs.shape[1]):
        alligmnet_tokens = vocab.lookup_tokens(tokens[:, idx][:logprobs_lens[idx]].detach().cpu().numpy())
        alligmnet = ''.join(alligmnet_tokens)
        hypo = decode(alligmnet)
        predictions += [[(hypo, 1.0)]]

    return predictions


def beam_search_decode(logprobs, logprobs_lens, vocab, beam_size, cutoff_top_n, cutoff_prob, ext_scoring_func, alpha):
    """
    logprobs: [num_timesteps, alphabet_len]
    """

    def _beam_search_decode(
            _logprobs, _logprobs_len,
            _vocab, _beam_size, _cutoff_top_n, _cutoff_prob, _ext_scoring_func, _alpha
    ):
        assert (_cutoff_top_n is None) or (_cutoff_prob is None)
        if (_cutoff_top_n is None) and (_cutoff_prob is None):
            _cutoff_prob = 1.0

        hypos = set()
        indices2tokens = vocab.indices2tokens()
        probs_b, probs_nb = defaultdict(float), defaultdict(float)

        hypos.add('')
        probs_b[''], probs_nb[''] = 1.0, 0.0
        probs_b_new, probs_nb_new = dict(), dict()
        for t in range(_logprobs_len):
            hypos_new = set()
            probs_b_new, probs_nb_new = defaultdict(float), defaultdict(float)

            probs_t = torch.exp(_logprobs[t]).detach().cpu()
            decrease_ids = torch.argsort(-probs_t).detach().cpu().numpy()
            if _cutoff_prob is not None:
                cond_idxs = torch.where(torch.greater(torch.cumsum(probs_t[decrease_ids], dim=0), _cutoff_prob))[0]
                if len(cond_idxs) == 0:
                    _cutoff_top_n = len(decrease_ids)
                else:
                    _cutoff_top_n = cond_idxs[0].item()
                if _cutoff_top_n == 0:
                    _cutoff_top_n = 1

            for line in hypos:
                for c_idx in decrease_ids[:_cutoff_top_n]:
                    c, c_prob = indices2tokens[c_idx], probs_t[c_idx]

                    if c == '<blank>':
                        probs_b_new[line] += c_prob * (probs_b[line] + probs_nb[line])
                    else:
                        l_end = line[-1] if len(line) > 0 else ''
                        l_plus = line + c
                        if c == l_end:
                            probs_nb_new[line] += c_prob * probs_nb[line]
                            probs_nb_new[l_plus] += c_prob * probs_b[line]
                        elif c == ' ':
                            if _ext_scoring_func is None:
                                p_w = 1.0
                            else:
                                p_w = _ext_scoring_func(line)
                            probs_b_new[l_plus] += np.power(p_w, _alpha) * c_prob * (probs_b[line] + probs_nb[line])
                        else:
                            probs_nb_new[l_plus] += c_prob * (probs_b[line] + probs_nb[line])
                        hypos_new.add(l_plus)
                hypos_new.add(line)

            hypos_new = list(
                sorted(hypos_new, key=lambda hypo: probs_b_new[hypo] + probs_nb_new[hypo], reverse=True)
            )[:min(beam_size, len(hypos_new))]

            hypos, probs_b, probs_nb = hypos_new, probs_b_new, probs_nb_new

        return sorted(
            [(hypo, probs_b_new[hypo] + probs_nb_new[hypo]) for hypo in hypos],
            key=lambda key_value: key_value[1], reverse=True
        )

    predictions = []
    for idx in range(logprobs.shape[1]):
        predicted_hypos = _beam_search_decode(
            logprobs[:, idx], logprobs_lens[idx],
            _vocab=vocab, _beam_size=beam_size,
            _cutoff_top_n=cutoff_top_n, _cutoff_prob=cutoff_prob,
            _ext_scoring_func=ext_scoring_func, _alpha=alpha
        )
        predictions.append(predicted_hypos)

    return predictions


def fast_beam_search_decode(
        logprobs, logprobs_lens, vocab, beam_size,
        cutoff_top_n, cutoff_prob, ext_scoring_func, alpha, beta,
        num_processes
):
    blank_index = vocab['<blank>']

    labels = ''.join(vocab.indices2tokens()).replace('<blank>', '_').replace('<unk>', '')
    decoder = CTCBeamDecoder(
        labels=labels, blank_id=blank_index,
        cutoff_top_n=cutoff_top_n, cutoff_prob=cutoff_prob, beam_width=beam_size,
        model_path=ext_scoring_func, alpha=alpha, beta=beta,
        num_processes=num_processes,
        log_probs_input=False
    )
    probs = torch.exp(logprobs)
    beam_results, beam_scores, timesteps, out_lens = decoder.decode(torch.transpose(probs, 0, 1), logprobs_lens)
    beam_probas = torch.exp(-beam_scores)

    predictions = []
    for idx in range(beam_results.shape[0]):
        beam = []
        for jdx in range(beam_results.shape[1]):
            hypo = ''.join(vocab.lookup_tokens(beam_results[idx, jdx, :out_lens[idx, jdx]].tolist()))
            hypo_score = beam_probas[idx, jdx]
            beam.append((hypo, hypo_score))
        predictions.append(beam)

    return predictions
