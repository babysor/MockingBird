#!/usr/bin/env python3

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Common functions for ASR."""

import argparse
import editdistance
import json
import logging
import numpy as np
import six
import sys

from itertools import groupby


def end_detect(ended_hyps, i, M=3, D_end=np.log(1 * np.exp(-10))):
    """End detection.

    desribed in Eq. (50) of S. Watanabe et al
    "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition"

    :param ended_hyps:
    :param i:
    :param M:
    :param D_end:
    :return:
    """
    if len(ended_hyps) == 0:
        return False
    count = 0
    best_hyp = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[0]
    for m in six.moves.range(M):
        # get ended_hyps with their length is i - m
        hyp_length = i - m
        hyps_same_length = [x for x in ended_hyps if len(x['yseq']) == hyp_length]
        if len(hyps_same_length) > 0:
            best_hyp_same_length = sorted(hyps_same_length, key=lambda x: x['score'], reverse=True)[0]
            if best_hyp_same_length['score'] - best_hyp['score'] < D_end:
                count += 1

    if count == M:
        return True
    else:
        return False


# TODO(takaaki-hori): add different smoothing methods
def label_smoothing_dist(odim, lsm_type, transcript=None, blank=0):
    """Obtain label distribution for loss smoothing.

    :param odim:
    :param lsm_type:
    :param blank:
    :param transcript:
    :return:
    """
    if transcript is not None:
        with open(transcript, 'rb') as f:
            trans_json = json.load(f)['utts']

    if lsm_type == 'unigram':
        assert transcript is not None, 'transcript is required for %s label smoothing' % lsm_type
        labelcount = np.zeros(odim)
        for k, v in trans_json.items():
            ids = np.array([int(n) for n in v['output'][0]['tokenid'].split()])
            # to avoid an error when there is no text in an uttrance
            if len(ids) > 0:
                labelcount[ids] += 1
        labelcount[odim - 1] = len(transcript)  # count <eos>
        labelcount[labelcount == 0] = 1  # flooring
        labelcount[blank] = 0  # remove counts for blank
        labeldist = labelcount.astype(np.float32) / np.sum(labelcount)
    else:
        logging.error(
            "Error: unexpected label smoothing type: %s" % lsm_type)
        sys.exit()

    return labeldist


def get_vgg2l_odim(idim, in_channel=3, out_channel=128, downsample=True):
    """Return the output size of the VGG frontend.

    :param in_channel: input channel size
    :param out_channel: output channel size
    :return: output size
    :rtype int
    """
    idim = idim / in_channel
    if downsample:
        idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 1st max pooling
        idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 2nd max pooling
    return int(idim) * out_channel  # numer of channels


class ErrorCalculator(object):
    """Calculate CER and WER for E2E_ASR and CTC models during training.

    :param y_hats: numpy array with predicted text
    :param y_pads: numpy array with true (target) text
    :param char_list:
    :param sym_space:
    :param sym_blank:
    :return:
    """

    def __init__(self, char_list, sym_space, sym_blank, report_cer=False, report_wer=False,
                 trans_type="char"):
        """Construct an ErrorCalculator object."""
        super(ErrorCalculator, self).__init__()

        self.report_cer = report_cer
        self.report_wer = report_wer
        self.trans_type = trans_type
        self.char_list = char_list
        self.space = sym_space
        self.blank = sym_blank
        self.idx_blank = self.char_list.index(self.blank)
        if self.space in self.char_list:
            self.idx_space = self.char_list.index(self.space)
        else:
            self.idx_space = None

    def __call__(self, ys_hat, ys_pad, is_ctc=False):
        """Calculate sentence-level WER/CER score.

        :param torch.Tensor ys_hat: prediction (batch, seqlen)
        :param torch.Tensor ys_pad: reference (batch, seqlen)
        :param bool is_ctc: calculate CER score for CTC
        :return: sentence-level WER score
        :rtype float
        :return: sentence-level CER score
        :rtype float
        """
        cer, wer = None, None
        if is_ctc:
            return self.calculate_cer_ctc(ys_hat, ys_pad)
        elif not self.report_cer and not self.report_wer:
            return cer, wer

        seqs_hat, seqs_true = self.convert_to_char(ys_hat, ys_pad)
        if self.report_cer:
            cer = self.calculate_cer(seqs_hat, seqs_true)

        if self.report_wer:
            wer = self.calculate_wer(seqs_hat, seqs_true)
        return cer, wer

    def calculate_cer_ctc(self, ys_hat, ys_pad):
        """Calculate sentence-level CER score for CTC.

        :param torch.Tensor ys_hat: prediction (batch, seqlen)
        :param torch.Tensor ys_pad: reference (batch, seqlen)
        :return: average sentence-level CER score
        :rtype float
        """
        cers, char_ref_lens = [], []
        for i, y in enumerate(ys_hat):
            y_hat = [x[0] for x in groupby(y)]
            y_true = ys_pad[i]
            seq_hat, seq_true = [], []
            for idx in y_hat:
                idx = int(idx)
                if idx != -1 and idx != self.idx_blank and idx != self.idx_space:
                    seq_hat.append(self.char_list[int(idx)])

            for idx in y_true:
                idx = int(idx)
                if idx != -1 and idx != self.idx_blank and idx != self.idx_space:
                    seq_true.append(self.char_list[int(idx)])
            if self.trans_type == "char":
                hyp_chars = "".join(seq_hat)
                ref_chars = "".join(seq_true)
            else:
                hyp_chars = " ".join(seq_hat)
                ref_chars = " ".join(seq_true)

            if len(ref_chars) > 0:
                cers.append(editdistance.eval(hyp_chars, ref_chars))
                char_ref_lens.append(len(ref_chars))

        cer_ctc = float(sum(cers)) / sum(char_ref_lens) if cers else None
        return cer_ctc

    def convert_to_char(self, ys_hat, ys_pad):
        """Convert index to character.

        :param torch.Tensor seqs_hat: prediction (batch, seqlen)
        :param torch.Tensor seqs_true: reference (batch, seqlen)
        :return: token list of prediction
        :rtype list
        :return: token list of reference
        :rtype list
        """
        seqs_hat, seqs_true = [], []
        for i, y_hat in enumerate(ys_hat):
            y_true = ys_pad[i]
            eos_true = np.where(y_true == -1)[0]
            eos_true = eos_true[0] if len(eos_true) > 0 else len(y_true)
            # To avoid wrong higher WER than the one obtained from the decoding
            # eos from y_true is used to mark the eos in y_hat
            # because of that y_hats has not padded outs with -1.
            seq_hat = [self.char_list[int(idx)] for idx in y_hat[:eos_true]]
            seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
            # seq_hat_text = "".join(seq_hat).replace(self.space, ' ')
            seq_hat_text = " ".join(seq_hat).replace(self.space, ' ')
            seq_hat_text = seq_hat_text.replace(self.blank, '')
            # seq_true_text = "".join(seq_true).replace(self.space, ' ')
            seq_true_text = " ".join(seq_true).replace(self.space, ' ')
            seqs_hat.append(seq_hat_text)
            seqs_true.append(seq_true_text)
        return seqs_hat, seqs_true

    def calculate_cer(self, seqs_hat, seqs_true):
        """Calculate sentence-level CER score.

        :param list seqs_hat: prediction
        :param list seqs_true: reference
        :return: average sentence-level CER score
        :rtype float
        """
        char_eds, char_ref_lens = [], []
        for i, seq_hat_text in enumerate(seqs_hat):
            seq_true_text = seqs_true[i]
            hyp_chars = seq_hat_text.replace(' ', '')
            ref_chars = seq_true_text.replace(' ', '')
            char_eds.append(editdistance.eval(hyp_chars, ref_chars))
            char_ref_lens.append(len(ref_chars))
        return float(sum(char_eds)) / sum(char_ref_lens)

    def calculate_wer(self, seqs_hat, seqs_true):
        """Calculate sentence-level WER score.

        :param list seqs_hat: prediction
        :param list seqs_true: reference
        :return: average sentence-level WER score
        :rtype float
        """
        word_eds, word_ref_lens = [], []
        for i, seq_hat_text in enumerate(seqs_hat):
            seq_true_text = seqs_true[i]
            hyp_words = seq_hat_text.split()
            ref_words = seq_true_text.split()
            word_eds.append(editdistance.eval(hyp_words, ref_words))
            word_ref_lens.append(len(ref_words))
        return float(sum(word_eds)) / sum(word_ref_lens)


class ErrorCalculatorTrans(object):
    """Calculate CER and WER for transducer models.

    Args:
        decoder (nn.Module): decoder module
        args (Namespace): argument Namespace containing options
        report_cer (boolean): compute CER option
        report_wer (boolean): compute WER option

    """

    def __init__(self, decoder, args, report_cer=False, report_wer=False):
        """Construct an ErrorCalculator object for transducer model."""
        super(ErrorCalculatorTrans, self).__init__()

        self.dec = decoder

        recog_args = {'beam_size': args.beam_size,
                      'nbest': args.nbest,
                      'space': args.sym_space,
                      'score_norm_transducer': args.score_norm_transducer}

        self.recog_args = argparse.Namespace(**recog_args)

        self.char_list = args.char_list
        self.space = args.sym_space
        self.blank = args.sym_blank

        self.report_cer = args.report_cer
        self.report_wer = args.report_wer

    def __call__(self, hs_pad, ys_pad):
        """Calculate sentence-level WER/CER score for transducer models.

        Args:
            hs_pad (torch.Tensor): batch of padded input sequence (batch, T, D)
            ys_pad (torch.Tensor): reference (batch, seqlen)

        Returns:
            (float): sentence-level CER score
            (float): sentence-level WER score

        """
        cer, wer = None, None

        if not self.report_cer and not self.report_wer:
            return cer, wer

        batchsize = int(hs_pad.size(0))
        batch_nbest = []

        for b in six.moves.range(batchsize):
            if self.recog_args.beam_size == 1:
                nbest_hyps = self.dec.recognize(hs_pad[b], self.recog_args)
            else:
                nbest_hyps = self.dec.recognize_beam(hs_pad[b], self.recog_args)
            batch_nbest.append(nbest_hyps)

        ys_hat = [nbest_hyp[0]['yseq'][1:] for nbest_hyp in batch_nbest]

        seqs_hat, seqs_true = self.convert_to_char(ys_hat, ys_pad.cpu())

        if self.report_cer:
            cer = self.calculate_cer(seqs_hat, seqs_true)

        if self.report_wer:
            wer = self.calculate_wer(seqs_hat, seqs_true)

        return cer, wer

    def convert_to_char(self, ys_hat, ys_pad):
        """Convert index to character.

        Args:
            ys_hat (torch.Tensor): prediction (batch, seqlen)
            ys_pad (torch.Tensor): reference (batch, seqlen)

        Returns:
            (list): token list of prediction
            (list): token list of reference

        """
        seqs_hat, seqs_true = [], []

        for i, y_hat in enumerate(ys_hat):
            y_true = ys_pad[i]

            eos_true = np.where(y_true == -1)[0]
            eos_true = eos_true[0] if len(eos_true) > 0 else len(y_true)

            seq_hat = [self.char_list[int(idx)] for idx in y_hat[:eos_true]]
            seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]

            seq_hat_text = "".join(seq_hat).replace(self.space, ' ')
            seq_hat_text = seq_hat_text.replace(self.blank, '')
            seq_true_text = "".join(seq_true).replace(self.space, ' ')

            seqs_hat.append(seq_hat_text)
            seqs_true.append(seq_true_text)

        return seqs_hat, seqs_true

    def calculate_cer(self, seqs_hat, seqs_true):
        """Calculate sentence-level CER score for transducer model.

        Args:
            seqs_hat (torch.Tensor): prediction (batch, seqlen)
            seqs_true (torch.Tensor): reference (batch, seqlen)

        Returns:
            (float): average sentence-level CER score

        """
        char_eds, char_ref_lens = [], []

        for i, seq_hat_text in enumerate(seqs_hat):
            seq_true_text = seqs_true[i]
            hyp_chars = seq_hat_text.replace(' ', '')
            ref_chars = seq_true_text.replace(' ', '')

            char_eds.append(editdistance.eval(hyp_chars, ref_chars))
            char_ref_lens.append(len(ref_chars))

        return float(sum(char_eds)) / sum(char_ref_lens)

    def calculate_wer(self, seqs_hat, seqs_true):
        """Calculate sentence-level WER score for transducer model.

        Args:
            seqs_hat (torch.Tensor): prediction (batch, seqlen)
            seqs_true (torch.Tensor): reference (batch, seqlen)

        Returns:
            (float): average sentence-level WER score

        """
        word_eds, word_ref_lens = [], []

        for i, seq_hat_text in enumerate(seqs_hat):
            seq_true_text = seqs_true[i]
            hyp_words = seq_hat_text.split()
            ref_words = seq_true_text.split()

            word_eds.append(editdistance.eval(hyp_words, ref_words))
            word_ref_lens.append(len(ref_words))

        return float(sum(word_eds)) / sum(word_ref_lens)
