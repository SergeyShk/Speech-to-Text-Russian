#!/usr/bin/python
import argparse
import subprocess
from pathlib import Path
import logging
from tools.utils import make_spk2utt
from kaldi.segmentation import NnetSAD, SegmentationProcessor
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.util.table import SequentialMatrixReader

class Segmenter(object):
    """Класс для сегментации аудио с помощью алгоритма обнаружения активности голоса (VAD)"""

    def __init__(self, scp, model, post, conf, output, log=False):
        """
        Инициализация сегментатора
        
        Аргументы:
            scp: путь к .SCP файлу с аудио
            model: путь к .RAW файлу модели сегментации
            post: путь к .VEC файлу апостериорных вероятностей сегментации
            conf: путь к .CONF конфигурационному файлу сегментации
            output: путь к директории с результатами сегментации
            log: признак логирования
        """  
        self.scp = scp
        self.model = model
        self.post = post
        self.conf = conf
        self.output = Path(output)
        self.log = log

        sad_model = NnetSAD.read_model(model)
        sad_post = NnetSAD.read_average_posteriors(post)
        sad_transform = NnetSAD.make_sad_transform(sad_post)
        sad_graph = NnetSAD.make_sad_graph()
        decodable_opts = NnetSimpleComputationOptions()
        decodable_opts.extra_left_context = 79
        decodable_opts.extra_right_context = 21
        decodable_opts.extra_left_context_initial = 0
        decodable_opts.extra_right_context_final = 0
        decodable_opts.frames_per_chunk = 150
        decodable_opts.acoustic_scale = 0.3
        self.sad = NnetSAD(sad_model, sad_transform, sad_graph, decodable_opts=decodable_opts)
        self.seg = SegmentationProcessor([2])
    
    def segment(self):
        """
        Выполнение сегментации
        
        Результат:
            segments: путь к файлу описания сегментов
        """
        feats_rspec = "ark:compute-mfcc-feats --verbose=0 --config=" + self.conf + " scp:" + self.scp + " ark:- |"
        segments = str(self.output / 'segments')
        with SequentialMatrixReader(feats_rspec) as f, open(segments, 'w') as s:
            for key, feats in f:
                out = self.sad.segment(feats)
                segs, _ = self.seg.process(out['alignment'])
                self.seg.write(key, segs, s)
                logging.info("Сегментирован файл '" + key + "'")
        return segments

    def extract_segments(self, segments):
        """
        Извлечение сегментов
        
        Аргументы:
            segments: путь к файлу описания сегментов

        Результат:
            wav_segments: путь к .SCP файлу с аудио сегментов
            utt2spk: путь к файлу сопоставления сегментов и говорящих
            spk2utt: путь к файлу перечисления сегментов для каждого говорящего
        """
        wav_segments = str(self.output / 'wav_segments.scp')
        utt2spk = str(self.output / 'utt2spk')
        with open(segments, 'r') as s, \
            open(wav_segments, 'w') as ws, \
            open(utt2spk, 'w') as u:
            for segment in s:
                segment_info = segment.split(' ')
                segment_id = segment_info[0]
                speaker_id = segment.split(' ')[1].split('.')[-1] or segment_id
                ws.write(segment_id + '\t' + str(self.output / '@')[:-1] + segment_id + '.wav' + '\n')
                u.write(segment_id + '\tКанал ' + speaker_id + '\n')
        spk2utt = make_spk2utt(utt2spk)
        extract_command = "extract-segments scp:" + self.scp + " " + str(self.output / 'segments') + " scp:" + str(self.output / 'wav_segments.scp')
        with subprocess.Popen(extract_command, shell=True):
            pass
        return wav_segments, utt2spk, spk2utt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Утилита для сегментации аудио')
    parser.add_argument('-s', '--scp', metavar='SCP', help='Путь к .SCP файлу с аудио')
    parser.add_argument('-m', '--model', metavar='RAW', help='Путь к .RAW файлу модели сегментации')
    parser.add_argument('-p', '--post', metavar='VEC', help='Путь к .VEC файлу апостериорных вероятностей сегментации')
    parser.add_argument('-c', '--conf', metavar='CONF', help='Путь к .CONF конфигурационному файлу сегментации')
    parser.add_argument('-o', '--output', metavar='OUT', help='Путь к директории с результатами сегментации')
    parser.add_argument('-l', '--log', dest='log', action='store_true', help='Логировать результат сегментации')

    args = parser.parse_args()

    try:
        segmenter = Segmenter(args.scp, args.model, args.post, args.conf, args.output, args.log)
        segments = segmenter.segment()
    except:
        logging.error("Не удалось выполнить сегментацию аудио")
    
    try:
        segmenter.extract_segments(segments)
    except:
        logging.error("Не удалось выполнить извлечение сегментов")