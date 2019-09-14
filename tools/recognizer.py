#!/usr/bin/python
import argparse
from pathlib import Path
import logging
from kaldi.asr import NnetLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.util.table import SequentialMatrixReader, CompactLatticeWriter

class Recognizer(object):
    """Класс для распознавания речи с помощью алгоритма nnet3"""

    def __init__(self, scp, model, graph, words, conf, iconf, spk2utt, output, printed=False, log=False):
        """
        Инициализация транскриптора
        
        Аргументы:
            scp: путь к .SCP файлу с аудио
            model: путь к .MDL файлу модели распознавания
            graph: путь к .FST файлу общего графа распознавания
            words: путь к .TXT файлу текстового корпуса
            conf: путь к .CONF конфигурационному файлу распознавания
            iconf: путь к .CONF конфигурационному файлу векторного экстрактора
            spk2utt: путь к файлу перечисления сегментов для каждого говорящего
            output: путь к директории с результатами распознавания
            printed: признак печати результатов распознавания
            log: признак логирования
        """  
        self.scp = scp
        self.model = model
        self.graph = graph
        self.words = words
        self.conf = conf
        self.iconf = iconf
        self.spk2utt = spk2utt
        self.output = Path(output)
        self.printed = printed
        self.log = log

        decoder_opts = LatticeFasterDecoderOptions()
        decoder_opts.beam = 13
        decoder_opts.max_active = 7000
        decodable_opts = NnetSimpleComputationOptions()
        decodable_opts.acoustic_scale = 1.0
        decodable_opts.frame_subsampling_factor = 3
        self.asr = NnetLatticeFasterRecognizer.from_files(self.model, self.graph, self.words,
                decoder_opts=decoder_opts, decodable_opts=decodable_opts)
    
    def recognize(self, wav=None):
        """
        Распознавание речи       
        
        Аргументы:
            wav: наименование аудио файла

        Результат:
            transcriptions: путь к файлу транскрибации
        """
        transcriptions = str(self.output / wav) if wav else 'transcriptions'
        feats_rspec = ("ark:compute-mfcc-feats --config=" + self.conf + " scp:" + self.scp + " ark:- |")
        ivectors_rspec = (feats_rspec + "ivector-extract-online2 "
                        "--config=" + self.iconf + " "
                        "ark:" + self.spk2utt + " ark:- ark:- |")
        lat_wspec = "ark:| gzip -c > lat.gz"   
        with SequentialMatrixReader(feats_rspec) as feats_reader, \
            SequentialMatrixReader(ivectors_rspec) as ivectors_reader, \
            CompactLatticeWriter(lat_wspec) as lat_writer:
            for (fkey, feats), (ikey, ivectors) in zip(feats_reader, ivectors_reader):
                assert(fkey == ikey)
                out = self.asr.decode((feats, ivectors))
                lat_writer[fkey] = out['lattice']
                if self.printed:
                    print(fkey, out['text'], flush=True)
                with open(transcriptions, 'a') as f:
                    f.write(fkey + '\t' + out['text'].lower() + '\n')
        return transcriptions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Утилита для распознавания речи')
    parser.add_argument('-s', '--scp', metavar='SCP', help='Путь к .SCP файлу с аудио')
    parser.add_argument('-m', '--model', metavar='MDL', help='Путь к .MDL файлу модели распознавания')
    parser.add_argument('-g', '--graph', metavar='FST', help='Путь к .FST файлу общего графа распознавания')
    parser.add_argument('-w', '--words', metavar='TXT', help='Путь к .TXT файлу текстового корпуса')
    parser.add_argument('-c', '--conf', metavar='CONF', help='Путь к .CONF конфигурационному файлу распознавания')
    parser.add_argument('-i', '--iconf', metavar='CONF', help='Путь к .CONF конфигурационному файлу векторного экстрактора')
    parser.add_argument('-u', '--spk2utt', help='Путь к файлу перечисления сегментов для каждого говорящего')
    parser.add_argument('-o', '--output', metavar='OUT', help='Путь к директории с результатами распознавания')
    parser.add_argument('-p', '--printed', dest='printed', action='store_true', help='Печатать результат распознавания')
    parser.add_argument('-l', '--log', dest='log', action='store_true', help='Логировать результат распознавания')

    args = parser.parse_args()

    recognizer = Recognizer(args.scp, args.model, args.graph, args.words, args.conf, args.iconf, args.spk2utt, args.output, args.printed, args.log)
    recognizer.recognize()