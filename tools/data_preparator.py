#!/usr/bin/python
import argparse
import os
import glob
from pathlib import Path
import logging
import wave

class DataPreparator(object):
    """Класс для подготовки данных для распознавания речи"""

    def __init__(self, wav, output, log=False):
        """
        Инициализация препаратора данных для распознавания речи
        
        Аргументы:
            wav: путь к .WAV файлам аудио
            output: путь к подготовленным файлам
            log: признак логирования
        """  
        self.wav = Path(wav)
        self.output = Path(output)
        self.log = log

    def create_directories(self):
        """
        Создание необходимых директорий
        
        Результат:
            log_dir: путь к директории с файлами логов
            temp_dir: путь к директории с временными файлами
            ass_dir: путь к директории с файлами транскрибаций
            error_dir: путь к директории с .WAV файлами, которые не удалось распознать
        """
        log_dir = self.output / 'logs'
        temp_dir = self.output / 'temp'
        ass_dir = self.output / 'ass'
        error_dir = self.output / 'error'
        os.makedirs(str(log_dir), exist_ok=True)
        os.makedirs(str(temp_dir), exist_ok=True)
        os.makedirs(str(ass_dir), exist_ok=True)
        os.makedirs(str(error_dir), exist_ok=True)
        return log_dir, temp_dir, ass_dir, error_dir

    def rename_wav(self, wav_files=None):
        """
        Переименование файлов под формат Kaldi

        Аргументы:
            wav_files: список .WAV файлов

        Результат:
            wav_files: список переименованных .WAV файлов
        """
        if not wav_files:
            wav_files = glob.glob(str(self.wav / '*.wav'))
        for wav_file in wav_files:
            os.rename(wav_file, wav_file.replace(' ', '_'))
        return [wav_file.replace(' ', '_') for wav_file in wav_files]

    def make_wav_scp(self):
        """
        Формирование .SCP файла для аудио
        
        Результат:
            wav_scp: путь к .SCP файлу с аудио
        """
        wav_files = glob.glob(str(self.wav / '*.wav'))
        wav_scp = str(self.output / 'wav.scp')
        with open(wav_scp, 'w') as f:
            for wav_file in wav_files:
                wav = wave.open(wav_file, 'r')
                if wav.getnchannels() == 1:
                    f.write(str(Path(wav_file).stem) + '.0\t' + 'sox ' + wav_file + ' -t wav - remix 1 |\n')
                else:
                    f.write(str(Path(wav_file).stem) + '.0\t' + 'sox ' + wav_file + ' -t wav - remix 1 |\n')
                    f.write(str(Path(wav_file).stem) + '.1\t' + 'sox ' + wav_file + ' -t wav - remix 2 |\n')
        return wav_scp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Утилита для подготовки данных')
    parser.add_argument('-w', '--wav', metavar='WAV', help='Путь к .WAV файлам аудио')
    parser.add_argument('-o', '--output', metavar='OUT', help='Путь к подготовленным файлам')
    parser.add_argument('-l', '--log', dest='log', action='store_true', help='Логировать результат сегментации')

    args = parser.parse_args()

    data_preparator = DataPreparator(args.wav, args.output, args.log)
    data_preparator.create_directories()
    data_preparator.rename_wav()
    data_preparator.make_wav_scp()