#!/usr/bin/python
import os
import glob
import argparse
import csv
import time
import tqdm
import pickle
import logging
import numpy as np
import pandas as pd
import pysubs2
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tools.utils import create_logger

class TranscriptionsParser(object):
    """Класс для парсинга файлов транскрибации"""

    def __init__(self, ass, output, log, processes, batch_size, csv=None, pickle=False):
        """
        Инициализация парсера
        
        Аргументы:
            ass: путь к директории с .ASS файлами
            output: путь к директории с результатами парсинга
            log: путь к файлу логов
            processes: количество процессов для обработки файлов
            batch_size: размер пакета для обработки файлов
            csv: путь к .CSV файлу парсинга
            pickle: признак сериализации
        """  
        self.ass = ass
        self.output = output
        self.csv = csv if csv else str(self.output / 'transcriptions.csv')
        self.log = log
        self.processes = processes
        self.batch_size = batch_size
        self.pickle = pickle        

    def get_info(self, transcription):
        """
        Извлечение информации о .ASS файле
        
        Аргументы:
            transcription: загруженный .ASS файл
            
        Результат:
            info: информация о .ASS файле
        """    
        info = {'Title': transcription.info['Title'],
                'ScriptType': transcription.info['ScriptType'],
                'WrapStyle': transcription.info['WrapStyle'],
                'ScaledBorderAndShadow': transcription.info['ScaledBorderAndShadow'],
                'YCbCr Matrix': transcription.info['YCbCr Matrix']}
        return info

    def get_style_attributes(self, style):
        """
        Извлечение атрибутов стиля
        
        Аргументы:
            event: строка стиля
            
        Результат:
            attributes: словарь атрибутов стиля
        """
        attributes = {'Fontname': style.fontname,
                    'Fontsize': style.fontsize,
                    'PrimaryColour': style.primarycolor,
                    'SecondaryColour': style.secondarycolor,
                    'OutlineColour': style.outlinecolor,
                    'BackColour': style.backcolor,
                    'Bold': style.bold,
                    'Italic': style.italic,
                    'Underline': style.underline,
                    'StrikeOut': style.strikeout,
                    'ScaleX': style.scalex,
                    'ScaleY': style.scaley,
                    'Spacing': style.spacing,
                    'Angle': style.angle,
                    'BorderStyle': style.borderstyle,
                    'Outline': style.outline,
                    'Shadow': style.shadow,
                    'Alignment': style.alignment,
                    'MarginL': style.marginl,
                    'MarginR': style.marginr,
                    'MarginV': style.marginv,
                    'Encoding': style.encoding}
        return attributes

    def get_event_attributes(self, event):
        """
        Извлечение атрибутов события
        
        Аргументы:
            event: строка события
            
        Результат:
            attributes: словарь атрибутов события
        """
        attributes = {'Layer': event.layer,
                    'Start': event.start,
                    'End': event.end,
                    'Style': event.style,
                    'Name': event.name,
                    'MarginL': event.marginl,
                    'MarginR': event.marginr,
                    'MarginV': event.marginv,
                    'Effect': event.effect,
                    'Text': event.text}
        return attributes

    def process_batch_files(self, batch):
        """
        Обработка пакета .ASS файлов и запись результата в файл .CSV
        
        Аргументы:
            batch: пакет файлов            
        """
        transcriptions = pd.DataFrame(columns=['Audio File', 'Start', 'End', 'Name', 'Text'])
        if self.log:
            logger = create_logger('logger_' + str(os.getpid()), 'file', logging.DEBUG, self.log)
        else:
            logger = create_logger('logger','stream', logging.DEBUG)        
        for file in batch:
            try:
                transcription = pysubs2.load(file)
                if not transcription.events:
                    logger.debug("В файле '{}' отсутствуют события".format(file))
                else:
                    for event in transcription.events:
                        attributes = self.get_event_attributes(event)
                        attributes['Audio File'] = transcription.aegisub_project['Audio File']
                        transcriptions = transcriptions.append(pd.DataFrame(attributes, index=[0])[transcriptions.columns], ignore_index=True)
            except:
                logger.error("Не удалось обработать файл '{}'".format(file))
        with open(self.csv, 'a') as f:
            transcriptions.to_csv(f, header=False, index=False, encoding='cp1251')

    def process_file(self, file):
        """
        Обработка одного .ASS файла и возврат результата в виде DataFrame
        
        Аргументы:
            file: .ASS файл
        """        
        transcriptions = pd.DataFrame(columns=['Audio File', 'Start', 'End', 'Name', 'Text'])
        transcription = pysubs2.load(file)
        if transcription.events:
            for event in transcription.events:
                attributes = self.get_event_attributes(event)
                attributes['Audio File'] = transcription.aegisub_project['Audio File']
                transcriptions = transcriptions.append(pd.DataFrame(attributes, index=[0])[transcriptions.columns], ignore_index=True)
        return transcriptions

    
def split_files_by_batch(files, batch_size):
    """
    Разделение списка файлов на пакеты
    
    Аргументы:
        files: список файлов
        batch_size: размер пакета
    """    
    for i in range(0, len(files), batch_size):
        yield files[i: i + batch_size]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Парсер транскрибированных файлов')
    parser.add_argument('ass', metavar='ASS', help='Путь к директории с .ASS файлами')
    parser.add_argument('output', metavar='OUT', help='Путь к директории с результатами парсинга')
    parser.add_argument('-l', '--log', help='Путь к директории с логами')
    parser.add_argument('-p', '--processes', default=None, type=int, help='Количество процессов для обработки файлов')
    parser.add_argument('-b', '--batch_size', default=10, type=int, help='Размер пакета для обработки файлов')
    parser.add_argument('-c', '--csv', help='Путь к .CSV файлу парсинга')
    parser.add_argument('-s', '--pickle', dest='pickle', action='store_true', help='Сериализовать результат парсинга')

    args = parser.parse_args()

    ASS_DIR = Path(args.ass)
    OUTPUT_DIR = Path(args.output)
    LOG_DIR = Path(args.log) if args.log else ''
    PROCESSES = args.processes or cpu_count()
    BATCH_SIZE = args.batch_size
    CSV = args.csv
    IS_PICKLE = args.pickle

    try:
        transcriptions_csv = CSV if CSV else str(OUTPUT_DIR / 'transcriptions.csv')
        with open(transcriptions_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Audio File', 'Start', 'End', 'Name', 'Text'])
    except:
        raise Exception("Не удалось создать результирующий .CSV-файл")
        
    if LOG_DIR:
        try:
            log_name = str(LOG_DIR / str(time.strftime('%Y%m%d-%H%M%S') + '.log'))
            logger = create_logger('logger', 'file', logging.DEBUG, log_name)
        except:
            raise Exception("Не удалось создать лог-файл")
    else:
        logger = create_logger('logger', 'stream', logging.INFO)
    
    pool = Pool(PROCESSES)
    logger.info("Запуск парсинга файлов")
    logger.debug("Количество процессов: {}".format(PROCESSES))
    logger.debug("Размер пакета: {}".format(BATCH_SIZE))
    
    transcr_parser = TranscriptionsParser(ASS_DIR, OUTPUT_DIR, log_name, PROCESSES, BATCH_SIZE, CSV, IS_PICKLE)
    tq = tqdm.tqdm
    files = glob.glob(str(ASS_DIR / '*.ass'))
    batches = list(split_files_by_batch(files, BATCH_SIZE))
    for _ in tq(pool.imap(transcr_parser.process_batch_files, batches), total=len(batches)):
        pass
    pool.close()
    pool.join()
    
    if IS_PICKLE:
        try:
            logger.info("Запуск сериализации результата парсинга")
            transcriptions_pkl = str(Path(OUTPUT_DIR) / 'transcriptions.pkl')
            with open(transcriptions_pkl, 'wb') as f:
                transcriptions = pd.read_csv(transcriptions_csv, encoding='cp1251', low_memory=True)
                pickle.dump(transcriptions, f)
            logger.info("Завершение сериализации результата парсинга")
        except:
            logger.error("Не удалось выполнить сериализацию результата парсинга")
    logger.info("Завершение парсинга файлов")