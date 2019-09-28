#!/usr/bin/python
import os
import base64
import sys
import sox
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from librosa import display
from time import time, gmtime, strftime
from pathlib import Path
from flask import Flask, render_template, request, redirect, flash

sys.path.append('..')
from tools import data_preparator, segmenter, recognizer, transcriptions_parser
from tools.utils import make_ass, make_wav_scp, delete_folder

app = Flask(__name__)
app.config['SECRET_KEY'] = '8dgn89vdf8vff8v9df99f'
app.config['ALLOWED_EXTENSIONS'] = ['wav']
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = Path('data')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def recognize(temp, wav):
    wav_scp = str(Path(temp) / 'wav.scp')
    make_wav_scp(wav, wav_scp)
    segm = segmenter.Segmenter(wav_scp, '../model/final.raw', '../model/conf/post_output.vec', '../model/conf/mfcc_hires.conf', temp)
    segments = segm.segment()
    wav_segments_scp, utt2spk, spk2utt = segm.extract_segments(segments)
    rec = recognizer.Recognizer(wav_segments_scp, '../model/final.mdl', '../model/HCLG.fst', '../model/words.txt', 
                                '../model/conf/mfcc.conf', '../model/conf/ivector_extractor.conf', spk2utt, temp)
    transcriptions = rec.recognize(Path(wav).stem)
    ass = str(Path(temp) / 'wav.ass')
    make_ass(Path(wav).name, segments, transcriptions, utt2spk, ass)
    pars = transcriptions_parser.TranscriptionsParser('', '', '', 0, 0, 'wav.csv')
    transcriptions_df = pars.process_file(ass)
    return transcriptions_df

def plot_waveform(temp, wav, channels):
    y, sr = librosa.load(wav, mono=False)
    if channels == 1:
        y = y.reshape(-1, len(y))
    DPI = 72
    plt.figure(1, figsize=(16, 9), dpi=DPI) 
    plt.subplots_adjust(wspace=0, hspace=0) 
    for n in range(channels): 
        plt.subplot(2, 1, n + 1, facecolor='200')
        display.waveplot(y[n], sr)
        plt.grid(True, color='w') 
    waveform = str(Path(temp) / 'waveform.png')
    plt.savefig(waveform, dpi=DPI)
    waveform = str(base64.b64encode(open(waveform, 'rb').read()))[2: -1]
    return waveform

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Отсутствует файл')
            return redirect('/')
        file = request.files['file']
        if file.filename == '':
            flash('Не выбран файл для загрузки')
            return redirect('/')
        if not allowed_file(file.filename):
            flash('Файл должен иметь расширение .WAV')
            return redirect('/')
        filename = file.filename.replace(' ', '_')
        wav = str(app.config['UPLOAD_FOLDER'] / filename)
        file.save(wav)
        wav_info = sox.file_info.info(wav)
        info = {}
        info['Длительность аудио'] = str(round(wav_info['duration'], 2)) + ' с'
        info['Число каналов'] = wav_info['channels']
        info['Частота дискретизации'] = str(int(wav_info['sample_rate'])) + ' Гц'
        start_time = time()
        temp = str(app.config['UPLOAD_FOLDER'] / Path(wav).stem)
        os.makedirs(temp, exist_ok=True)
        transcriptions = recognize(temp, wav)
        waveform = plot_waveform(temp, wav, wav_info['channels']) if request.form.get('plotWaveform') else None
        delete_folder(temp)
        os.remove(wav)
        info['Время выполнения'] = str(round(time() - start_time, 2)) + ' с'
        transcriptions = transcriptions[['Name', 'Start', 'End', 'Text']]
        transcriptions.columns = ['Канал', 'Начало', 'Конец', 'Текст']
        with pd.option_context('display.max_colwidth', -1):
            transcriptions_html = transcriptions.to_html(index=False, justify='center', escape=False)        
        return render_template('results.html', filename='.'.join(filename.split('.')[:-1]), 
                                info=info, waveform=waveform, transcriptions=transcriptions_html)
    return render_template('index.html')

@app.errorhandler(413)
def request_entity_too_large(e):
        flash('Размер файла не должен превышать 20 МБ')
        return redirect('/')

if __name__ == '__main__':
    app.run(host='0.0.0.0')