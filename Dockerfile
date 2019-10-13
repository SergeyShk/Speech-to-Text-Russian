# Dockerfile для сборки образа проекта распознавания речи
FROM pykaldi/pykaldi

# Настройка окружения
ENV PATH /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/kaldi/src/featbin:/kaldi/src/ivectorbin:/kaldi/src/online2bin:/kaldi/src/rnnlmbin:/kaldi/src/fstbin:$PATH
ENV LC_ALL C.UTF-8
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Установка необходимых python-библиотек
RUN pip install --upgrade pip \
	tqdm \
	pandas \
	matplotlib \
	seaborn \
	librosa \
	sox \
	pysubs2 \
	flask \
	soundfile

# Копирование файлов проекта
RUN mkdir speech_recognition	
WORKDIR speech_recognition
RUN echo "cat motd" >> /root/.bashrc
COPY . ./