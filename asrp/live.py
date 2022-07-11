import numpy as np
import threading
import time
from queue import Queue

from asrp.interface import HFSpeechInference


def live_vad_process(device_name, asr_input_queue, vad_mode=1):
    import webrtcvad
    import pyaudio
    vad = webrtcvad.Vad()
    vad.set_mode(vad_mode)

    audio = pyaudio.PyAudio()
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    # A frame must be either 10, 20, or 30 ms in duration for webrtcvad
    FRAME_DURATION = 30
    CHUNK = int(RATE * FRAME_DURATION / 1000)

    microphones = live_list_microphones(audio)
    selected_input_device_id = live_get_input_device_id(
        device_name, microphones)

    stream = audio.open(input_device_index=selected_input_device_id,
                        format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    frames = b''
    while True:
        if LiveHFSpeech.exit_event.is_set():
            break
        frame = stream.read(CHUNK, exception_on_overflow=False)
        is_speech = vad.is_speech(frame, RATE)
        if is_speech:
            frames += frame
        else:
            if len(frames) > 1:
                asr_input_queue.put(frames)
            frames = b''
    stream.stop_stream()
    stream.close()
    audio.terminate()


def live_asr_process(model_name, in_queue, output_queue):
    wave2vec_asr = HFSpeechInference(model_name)

    print("\nlistening to your voice\n")
    while True:
        audio_frames = in_queue.get()
        if audio_frames == "close":
            break

        float64_buffer = np.frombuffer(
            audio_frames, dtype=np.int16) / 32767
        start = time.perf_counter()
        text = wave2vec_asr.buffer_to_text(float64_buffer).lower()
        inference_time = time.perf_counter() - start
        sample_length = len(float64_buffer) / 16000  # length in sec
        if text != "":
            output_queue.put([text, sample_length, inference_time])


def live_get_input_device_id(device_name, microphones):
    for device in microphones:
        if device_name in device[1]:
            return device[0]


def live_list_microphones(pyaudio_instance):
    info = pyaudio_instance.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    result = []
    for i in range(0, numdevices):
        if (pyaudio_instance.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            name = pyaudio_instance.get_device_info_by_host_api_device_index(
                0, i).get('name')
            result += [[i, name]]
    return result


class LiveHFSpeech:
    exit_event = threading.Event()

    def __init__(self, model_name, device_name="default"):
        self.model_name = model_name
        self.device_name = device_name
        self.asr_output_queue = Queue()
        self.asr_input_queue = Queue()
        self.live_asr_process = threading.Thread(target=live_asr_process, args=(
            self.model_name, self.asr_input_queue, self.asr_output_queue,))
        self.live_vad_process = threading.Thread(target=live_vad_process, args=(
            self.device_name, self.asr_input_queue,))

    def stop(self):
        """stop the asr process"""
        LiveHFSpeech.exit_event.set()
        self.asr_input_queue.put("close")
        print("asr stopped")

    def start(self):
        """start the asr process"""
        self.live_asr_process.start()
        time.sleep(5)  # start vad after asr model is loaded
        self.live_vad_process.start()

    def get_last_text(self):
        """returns the text, sample length and inference time in seconds."""
        return self.asr_output_queue.get()
