import re
import os
import sys
import time
import onnx
import json
import faiss
import torch
import codecs
import shutil
import librosa
import logging
import warnings
import requests
import parselmouth
import onnxruntime
import logging.handlers

import numpy as np
import soundfile as sf
import torch.nn.functional as F

from tqdm import tqdm
from scipy import signal
from fairseq import checkpoint_utils
from pydub import AudioSegment, silence

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())

from lib.FCPE import FCPE
from lib.RMVPE import RMVPE
from lib.WORLD import PYWORLD
from lib.CREPE import predict, mean, median

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

for l in ["torch", "faiss", "httpx", "fairseq", "httpcore", "faiss.loader", "numba.core", "urllib3"]:
    logging.getLogger(l).setLevel(logging.ERROR)

def HF_download_file(url, output_path=None):
    url = url.replace("/blob/", "/resolve/").replace("?download=true", "").strip()

    if output_path is None: output_path = os.path.basename(url)
    else: output_path = os.path.join(output_path, os.path.basename(url)) if os.path.isdir(output_path) else output_path

    response = requests.get(url, stream=True, timeout=300)

    if response.status_code == 200:        
        progress_bar = tqdm(total=int(response.headers.get("content-length", 0)), ncols=100, unit="byte")

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                progress_bar.update(len(chunk))
                f.write(chunk)

        progress_bar.close()
        return output_path
    else: raise ValueError(response.status_code)

def check_predictors(method):
    def download(predictors):
        if not os.path.exists(os.path.join(predictors)): HF_download_file(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cerqvpgbef/", "rot13") + predictors, predictors)

    model_dict = {**dict.fromkeys(["rmvpe", "rmvpe-legacy"], "rmvpe.pt"), **dict.fromkeys(["rmvpe-onnx", "rmvpe-legacy-onnx"], "rmvpe.onnx"), **dict.fromkeys(["fcpe", "fcpe-legacy"], "fcpe.pt"), **dict.fromkeys(["fcpe-onnx", "fcpe-legacy-onnx"], "fcpe.onnx"), **dict.fromkeys(["crepe-full", "mangio-crepe-full"], "crepe_full.pth"), **dict.fromkeys(["crepe-full-onnx", "mangio-crepe-full-onnx"], "crepe_full.onnx"), **dict.fromkeys(["crepe-large", "mangio-crepe-large"], "crepe_large.pth"), **dict.fromkeys(["crepe-large-onnx", "mangio-crepe-large-onnx"], "crepe_large.onnx"), **dict.fromkeys(["crepe-medium", "mangio-crepe-medium"], "crepe_medium.pth"), **dict.fromkeys(["crepe-medium-onnx", "mangio-crepe-medium-onnx"], "crepe_medium.onnx"), **dict.fromkeys(["crepe-small", "mangio-crepe-small"], "crepe_small.pth"), **dict.fromkeys(["crepe-small-onnx", "mangio-crepe-small-onnx"], "crepe_small.onnx"), **dict.fromkeys(["crepe-tiny", "mangio-crepe-tiny"], "crepe_tiny.pth"), **dict.fromkeys(["crepe-tiny-onnx", "mangio-crepe-tiny-onnx"], "crepe_tiny.onnx"), **dict.fromkeys(["harvest", "dio"], "world.pth")}

    if "hybrid" in method:
        methods_str = re.search("hybrid\[(.+)\]", method)
        if methods_str: methods = [method.strip() for method in methods_str.group(1).split("+")]
        for method in methods:
            if method in model_dict: download(model_dict[method])
    elif method in model_dict: download(model_dict[method])

def check_embedders(hubert):
    if hubert in ["contentvec_base", "hubert_base", "japanese_hubert_base", "korean_hubert_base", "chinese_hubert_base", "Hidden_Rabbit_last", "portuguese_hubert_base"]:
        model_path = hubert + '.pt'
        if not os.path.exists(model_path): HF_download_file(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/rzorqqref/", "rot13") + f"{hubert}.pt", model_path)

def load_audio(file):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.isfile(file): raise FileNotFoundError(f"Không tìm thấy: {file}")

        audio, sr = sf.read(file)

        if len(audio.shape) > 1: audio = librosa.to_mono(audio.T)
        if sr != 16000: audio = librosa.resample(audio, orig_sr=sr, target_sr=16000, res_type="soxr_vhq")
    except Exception as e:
        raise RuntimeError(f"Xảy ra lỗi khi tải tệp âm thanh: {e}")
    return audio.flatten()

def process_audio(file_path, output_path):
    try:
        song = pydub_convert(AudioSegment.from_file(file_path))
        cut_files, time_stamps = [], []

        for i, (start_i, end_i) in enumerate(silence.detect_nonsilent(song, min_silence_len=750, silence_thresh=-70)):
            chunk = song[start_i:end_i]

            if len(chunk) > 10:
                chunk_file_path = os.path.join(output_path, f"chunk{i}.wav")
                if os.path.exists(chunk_file_path): os.remove(chunk_file_path)

                chunk.export(chunk_file_path, format="wav")

                cut_files.append(chunk_file_path)
                time_stamps.append((start_i, end_i))
            else: print(f"Phần {i} được bỏ qua do độ dài quá ngắn {len(chunk)}ms")

        print(f"Tổng số phần cắt: {len(cut_files)}")
        return cut_files, time_stamps
    except Exception as e:
        raise RuntimeError(f"Đã xảy ra lỗi khi cắt tệp âm thanh: {e}")

def merge_audio(files_list, time_stamps, original_file_path, output_path, format):
    try:
        def extract_number(filename):
            match = re.search(r'_(\d+)', filename)
            return int(match.group(1)) if match else 0

        total_duration = len(AudioSegment.from_file(original_file_path))
        combined = AudioSegment.empty() 
        current_position = 0 

        for file, (start_i, end_i) in zip(sorted(files_list, key=extract_number), time_stamps):
            if start_i > current_position: combined += AudioSegment.silent(duration=start_i - current_position)  
            combined += AudioSegment.from_file(file)  
            current_position = end_i

        if current_position < total_duration: combined += AudioSegment.silent(duration=total_duration - current_position)
        combined.export(output_path, format=format)
        return output_path
    except Exception as e:
        raise RuntimeError(f"Đã xảy ra lỗi khi ghép các tệp âm thanh lại: {e}")

def pydub_convert(audio):
    samples = np.frombuffer(audio.raw_data, dtype=np.int16)
    if samples.dtype != np.int16: samples = (samples * 32767).astype(np.int16)
    return AudioSegment(samples.tobytes(), frame_rate=audio.frame_rate, sample_width=samples.dtype.itemsize, channels=audio.channels)

def run_batch_convert(params):
    path, audio_temp, export_format, cut_files, pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0_method, pth_path, index_path, f0_autotune, f0_autotune_strength, clean_audio, clean_strength, embedder_model, resample_sr = params["path"], params["audio_temp"], params["export_format"], params["cut_files"], params["pitch"], params["filter_radius"], params["index_rate"], params["volume_envelope"], params["protect"], params["hop_length"], params["f0_method"], params["pth_path"], params["index_path"], params["f0_autotune"], params["f0_autotune_strength"], params["clean_audio"], params["clean_strength"], params["embedder_model"], params["resample_sr"]

    segment_output_path = os.path.join(audio_temp, f"output_{cut_files.index(path)}.{export_format}")
    if os.path.exists(segment_output_path): os.remove(segment_output_path)
    
    VoiceConverter().convert_audio(pitch=pitch, filter_radius=filter_radius, index_rate=index_rate, volume_envelope=volume_envelope, protect=protect, hop_length=hop_length, f0_method=f0_method, audio_input_path=path, audio_output_path=segment_output_path, model_path=pth_path, index_path=index_path, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, clean_audio=clean_audio, clean_strength=clean_strength, export_format=export_format, embedder_model=embedder_model, resample_sr=resample_sr)
    os.remove(path)

    if os.path.exists(segment_output_path): return segment_output_path
    else: raise FileNotFoundError(f"Không tìm thấy: {segment_output_path}")

def run_convert_script(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0_method, input_path, output_path, pth_path, index_path, f0_autotune, f0_autotune_strength, clean_audio, clean_strength, export_format, embedder_model, resample_sr, split_audio):
    check_predictors(f0_method)
    check_embedders(embedder_model)

    cvt = VoiceConverter()
    start_time = time.time()

    if not pth_path or not os.path.exists(pth_path) or os.path.isdir(pth_path) or not pth_path.endswith((".pth", ".onnx")): raise FileNotFoundError("Tệp mô hình không hợp lệ!")

    processed_segments = []
    audio_temp = os.path.join("audios_temp")
    if not os.path.exists(audio_temp) and split_audio: os.makedirs(audio_temp, exist_ok=True)

    if os.path.isdir(input_path):
        try:
            print(f"Đầu vào là thư mục, sẽ chuyển đổi tất cả tệp âm thanh bên trong thư mục")
            audio_files = [f for f in os.listdir(input_path) if f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"))]

            if not audio_files: raise FileNotFoundError("Không tìm thấy bất kì tệp âm thanh nào!")
            print(f"Tìm thấy {len(audio_files)} tệp âm thanh")

            for audio in audio_files:
                audio_path = os.path.join(input_path, audio)
                output_audio = os.path.join(input_path, os.path.splitext(audio)[0] + f"_output.{export_format}")

                if split_audio:
                    try:
                        cut_files, time_stamps = process_audio(audio_path, audio_temp)
                        params_list = [{"path": path, "audio_temp": audio_temp, "export_format": export_format, "cut_files": cut_files, "pitch": pitch, "filter_radius": filter_radius, "index_rate": index_rate, "volume_envelope": volume_envelope, "protect": protect, "hop_length": hop_length, "f0_method": f0_method, "pth_path": pth_path, "index_path": index_path, "f0_autotune": f0_autotune, "f0_autotune_strength": f0_autotune_strength, "clean_audio": clean_audio, "clean_strength": clean_strength, "embedder_model": embedder_model, "resample_sr": resample_sr} for path in cut_files]
                        
                        with tqdm(total=len(params_list), desc="Chuyển đổi âm thanh", ncols=100, unit="a") as pbar:
                            for params in params_list:
                                results = run_batch_convert(params)
                                processed_segments.append(results)
                                pbar.update(1)

                        merge_audio(processed_segments, time_stamps, audio_path, output_audio, export_format)
                    except Exception as e:
                        raise RuntimeError(f"Đã xảy ra lỗi khi chuyển đổi hàng loạt tệp: {e}")
                    finally:
                        if os.path.exists(audio_temp): shutil.rmtree(audio_temp, ignore_errors=True)
                else:
                    try:
                        print(f"Bắt đầu chuyển đổi tệp '{audio_path}'...")
                        if os.path.exists(output_audio): os.remove(output_audio)

                        with tqdm(total=1, desc="Chuyển đổi âm thanh", ncols=100, unit="a") as pbar:
                            cvt.convert_audio(pitch=pitch, filter_radius=filter_radius, index_rate=index_rate, volume_envelope=volume_envelope, protect=protect, hop_length=hop_length, f0_method=f0_method, audio_input_path=audio_path, audio_output_path=output_audio, model_path=pth_path, index_path=index_path, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, clean_audio=clean_audio, clean_strength=clean_strength, export_format=export_format, embedder_model=embedder_model, resample_sr=resample_sr)
                            pbar.update(1)
                    except Exception as e:
                        raise RuntimeError(f"Đã xảy ra lỗi khi chuyển đổi tệp: {e}")

            elapsed_time = time.time() - start_time
            print(f"Đã chuyển đổi hàng loạt hoàn tất sau {elapsed_time:.2f} giây. {output_path.replace('wav', export_format)}")
        except Exception as e:
            raise RuntimeError(f"Đã xảy ra lỗi khi chuyển đổi hàng loạt: {e}")
    else:
        print(f"Bắt đầu chuyển đổi tệp '{input_path}'...")
        if not os.path.exists(input_path): raise FileExistsError("Không tìm thấy đầu vào")
        if os.path.exists(output_path): os.remove(output_path)

        if split_audio:
            try:              
                cut_files, time_stamps = process_audio(input_path, audio_temp)
                params_list = [{"path": path, "audio_temp": audio_temp, "export_format": export_format, "cut_files": cut_files, "pitch": pitch, "filter_radius": filter_radius, "index_rate": index_rate, "volume_envelope": volume_envelope, "protect": protect, "hop_length": hop_length, "f0_method": f0_method, "pth_path": pth_path, "index_path": index_path, "f0_autotune": f0_autotune, "f0_autotune_strength": f0_autotune_strength, "clean_audio": clean_audio, "clean_strength": clean_strength, "embedder_model": embedder_model, "resample_sr": resample_sr} for path in cut_files]
                
                with tqdm(total=len(params_list), desc="Chuyển đổi âm thanh", ncols=100, unit="a") as pbar:
                    for params in params_list:
                        results = run_batch_convert(params)
                        processed_segments.append(results)
                        pbar.update(1)

                merge_audio(processed_segments, time_stamps, input_path, output_path.replace("wav", export_format), export_format)
            except Exception as e:
                raise RuntimeError(f"Đã xảy ra lỗi khi chuyển đổi hàng loạt tệp: {e}")
            finally:
                if os.path.exists(audio_temp): shutil.rmtree(audio_temp, ignore_errors=True)
        else:
            try:
                with tqdm(total=1, desc="Chuyển đổi âm thanh", ncols=100, unit="a") as pbar:
                    cvt.convert_audio(pitch=pitch, filter_radius=filter_radius, index_rate=index_rate, volume_envelope=volume_envelope, protect=protect, hop_length=hop_length, f0_method=f0_method, audio_input_path=input_path, audio_output_path=output_path, model_path=pth_path, index_path=index_path, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, clean_audio=clean_audio, clean_strength=clean_strength, export_format=export_format, embedder_model=embedder_model, resample_sr=resample_sr)
                    pbar.update(1)
            except Exception as e:
                raise RuntimeError(f"Đã xảy ra lỗi khi chuyển đổi tệp: {e}")

        elapsed_time = time.time() - start_time
        print(f"Chuyển đổi {input_path} thành {output_path.replace('wav', export_format)} hoàn tất sau {elapsed_time:.2f}s")

def change_rms(source_audio, source_rate, target_audio, target_rate, rate):
    rms2 = F.interpolate(torch.from_numpy(librosa.feature.rms(y=target_audio, frame_length=target_rate // 2 * 2, hop_length=target_rate // 2)).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze()
    return (target_audio * (torch.pow(F.interpolate(torch.from_numpy(librosa.feature.rms(y=source_audio, frame_length=source_rate // 2 * 2, hop_length=source_rate // 2)).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze(), 1 - rate) * torch.pow(torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6), rate - 1)).numpy())

def get_providers():
    ort_providers = onnxruntime.get_available_providers()
    providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in ort_providers else ["CPUExecutionProvider"]
    return providers

def device_config():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_mem = torch.cuda.get_device_properties(int(device.split(":")[-1])).total_memory // (1024**3) if device.startswith("cuda") else None

    if gpu_mem is not None and gpu_mem <= 4: return [1, 5, 30, 32]
    return [1, 6, 38, 41]

class Autotune:
    def __init__(self, ref_freqs):
        self.ref_freqs = ref_freqs
        self.note_dict = self.ref_freqs

    def autotune_f0(self, f0, f0_autotune_strength):
        autotuned_f0 = np.zeros_like(f0)

        for i, freq in enumerate(f0):
            autotuned_f0[i] = freq + (min(self.note_dict, key=lambda x: abs(x - freq)) - freq) * f0_autotune_strength

        return autotuned_f0

class VC:
    def __init__(self, tgt_sr):
        self.device_config = device_config()
        self.x_pad = self.device_config[0]
        self.x_query = self.device_config[1] 
        self.x_center = self.device_config[2]
        self.x_max = self.device_config[3] 
        self.sample_rate = 16000
        self.window = 160
        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max
        self.time_step = self.window / self.sample_rate * 1000
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.ref_freqs = [49.00, 51.91, 55.00, 58.27, 61.74, 65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00,  207.65, 220.00, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77, 1046.50]
        self.autotune = Autotune(self.ref_freqs)
        self.note_dict = self.autotune.note_dict

    def get_f0_pm(self, x, p_len):
        f0 = (parselmouth.Sound(x, self.sample_rate).to_pitch_ac(time_step=self.window / self.sample_rate * 1000 / 1000, voicing_threshold=0.6, pitch_floor=self.f0_min, pitch_ceiling=self.f0_max).selected_array["frequency"])
        pad_size = (p_len - len(f0) + 1) // 2

        if pad_size > 0 or p_len - len(f0) - pad_size > 0: f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
        return f0
 
    def get_f0_mangio_crepe(self, x, p_len, hop_length, model="full", onnx=False):
        providers = get_providers() if onnx else None

        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)

        audio = torch.unsqueeze(torch.from_numpy(x).to(self.device, copy=True), dim=0)
        if audio.ndim == 2 and audio.shape[0] > 1: audio = torch.mean(audio, dim=0, keepdim=True).detach()

        p_len = p_len or x.shape[0] // hop_length
        source = np.array(predict(audio.detach(), self.sample_rate, hop_length, self.f0_min, self.f0_max, model, batch_size=hop_length * 2, device=self.device, pad=True, providers=providers, onnx=onnx).squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan
        return np.nan_to_num(np.interp(np.arange(0, len(source) * p_len, len(source)) / p_len, np.arange(0, len(source)), source))

    def get_f0_crepe(self, x, model="full", onnx=False):
        providers = get_providers() if onnx else None
        
        f0, pd = predict(torch.tensor(np.copy(x))[None].float(), self.sample_rate, self.window, self.f0_min, self.f0_max, model, batch_size=512, device=self.device, return_periodicity=True, providers=providers, onnx=onnx)
        f0, pd = mean(f0, 3), median(pd, 3)
        f0[pd < 0.1] = 0

        return f0[0].cpu().numpy()

    def get_f0_fcpe(self, x, p_len, hop_length, onnx=False, legacy=False):
        providers = get_providers() if onnx else None

        model_fcpe = FCPE("fcpe" + (".onnx" if onnx else ".pt"), hop_length=int(hop_length), f0_min=int(self.f0_min), f0_max=int(self.f0_max), dtype=torch.float32, device=self.device, sample_rate=self.sample_rate, threshold=0.03, providers=providers, onnx=onnx) if legacy else FCPE("fcpe" + (".onnx" if onnx else ".pt"), hop_length=self.window, f0_min=0, f0_max=8000, dtype=torch.float32, device=self.device, sample_rate=self.sample_rate, threshold=0.006, providers=providers, onnx=onnx)
        f0 = model_fcpe.compute_f0(x, p_len=p_len)

        del model_fcpe
        return f0
    
    def get_f0_rmvpe(self, x, legacy=False, onnx=False):
        providers = get_providers() if onnx else None

        rmvpe_model = RMVPE("rmvpe" + (".onnx" if onnx else ".pt"), device=self.device, onnx=onnx, providers=providers)
        f0 = rmvpe_model.infer_from_audio_with_pitch(x, thred=0.03, f0_min=self.f0_min, f0_max=self.f0_max) if legacy else rmvpe_model.infer_from_audio(x, thred=0.03)

        del rmvpe_model
        return f0

    def get_f0_pyworld(self, x, filter_radius, model="harvest"):
        pw = PYWORLD()

        if model == "harvest": f0, t = pw.harvest(x.astype(np.double), fs=self.sample_rate, f0_ceil=self.f0_max, f0_floor=self.f0_min, frame_period=10)
        elif model == "dio": f0, t = pw.dio(x.astype(np.double), fs=self.sample_rate, f0_ceil=self.f0_max, f0_floor=self.f0_min, frame_period=10)
        else: raise ValueError(f"Phương pháp không xác định!")

        f0 = pw.stonemask(x.astype(np.double), self.sample_rate, t, f0)

        if filter_radius > 2 or model == "dio": f0 = signal.medfilt(f0, 3)
        return f0
    
    def get_f0_yin(self, x, hop_length, p_len):
        source = np.array(librosa.yin(x.astype(np.double), sr=self.sample_rate, fmin=self.f0_min, fmax=self.f0_max, hop_length=hop_length))
        source[source < 0.001] = np.nan

        return np.nan_to_num(np.interp(np.arange(0, len(source) * p_len, len(source)) / p_len, np.arange(0, len(source)), source))
    
    def get_f0_pyin(self, x, hop_length, p_len):
        f0, _, _ = librosa.pyin(x.astype(np.double), fmin=self.f0_min, fmax=self.f0_max, sr=self.sample_rate, hop_length=hop_length)
        source = np.array(f0)
        source[source < 0.001] = np.nan

        return np.nan_to_num(np.interp(np.arange(0, len(source) * p_len, len(source)) / p_len, np.arange(0, len(source)), source))

    def get_f0_hybrid(self, methods_str, x, p_len, hop_length, filter_radius):
        methods_str = re.search("hybrid\[(.+)\]", methods_str)
        if methods_str: methods = [method.strip() for method in methods_str.group(1).split("+")]

        f0_computation_stack, resampled_stack = [], []
        print(f"Tính toán cho các phương pháp: {methods}")

        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)

        for method in methods:
            f0 = None
            
            if method == "pm": f0 = self.get_f0_pm(x, p_len)
            elif method == "dio": f0 = self.get_f0_pyworld(x, filter_radius, "dio")
            elif method == "mangio-crepe-tiny": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "tiny")
            elif method == "mangio-crepe-tiny-onnx": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "tiny", onnx=True)
            elif method == "mangio-crepe-small": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "small")
            elif method == "mangio-crepe-small-onnx": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "small", onnx=True)
            elif method == "mangio-crepe-medium": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "medium")
            elif method == "mangio-crepe-medium-onnx": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "medium", onnx=True)
            elif method == "mangio-crepe-large": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "large")
            elif method == "mangio-crepe-large-onnx": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "large", onnx=True)
            elif method == "mangio-crepe-full": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "full")
            elif method == "mangio-crepe-full-onnx": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "full", onnx=True)
            elif method == "crepe-tiny": f0 = self.get_f0_crepe(x, "tiny")
            elif method == "crepe-tiny-onnx": f0 = self.get_f0_crepe(x, "tiny", onnx=True)
            elif method == "crepe-small": f0 = self.get_f0_crepe(x, "small")
            elif method == "crepe-small-onnx": f0 = self.get_f0_crepe(x, "small", onnx=True)
            elif method == "crepe-medium": f0 = self.get_f0_crepe(x, "medium")
            elif method == "crepe-medium-onnx": f0 = self.get_f0_crepe(x, "medium", onnx=True)
            elif method == "crepe-large": f0 = self.get_f0_crepe(x, "large")
            elif method == "crepe-large-onnx": f0 = self.get_f0_crepe(x, "large", onnx=True)
            elif method == "crepe-full": f0 = self.get_f0_crepe(x, "full")
            elif method == "crepe-full-onnx": f0 = self.get_f0_crepe(x, "full", onnx=True)
            elif method == "fcpe": f0 = self.get_f0_fcpe(x, p_len, int(hop_length))
            elif method == "fcpe-onnx": f0 = self.get_f0_fcpe(x, p_len, int(hop_length), onnx=True)
            elif method == "fcpe-legacy": f0 = self.get_f0_fcpe(x, p_len, int(hop_length), legacy=True)
            elif method == "fcpe-legacy-onnx": f0 = self.get_f0_fcpe(x, p_len, int(hop_length), onnx=True, legacy=True)
            elif method == "rmvpe": f0 = self.get_f0_rmvpe(x)
            elif method == "rmvpe-onnx": f0 = self.get_f0_rmvpe(x, onnx=True)
            elif method == "rmvpe-legacy": f0 = self.get_f0_rmvpe(x, legacy=True)
            elif method == "rmvpe-legacy-onnx": f0 = self.get_f0_rmvpe(x, legacy=True, onnx=True)
            elif method == "harvest": f0 = self.get_f0_pyworld(x, filter_radius, "harvest") 
            elif method == "yin": f0 = self.get_f0_yin(x, int(hop_length), p_len)
            elif method == "pyin": f0 = self.get_f0_pyin(x, int(hop_length), p_len)
            else: raise ValueError(f"Phương pháp không xác định!")
            
            f0_computation_stack.append(f0) 

        for f0 in f0_computation_stack:
            resampled_stack.append(np.interp(np.linspace(0, len(f0), p_len), np.arange(len(f0)), f0))

        return resampled_stack[0] if len(resampled_stack) == 1 else np.nanmedian(np.vstack(resampled_stack), axis=0)

    def get_f0(self, x, p_len, pitch, f0_method, filter_radius, hop_length, f0_autotune, f0_autotune_strength):
        if f0_method == "pm": f0 = self.get_f0_pm(x, p_len)
        elif f0_method == "dio": f0 = self.get_f0_pyworld(x, filter_radius, "dio")
        elif f0_method == "mangio-crepe-tiny": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "tiny")
        elif f0_method == "mangio-crepe-tiny-onnx": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "tiny", onnx=True)
        elif f0_method == "mangio-crepe-small": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "small")
        elif f0_method == "mangio-crepe-small-onnx": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "small", onnx=True)
        elif f0_method == "mangio-crepe-medium": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "medium")
        elif f0_method == "mangio-crepe-medium-onnx": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "medium", onnx=True)
        elif f0_method == "mangio-crepe-large": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "large")
        elif f0_method == "mangio-crepe-large-onnx": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "large", onnx=True)
        elif f0_method == "mangio-crepe-full": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "full")
        elif f0_method == "mangio-crepe-full-onnx": f0 = self.get_f0_mangio_crepe(x, p_len, int(hop_length), "full", onnx=True)
        elif f0_method == "crepe-tiny": f0 = self.get_f0_crepe(x, "tiny")
        elif f0_method == "crepe-tiny-onnx": f0 = self.get_f0_crepe(x, "tiny", onnx=True)
        elif f0_method == "crepe-small": f0 = self.get_f0_crepe(x, "small")
        elif f0_method == "crepe-small-onnx": f0 = self.get_f0_crepe(x, "small", onnx=True)
        elif f0_method == "crepe-medium": f0 = self.get_f0_crepe(x, "medium")
        elif f0_method == "crepe-medium-onnx": f0 = self.get_f0_crepe(x, "medium", onnx=True)
        elif f0_method == "crepe-large": f0 = self.get_f0_crepe(x, "large")
        elif f0_method == "crepe-large-onnx": f0 = self.get_f0_crepe(x, "large", onnx=True)
        elif f0_method == "crepe-full": f0 = self.get_f0_crepe(x, "full")
        elif f0_method == "crepe-full-onnx": f0 = self.get_f0_crepe(x, "full", onnx=True)
        elif f0_method == "fcpe": f0 = self.get_f0_fcpe(x, p_len, int(hop_length))
        elif f0_method == "fcpe-onnx": f0 = self.get_f0_fcpe(x, p_len, int(hop_length), onnx=True)
        elif f0_method == "fcpe-legacy": f0 = self.get_f0_fcpe(x, p_len, int(hop_length), legacy=True)
        elif f0_method == "fcpe-legacy-onnx": f0 = self.get_f0_fcpe(x, p_len, int(hop_length), onnx=True, legacy=True)
        elif f0_method == "rmvpe": f0 = self.get_f0_rmvpe(x)
        elif f0_method == "rmvpe-onnx": f0 = self.get_f0_rmvpe(x, onnx=True)
        elif f0_method == "rmvpe-legacy": f0 = self.get_f0_rmvpe(x, legacy=True)
        elif f0_method == "rmvpe-legacy-onnx": f0 = self.get_f0_rmvpe(x, legacy=True, onnx=True)
        elif f0_method == "harvest": f0 = self.get_f0_pyworld(x, filter_radius, "harvest") 
        elif f0_method == "yin": f0 = self.get_f0_yin(x, int(hop_length), p_len)
        elif f0_method == "pyin": f0 = self.get_f0_pyin(x, int(hop_length), p_len)
        elif "hybrid" in f0_method: f0 = self.get_f0_hybrid(f0_method, x, p_len, hop_length, filter_radius)
        else: raise ValueError(f"Phương pháp không xác định!")

        if f0_autotune: f0 = Autotune.autotune_f0(self, f0, f0_autotune_strength)

        f0 *= pow(2, pitch / 12)
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255

        return np.rint(f0_mel).astype(np.int32), f0.copy()

    def voice_conversion(self, model, net_g, sid, audio0, pitch, pitchf, index, big_npy, index_rate, version, protect):
        pitch_guidance = pitch != None and pitchf != None
        feats = torch.from_numpy(audio0).float()

        if feats.dim() == 2: feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()

        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
        inputs = {"source": feats.to(self.device), "padding_mask": padding_mask, "output_layer": 9 if version == "v1" else 12}

        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]

            if protect < 0.5 and pitch_guidance: feats0 = feats.clone()

            if (not isinstance(index, type(None)) and not isinstance(big_npy, type(None)) and index_rate != 0):
                npy = feats[0].cpu().numpy()
                score, ix = index.search(npy, k=8)
                weight = np.square(1 / score)
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                feats = (torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats)

            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            if protect < 0.5 and pitch_guidance: feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

            p_len = audio0.shape[0] // self.window

            if feats.shape[1] < p_len:
                p_len = feats.shape[1]
                if pitch_guidance:
                    pitch = pitch[:, :p_len]
                    pitchf = pitchf[:, :p_len]

            if protect < 0.5 and pitch_guidance:
                pitchff = pitchf.clone()
                pitchff[pitchf > 0] = 1
                pitchff[pitchf < 1] = protect
                pitchff = pitchff.unsqueeze(-1)
                feats = feats * pitchff + feats0 * (1 - pitchff)
                feats = feats.to(feats0.dtype)

            p_len = torch.tensor([p_len], device=self.device).long()
            audio1 = (net_g.run([net_g.get_outputs()[0].name], ({net_g.get_inputs()[0].name: feats.cpu().numpy().astype(np.float32), net_g.get_inputs()[1].name: p_len.cpu().numpy(), net_g.get_inputs()[2].name: np.array([sid.cpu().item()], dtype=np.int64), net_g.get_inputs()[3].name: np.random.randn(1, 192, p_len).astype(np.float32), net_g.get_inputs()[4].name: pitch.cpu().numpy().astype(np.int64), net_g.get_inputs()[5].name: pitchf.cpu().numpy().astype(np.float32)} if pitch_guidance else {net_g.get_inputs()[0].name: feats.cpu().numpy().astype(np.float32), net_g.get_inputs()[1].name: p_len.cpu().numpy(), net_g.get_inputs()[2].name: np.array([sid.cpu().item()], dtype=np.int64), net_g.get_inputs()[3].name: np.random.randn(1, 192, p_len).astype(np.float32)}))[0][0, 0])

        del feats, p_len, padding_mask, net_g

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return audio1
    
    def pipeline(self, model, net_g, sid, audio, pitch, f0_method, file_index, index_rate, pitch_guidance, filter_radius, tgt_sr, resample_sr, volume_envelope, version, protect, hop_length, f0_autotune, f0_autotune_strength):
        if file_index != "" and os.path.exists(file_index) and index_rate != 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as e:
                print(f"Đã xảy ra lỗi khi đọc chỉ mục: {e}")
                index = big_npy = None
        else: index = big_npy = None

        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts, audio_opt = [], []

        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)

            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]

            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(t - self.t_query + np.where(np.abs(audio_sum[t - self.t_query : t + self.t_query]) == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min())[0][0])

        s = 0
        t = None
        
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        p_len = audio_pad.shape[0] // self.window

        if pitch_guidance:
            pitch, pitchf = self.get_f0(audio_pad, p_len, pitch, f0_method, filter_radius, hop_length, f0_autotune, f0_autotune_strength)
            pitch, pitchf = pitch[:p_len], pitchf[:p_len]
            pitch, pitchf = torch.tensor(pitch, device=self.device).unsqueeze(0).long(), torch.tensor(pitchf, device=self.device).unsqueeze(0).float()

        for t in opt_ts:
            t = t // self.window * self.window
            audio_opt.append(self.voice_conversion(model, net_g, sid, audio_pad[s : t + self.t_pad2 + self.window], pitch[:, s // self.window : (t + self.t_pad2) // self.window] if pitch_guidance else None, pitchf[:, s // self.window : (t + self.t_pad2) // self.window] if pitch_guidance else None, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])    
            s = t
            
        audio_opt.append(self.voice_conversion(model, net_g, sid, audio_pad[t:], (pitch[:, t // self.window :] if t is not None else pitch) if pitch_guidance else None, (pitchf[:, t // self.window :] if t is not None else pitchf) if pitch_guidance else None, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])
        audio_opt = np.concatenate(audio_opt)

        if volume_envelope != 1: audio_opt = change_rms(audio, self.sample_rate, audio_opt, tgt_sr, volume_envelope)
        if resample_sr >= self.sample_rate and tgt_sr != resample_sr: audio_opt = librosa.resample(audio_opt, orig_sr=tgt_sr, target_sr=resample_sr, res_type="soxr_vhq")

        audio_max = np.abs(audio_opt).max() / 0.99
        if audio_max > 1: audio_opt /= audio_max

        if pitch_guidance: del pitch, pitchf
        del sid

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return audio_opt

class VoiceConverter:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.hubert_model = None
        self.tgt_sr = None 
        self.net_g = None 
        self.vc = None
        self.cpt = None  
        self.version = None 
        self.n_spk = None  
        self.use_f0 = None  
        self.loaded_model = None
        self.vocoder = "Default"

    def load_embedders(self, embedder_model):
        try:
            models, _, _ = checkpoint_utils.load_model_ensemble_and_task([embedder_model + '.pt'], suffix="")
        except Exception as e:
            raise Exception(f"Xảy ra lỗi khi đọc mô hình nhúng: {e}")
        self.hubert_model = models[0].to(self.device).float().eval()

    def convert_audio(self, audio_input_path, audio_output_path, model_path, index_path, embedder_model, pitch, f0_method, index_rate, volume_envelope, protect, hop_length, f0_autotune, f0_autotune_strength, filter_radius, clean_audio, clean_strength, export_format, resample_sr = 0, sid = 0):
        try:
            self.get_vc(model_path, sid)
            audio = load_audio(audio_input_path)

            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1: audio /= audio_max

            if not self.hubert_model: 
                if not os.path.exists(embedder_model + '.pt'): raise FileNotFoundError(f"Không tìm thấy mô hình: {embedder_model}")  
                self.load_embedders(embedder_model)

            if self.tgt_sr != resample_sr >= 16000: self.tgt_sr = resample_sr
            target_sr = min([8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 96000], key=lambda x: abs(x - self.tgt_sr))

            audio_output = self.vc.pipeline(model=self.hubert_model, net_g=self.net_g, sid=sid, audio=audio, pitch=pitch, f0_method=f0_method, file_index=(index_path.strip().strip('"').strip("\n").strip('"').strip().replace("trained", "added")), index_rate=index_rate, pitch_guidance=self.use_f0, filter_radius=filter_radius, tgt_sr=self.tgt_sr, resample_sr=target_sr, volume_envelope=volume_envelope, version=self.version, protect=protect, hop_length=hop_length, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength)
            
            if clean_audio:
                from lib.noisereduce import reduce_noise
                audio_output = reduce_noise(y=audio_output, sr=target_sr, prop_decrease=clean_strength) 

            sf.write(audio_output_path, audio_output, target_sr, format=export_format)
        except Exception as e:
            raise RuntimeError(f"Đã xảy ra lỗi khi chuyển đổi: {e}")

    def get_vc(self, weight_root, sid):
        if sid == "" or sid == []:
            self.cleanup()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        if not self.loaded_model or self.loaded_model != weight_root:
            self.loaded_model = weight_root
            self.load_model()
            if self.cpt is not None: self.setup()

    def cleanup(self):
        if self.hubert_model is not None:
            del self.net_g, self.n_spk, self.vc, self.hubert_model, self.tgt_sr
            self.hubert_model = self.net_g = self.n_spk = self.vc = self.tgt_sr = None

            if torch.cuda.is_available(): torch.cuda.empty_cache()

        del self.net_g, self.cpt
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        self.cpt = None

    def load_model(self):
        if os.path.isfile(self.loaded_model):
            sess_options = onnxruntime.SessionOptions()
            sess_options.log_severity_level = 3
            self.cpt = onnxruntime.InferenceSession(self.loaded_model, sess_options=sess_options, providers=get_providers())
        else: self.cpt = None

    def setup(self):
        if self.cpt is not None:
            model = onnx.load(self.loaded_model)
            metadata_dict = None
            for prop in model.metadata_props:
                if prop.key == "model_info":
                    metadata_dict = json.loads(prop.value)
                    break
            self.net_g = self.cpt
            self.tgt_sr = metadata_dict.get("sr", 32000)
            self.use_f0 = metadata_dict.get("f0", 1)
            self.vc = VC(self.tgt_sr)

if __name__ == "__main__": 
    run_convert_script(pitch=0, filter_radius=3, index_rate=0.7, volume_envelope=1, protect=0.5, hop_length=64, 
    f0_method="rmvpe-legacy-onnx", input_path="./audios.mp3", output_path="./output.wav", 
    pth_path="./model.onnx", index_path="./model.index", f0_autotune=False, 
    f0_autotune_strength=1, clean_audio=False, clean_strength=0.7, export_format="wav", 
    embedder_model="contentvec_base", resample_sr=0, split_audio=False)