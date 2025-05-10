import os
import torch
import librosa
import pandas as pd
import time
import threading
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from concurrent.futures import ThreadPoolExecutor, as_completed

model_path = r'C:\whisper-large-v3-turbo'
audio_dir = r'C:\Users\ENJD\Desktop\BA'
output_txt_dir = os.path.join(audio_dir, 'transcripts')
transcribed_log = os.path.join(audio_dir, 'transcribed.txt')
error_log = os.path.join(audio_dir, 'transcribe_errors.txt')
os.makedirs(output_txt_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_mode = False
test_limit = 10

# load
print('🔧 loading...')
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_path, torch_dtype=torch.float16, trust_remote_code=True).to(device)

# show gpu
if device == 'cuda' and torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    mem_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
    mem_reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
    print(f"🧠 GPU: {gpu_name}")
    print(f"🧠 memery:  {mem_allocated:.0f} MB / remaind {mem_reserved:.0f} MB")
else:
    print("⚠️ in cpu")

#finished load
completed = set()
if os.path.exists(transcribed_log):
    with open(transcribed_log, 'r', encoding='utf-8') as f:
        completed = set(line.strip() for line in f if line.strip())

# get file
all_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(('.m4a', '.mp3', '.wav'))]
pending_files = [f for f in all_files if os.path.splitext(f)[0] not in completed]
if test_mode:
    pending_files = pending_files[:test_limit]
total_files = len(pending_files)

#line lock
lock = threading.Lock()
data = []
task_status = []
progress = {
    "processed": 0,
    "start_time": time.time()
}

# transcribe
def transcribe(filename):
    bv = os.path.splitext(filename)[0]
    audio_path = os.path.join(audio_dir, filename)
    if os.path.getsize(audio_path) < 100 * 1024:
        with lock:
            task_status.append({"file": filename, "status": "skipped", "error": "file too small"})
        return None

    start_time = time.time()
    for attempt in range(3):
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            segment_duration = 30
            segment_samples = segment_duration * sr
            num_segments = int(np.ceil(len(audio) / segment_samples))

            full_text = ''
            for i in range(num_segments):
                segment = audio[i * segment_samples: (i + 1) * segment_samples]
                inputs = processor(segment, sampling_rate=16000, return_tensors="pt")
                inputs = {k: v.to(device, dtype=torch.float16) if v.dtype == torch.float32 else v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    generated_ids = model.generate(inputs["input_features"], max_new_tokens=128)
                    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    full_text += text.strip() + ' '

            with open(os.path.join(output_txt_dir, f"{bv}.txt"), "w", encoding="utf-8", errors="ignore") as f_out:
                f_out.write(full_text.strip())

            with lock:
                with open(transcribed_log, "a", encoding="utf-8", errors="ignore") as f_log:
                    f_log.write(bv + "\n")
                data.append({"BV Number": bv, "Transcript": full_text.strip()})
                task_status.append({"file": filename, "status": "success", "error": ""})
                progress["processed"] += 1

            duration = time.time() - start_time
            print(f"🕒 {filename} timecost：{duration:.2f} s")
            return bv
        except Exception as e:
            err_msg = str(e)
            if attempt == 2:
                with lock:
                    task_status.append({"file": filename, "status": "failed", "error": err_msg})
                    with open(error_log, "a", encoding="utf-8", errors="ignore") as f_err:
                        f_err.write(f"{filename}: {err_msg}\n")
            time.sleep(2)
    return None

# monitor
start_time = time.time()
completed_count = 0

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {executor.submit(transcribe, fname): fname for fname in pending_files}

    with tqdm(total=total_files, desc="🎧 转录中", ncols=120) as pbar:
        for future in as_completed(futures):
            _ = future.result()
            completed_count += 1
            elapsed = time.time() - start_time
            speed = completed_count / (elapsed / 60) if elapsed > 0 else 0
            remaining = total_files - completed_count
            eta_minutes = remaining / speed if speed > 0 else 0
            eta_h = int(eta_minutes // 60)
            eta_m = int(eta_minutes % 60)
            pbar.set_description(f"🎧 : {completed_count}/{total_files} | speed: {speed:.2f}/min | ETA: {eta_h}h{eta_m}m")
            pbar.update(1)


df = pd.DataFrame(data)
if not df.empty:
    df.to_csv(os.path.join(audio_dir, 'transcripts.csv'), index=False, encoding='utf-8')
    df.to_json(os.path.join(audio_dir, 'transcripts.json'), force_ascii=False, indent=2)
    df.to_excel(os.path.join(audio_dir, 'transcripts.xlsx'), index=False)


status_df = pd.DataFrame(task_status)
status_df.to_excel(os.path.join(audio_dir, 'transcription_status.xlsx'), index=False)

print('✅ all done：')
print('📄 transcripts.csv / .json / .xlsx')
print('📄 transcription_status.xlsx ')
