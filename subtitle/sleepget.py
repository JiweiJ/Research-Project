import os
import subprocess
import time
import random
import pandas as pd
import threading
from queue import Queue

output_dir = r'C:\Users\ENJD\Desktop\BA'
os.makedirs(output_dir, exist_ok=True)
df = pd.read_excel('selected_bv_list.xlsx')

finished_file = 'finished.txt'
invalid_file = 'invalid.txt'
finished = set()
invalid = set()

if os.path.exists(finished_file):
    with open(finished_file, 'r', encoding='utf-8') as f:
        finished = set(line.strip() for line in f if line.strip())

if os.path.exists(invalid_file):
    with open(invalid_file, 'r', encoding='utf-8') as f:
        invalid = set(line.strip() for line in f if line.strip())

q = Queue()
for idx, row in df.iterrows():
    bv = str(row['BV'])
    link = str(row['Link'])
    if bv not in finished and bv not in invalid:
        q.put((idx + 1, len(df), bv, link))

lock = threading.Lock()

pause_event = threading.Event()

#time sleep
def pause_controller():
    while not q.empty():
        time.sleep(20 * 60)  
        print('⏸️ sleep one hour')
        pause_event.set()    
        time.sleep(60 * 60)  
        pause_event.clear()  
        print('▶️ go ahead')

# sleep check
def download_worker():
    while not q.empty():
        while pause_event.is_set():
            time.sleep(1)
            
        try:
            idx, total, bv, link = q.get_nowait()
        except Exception:
            break

        out_file = os.path.join(output_dir, f'{bv}.m4a')
        print(f'▶️ downloading {idx}/{total} → BV: {bv}')

        for attempt in range(3):
            try:
                command = f'yt-dlp -f ba --user-agent "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" -o "{out_file}" "{link}"'
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    with lock:
                        with open(finished_file, 'a', encoding='utf-8') as f_done:
                            f_done.write(bv + '\n')
                    print(f'✅ : {bv}')
                    break
                elif '403' in result.stderr or '404' in result.stderr:
                    with lock:
                        with open(invalid_file, 'a', encoding='utf-8') as f_bad:
                            f_bad.write(bv + '\n')
                    print(f'❌ （{bv}）: {link}')
                    break
                else:
                    print(f'⚠️  {attempt+1} false: {bv}')
                    time.sleep(random.uniform(5, 10))
            except Exception as e:
                print(f'❌ error: {bv} → {str(e)}')
                time.sleep(random.uniform(1, 2))
        q.task_done()

# multiline
num_threads = 5
threads = []
for _ in range(num_threads):
    t = threading.Thread(target=download_worker)
    t.start()
    threads.append(t)

pause_thread = threading.Thread(target=pause_controller, daemon=True)
pause_thread.start()

for t in threads:
    t.join()

print('🎉 all done')
