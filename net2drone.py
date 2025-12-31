import threading
import queue
import numpy as np
import sounddevice as sd
from scapy.all import sniff
import tkinter as tk
import time
import os
import random
from tkinterdnd2 import TkinterDnD, DND_FILES

# GLOBALS
DAC_SAMPLE_RATE = 44100
CHANNELS = 1
audio_queue = queue.Queue(maxsize=50)

rate = 44100
density = 100
level = 1.0
acc_decay = 0.995
acc_feedback = 0.92
dir_feedback = 0.92
acc_buffersize = 2048
probability = 0.5
blocksize = 1024
max_blocksize = 4 * blocksize

acc_active = True
direct_active = True

accumulator = np.zeros(acc_buffersize, dtype=np.float32)
direct_memory = np.zeros(4096, dtype=np.float32)

lock = threading.Lock()
stream = None
BUFFER_SIZES = [512, 1024, 2048, 4096, 8192]

# AUDIO STREAM
def recreate_stream(bs):
    global stream, blocksize, max_blocksize
    if stream:
        stream.stop()
        stream.close()
        time.sleep(0.05)
    blocksize = int(np.clip(bs, 64, acc_buffersize))
    max_blocksize = 4 * blocksize
    stream = sd.OutputStream(
        samplerate=DAC_SAMPLE_RATE,
        blocksize=blocksize,
        channels=CHANNELS,
        dtype='float32',
        callback=audio_callback
    )
    stream.start()

# MAIN POOL
class PoolByteSource:
    def __init__(self):
        self.raw = None
        self.active = False
        self.ptr = 0
        self.current_file = None

    def load(self, path):
        if not os.path.isfile(path):
            print("file non valido")
            return
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size == 0:
            print("file vuoto")
            return
        # normalizza [-1,1]
        norm_audio = (raw.astype(np.float32)-128)/128
        with lock:
            self.raw = norm_audio
            self.ptr = 0
            self.active = True
            self.current_file = path
        print(f">>> audio pool: {os.path.basename(path)}")

    def next_file(self):
        if not self.current_file:
            return
        directory = os.path.dirname(self.current_file)
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]
        if not files: return
        next_path = os.path.join(directory, random.choice(files))
        self.load(next_path)

    def stop(self):
        with lock:
            self.active = False
            self.raw = None

    def next_byte(self):
        with lock:
            if not self.active or self.raw is None: return None
            b = int((self.raw[self.ptr % len(self.raw)]+1.0)*127.5)
            self.ptr += 1
            return b

    def next_chunk(self, size):
        with lock:
            if not self.active or self.raw is None: return None
            idxs = (np.arange(size) + self.ptr) % len(self.raw)
            chunk = self.raw[idxs]
            self.ptr += size
            return chunk.astype(np.float32)

pool_source = PoolByteSource()

# CONTROL POOL
class ControlPool:
    def __init__(self):
        self.raw = None
        self.ptr = 0
        self.active = False
        self.current_file = None

    def load(self, path):
        if not os.path.isfile(path):
            print("file non valido")
            return
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size==0: 
            print("file vuoto")
            return
        with lock:
            self.raw = raw
            self.ptr = 0
            self.active = True
            self.current_file = path
        print(f">>> control pool: {os.path.basename(path)}")

    def next_file(self):
        if not self.current_file:
            return
        directory = os.path.dirname(self.current_file)
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]
        if not files: return
        next_path = os.path.join(directory, random.choice(files))
        self.load(next_path)

    def stop(self):
        with lock:
            self.active = False
            self.raw = None

    def step(self):
        with lock:
            if not self.active or self.raw is None: return None
            b = int(self.raw[self.ptr % len(self.raw)])
            self.ptr += 1
            return b

control_pool = ControlPool()

# CONTROL POOL THREAD
def smooth(val, target, amt=0.15): 
    return val + (target - val) * amt

def sigmoid(x, k=6):
    return 1 / (1 + np.exp(-k*(x - 0.5)))

def bias(x, pow=2):
    return 1 - (1 - x)**pow

def control_pool_thread():
    global blocksize, rate, level, acc_decay, probability

    # target 
    blocksize_target = blocksize
    rate_target = rate
    level_target = level
    acc_decay_target = acc_decay
    probability_target = probability

    # time
    last_update = {
        "blocksize": time.time(),
        "rate": time.time(),
        "level": time.time(),
        "acc_decay": time.time(),
        "probability": time.time(),
    }

    # valori time in secondi
    update_interval = {
        "blocksize": 1.5,
        "rate": 2.25,
        "level": 0.9,
        "acc_decay": 0.7,
        "probability": 1.2,
    }

    while True:
        byte = control_pool.step()
        if byte is not None:
            b = byte / 255.0
            now = time.time()
                        
            if now - last_update["blocksize"] > update_interval["blocksize"]:
                blocksize_target = 128 + (8192-128) * sigmoid(b, k=5)
                last_update["blocksize"] = now
            
            if now - last_update["rate"] > update_interval["rate"]:
               rate_target = 8000 + (88200-8000) * bias(b, pow=3)
               last_update["rate"] = now

            if now - last_update["level"] > update_interval["level"]:
               level_target = 0.1 + 0.9 * sigmoid(b, k=6)
               last_update["level"] = now

            if now - last_update["acc_decay"] > update_interval["acc_decay"]:
               acc_decay_target = 0.8 + 0.2 * bias(b, pow=2)
               last_update["acc_decay"] = now

            if now - last_update["probability"] > update_interval["probability"]:
               probability_target = 0.3 + 0.7 * bias(b, pow=0.3)
               last_update["probability"] = now

        # smooth
        with lock:
            blocksize = int(smooth(blocksize, blocksize_target, 0.1))
            rate = int(smooth(rate, rate_target, 0.09))
            level = smooth(level, level_target, 0.2)
            acc_decay = smooth(acc_decay, acc_decay_target, 0.2)
            probability = smooth(probability, probability_target, 0.1)

        time.sleep(0.05)

threading.Thread(target=control_pool_thread, daemon=True).start()

# UTILITY
def bytes_to_float(raw_bytes):
    return np.array(raw_bytes, dtype=np.float32)

def resample_block(block, target_len):
    if len(block)==0: return np.zeros(target_len, dtype=np.float32)
    x_old = np.linspace(0,1,len(block))
    x_new = np.linspace(0,1,target_len)
    return np.interp(x_new,x_old,block).astype(np.float32)

def process_direct(raw):
    global direct_memory
    if not direct_active: return None
    b = raw
    n = min(len(b), len(direct_memory))
    direct_memory *= dir_feedback
    direct_memory[:n] += b[:n]
    np.clip(direct_memory,-1.0,1.0,out=direct_memory)
    return direct_memory[:n].copy()

def process_accumulator(raw):
    global accumulator
    if not acc_active: return None
    b = raw
    n = min(len(b), acc_buffersize)
    accumulator *= acc_decay
    accumulator += acc_feedback*accumulator
    np.clip(accumulator,-1.0,1.0,out=accumulator)
    mask = (np.random.rand(n)<probability).astype(np.float32)
    accumulator[:n] += b[:n]*mask
    return accumulator.copy()

def audio_callback(outdata, frames, t, status):
    out = np.zeros(frames, dtype=np.float32)
    filled = 0
    while filled<frames:
        try: block = audio_queue.get_nowait()
        except queue.Empty: break
        take = min(len(block), frames-filled)
        out[filled:filled+take] = block[:take]
        if take<len(block):
            audio_queue.put_nowait(block[take:])
        filled += take
    max_val = np.max(np.abs(out))
    if max_val > 1e-6:
        out = out / max_val  
    out = np.clip(out, -1.0, 1.0)  
    outdata[:, 0] = out * 0.99
# PACKET HANDLER 
NEXT_PROB_TCP = 0.02
NEXT_PROB_UDP = 0.03
last_trigger = {"pool_a": 0, "pool_b": 0}
COOLDOWN = 1.0

def on_packet(pkt):
    # audio normale
    if pool_source.active:
        size = len(bytes(pkt))
        chunk = pool_source.next_chunk(min(size,1024))
        if chunk is not None:
            blocks = []
            d = process_direct(chunk)
            if d is not None: blocks.append(d)
            a = process_accumulator(chunk)
            if a is not None: blocks.append(a)
            if blocks:
                min_len = min(len(b) for b in blocks)
                audio_block = sum(b[:min_len] for b in blocks)
                maxv = np.max(np.abs(audio_block))
                if maxv > 1e-6:
                    audio_block /= maxv
                    factor = rate / DAC_SAMPLE_RATE
                    target_len = max(1, int(len(audio_block)*factor))
                    audio_block = resample_block(audio_block,target_len)
                    audio_block = audio_block[:max_blocksize]
                    try: audio_queue.put_nowait(audio_block)
                    except queue.Full: pass

    # probabilizzazione del next
    now = time.time()
    if pkt.haslayer("TCP") and (now - last_trigger["pool_a"] > COOLDOWN):
        byte_val = pkt["TCP"].seq % 256
        prob = byte_val / 255.0
        if random.random() < prob * NEXT_PROB_TCP:
            pool_source.next_file()
            last_trigger["pool_a"] = now
    elif pkt.haslayer("UDP") and (now - last_trigger["pool_b"] > COOLDOWN):
        byte_val = pkt["UDP"].len % 256
        prob = byte_val / 255.0
        if random.random() < prob * NEXT_PROB_UDP:
            control_pool.next_file()
            last_trigger["pool_b"] = now

threading.Thread(target=lambda: sniff(prn=on_packet, store=False), daemon=True).start()

# GUI
root = TkinterDnD.Tk()
root.title("net2drone")

direct_var = tk.BooleanVar(master=root, value=True)
acc_var = tk.BooleanVar(master=root, value=True)

# AUDIO POOL DROP
frame_drop = tk.Frame(root, bd=2, relief="ridge")
frame_drop.pack(padx=10,pady=10, fill="x")
lab_drop = tk.Label(frame_drop, text="audio", height=3)
lab_drop.pack(fill="x")
lab_drop.drop_target_register(DND_FILES)
lab_drop.dnd_bind("<<Drop>>", lambda e: pool_source.load(e.data.strip("{}")))
tk.Button(frame_drop,text="stop",command=pool_source.stop).pack(pady=2)
tk.Button(frame_drop,text="next",command=pool_source.next_file).pack(pady=2)

# CONTROL POOL DROP
frame_ctrl = tk.Frame(root, bd=2, relief="ridge")
frame_ctrl.pack(padx=10,pady=10, fill="x")
lab_ctrl = tk.Label(frame_ctrl,text="control", height=3)
lab_ctrl.pack(fill="x")
lab_ctrl.drop_target_register(DND_FILES)
lab_ctrl.dnd_bind("<<Drop>>", lambda e: control_pool.load(e.data.strip("{}")))
tk.Button(frame_ctrl,text="stop",command=control_pool.stop).pack(pady=2)
tk.Button(frame_ctrl,text="next",command=control_pool.next_file).pack(pady=2)

# COMMANDS
tk.Label(root,text="network density").pack()
tk.Scale(root, from_=0,to=100,orient="horizontal",
         command=lambda v: globals().update(density=int(v)), length=400).pack()
tk.Checkbutton(root,text="direct",variable=direct_var,
               command=lambda: globals().update(direct_active=bool(direct_var.get()))).pack(pady=3)
tk.Checkbutton(root,text="accumulator",variable=acc_var,
               command=lambda: globals().update(acc_active=bool(acc_var.get()))).pack(pady=3)

tk.Label(root,text="buffer size").pack()
def update_acc_buffersize(idx):
    global acc_buffersize, accumulator
    acc_buffersize = BUFFER_SIZES[int(idx)]
    accumulator = np.zeros(acc_buffersize, dtype=np.float32)
    recreate_stream(blocksize)

tk.OptionMenu(root, tk.IntVar(value=2), *range(len(BUFFER_SIZES)), command=update_acc_buffersize).pack(pady=3)

# INIT
recreate_stream(blocksize)
root.mainloop()
