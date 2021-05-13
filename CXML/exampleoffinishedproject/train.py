import torch
import os
from sample import sample_sequence
from model import CompleteModel, Config
import torch.optim as optim
import glob
import numpy as np
import torch.cuda as tcuda
import sys
import tqdm
import time
import librosa
import scipy.io.wavfile as wvf

END_TOKEN = librosa.cqt(librosa.core.tone(5120, sr=44100, duration=1.0))[:, 0]
DATASET_REDUCTION = 25
PATH = 'XXXXXXXXX' #nice try.
MODEL_NAME = "test1"
LENGTH = 10
SAMPLES = 2
TEMPERATURE = 1.0
TOP_K = 40
SAVE_EVERY = 5000
SAMPLE_EVERY = 10000000000
SAMPLE_POOP = 500000000000
LOSS_FRAME = 2000
BATCH_SIZE = 1
LR = 1e-3
GROW_THRESHOLD = 0.01
MAX_LAYER = 6
MINIMUM_SEPARATION = 100000
STACK_LOSS = False


def save_chunks(chunks):
    import pickle
    if not os.path.isdir(os.path.join('dataset', MODEL_NAME)):
        os.makedirs(os.path.join('dataset', MODEL_NAME))
    with open(os.path.join('dataset', MODEL_NAME, 'encoded_data'), 'wb') as file:
        pickle.dump(chunks, file)


def load_chunks():
    if not os.path.isdir(os.path.join('dataset', MODEL_NAME)):
        os.makedirs(os.path.join('dataset', MODEL_NAME))
    if not os.path.exists(os.path.join('dataset', MODEL_NAME, 'encoded_data')):
        return None
    import pickle
    with open(os.path.join('dataset', MODEL_NAME, 'encoded_data'), 'rb') as file:
        chunks = pickle.load(file)
    return chunks


def load_dataset(path):
    """expects audio in dataset to be 16-bit PCM, 44.1 KHz, mono"""
    qdata = load_chunks()
    if qdata is None:
        qdata = []
        for filename in glob.glob(path):
            print(filename)
            rate, data = wvf.read(filename)
            p = lambda t: 1.0 if t/32768.0 > 1.0 else (-1.0 if t/32768.0 < -1.0 else t/32768.0)
            data = data.astype(np.float32)
            ddata = np.array([p(x) for x in data])
            if rate == 44100:  # (bins, len)
                qd = librosa.cqt(ddata, sr=rate, n_bins=48*7, bins_per_octave=48)
                print(qd.shape)
                #np.append(qd, END_TOKEN, axis=1)
                rdata = qd.real.astype(np.float32)
                idata = qd.imag.astype(np.float32)
                qd = np.append(rdata, idata, axis=0)
                print(qd.shape)
                qdata.append(qd)

        save_chunks(qdata)

    return qdata

def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class Sampler(object):
    """borrowed from nshepperd's gpt2 sampling code"""

    def __init__(self, chunks, seed=None):
        self.chunks = chunks
        self.total_size = sum(chunk.shape[1] for chunk in chunks)
        self.boundaries = [0]
        for i in range(len(chunks)):
            self.boundaries.append(self.boundaries[-1] + chunks[i].shape[1])
        self.rs = np.random.RandomState(seed=seed)


    def sample(self, length):
        assert length < self.total_size // len(self.chunks)
        while True:
            index = self.rs.randint(0, self.total_size - length - 1)
            i = binary_search(lambda j: self.boundaries[j] > index, 0, len(self.boundaries) - 1) - 1
            if self.boundaries[i + 1] > index + length:
                wc = index - self.boundaries[i]
                return self.chunks[i][:, wc:wc + length]


def main():
    global SAMPLE_EVERY
    global SAVE_EVERY
    global SAMPLE_POOP
    device = torch.device("cuda" if tcuda.is_available() else "cpu")
    counter = 1
    # initialize
    logs = []
    config = Config()

    model = CompleteModel(config)
    model.to(device)

    def log(txt):
        print(txt)
        logs.append(txt)

    def save():
        if not os.path.isdir("checkpoint/" + MODEL_NAME):
            os.makedirs("checkpoint/" + MODEL_NAME)
        #delete previous model
        prevmodels = [f for f in os.listdir(os.path.join("checkpoint", MODEL_NAME)) if "model" in f]
        for f in prevmodels:
            os.remove(os.path.join("checkpoint", MODEL_NAME, f))
        log("Saving " + os.path.join("checkpoint", MODEL_NAME, "model-{}").format(counter))
        torch.save(model.state_dict(), os.path.join("checkpoint", MODEL_NAME, "model-{}").format(counter))
        with open(os.path.join("checkpoint", MODEL_NAME, "counter"), "w") as cf:
            cf.write(str(counter))
        with open(os.path.join("checkpoint", MODEL_NAME, "log2.txt"), "w", encoding="utf-8") as logf:
            logf.writelines(logs)
        log("Saved.")

    def load():
        nonlocal counter
        nonlocal logs
        if not os.path.isdir(MODEL_NAME):
            os.makedirs(MODEL_NAME)
        if not os.path.exists(os.path.join("checkpoint", MODEL_NAME, "counter")):
            return False
        log("Loading model...")
        with open(os.path.join("checkpoint", MODEL_NAME, "counter"), "r") as cf:
            counter = int(cf.read())
        model.load_state_dict(torch.load(os.path.join("checkpoint", MODEL_NAME, "model-{}").format(counter)))
        with open(os.path.join("checkpoint", MODEL_NAME, "log2.txt"), "r", encoding="utf-8") as logf:
            logs = logf.readlines()
        log("Model loaded.")
        counter += 1
        return True

    def generate_samples(data_sampler):
        for i in range(SAMPLES):
            sample_sequence(model=model, length=LENGTH, path="{p}-{j}.wav".format(p=counter, j=i),
                            context=None, start_token=data_sampler.sample(512),
                            batch_size=1, temperature=TEMPERATURE, top_k=TOP_K, device=device)

    def get_batch():
        nonlocal config
        return [data_sampler.sample(config.n_ctx*(2**config.n_stacks) + 1) for _ in range(BATCH_SIZE)]

    '''def get_examp():
        mydata = data_sampler.chunks[0]
        freqfat, phsfat = np.split(mydata, 2, axis=0)
        freql = freqfat[:336, :]
        phsl = phsfat[:336, :]
        complxl = freql.astype(np.complex) + 1j * phsl.astype(np.complex)
        moreoutl = librosa.core.icqt(complxl, sr=44100, hop_length=512,
                                    bins_per_octave=48)
        h = lambda t: 32767.0 if t * 32768.0 > 32767.0 else (-32768.0 if t * 32768.0 < -32768.0 else t * 32768.0)
        prout = np.array([h(x) for x in moreoutl]).astype(np.short)
        wvf.write("!special122l.wav", 44100, prout)
        prout = np.array([h(x) for x in moreouth]).astype(np.short)
        wvf.write("!special122hl.wav", 44100, prout)
        prout = np.array([h(x) for x in out]).astype(np.short)
        wvf.write("!special122m.wav", 44100, prout)
        print("successfully saved output to {}".format("asdawdasdw"))'''

    # load dataset into chunks
    log("loading dataset...")
    chunks = load_dataset(PATH)
    data_sampler = Sampler(chunks)
    log("dataset has " + str(data_sampler.total_size) + " tokens.")

    #get_examp()

    # check for and load available save state
    log("checking for previous model...")
    mfound = load()
    log("successfully retrieved previous model ({})".format(counter) if mfound else "no model found. using new model.")

    # train
    optimizer = optim.Adam(model.parameters(), lr=LR)

    #generate_samples(data_sampler)

    avg_loss = (0.0, 0.0)
    start_time = time.time()

    try:
        while True:
            if counter % SAVE_EVERY == 0:
                save()
            if counter % SAMPLE_EVERY == 0:
                generate_samples(data_sampler)
            if counter % SAMPLE_POOP == 0:
                SAMPLE_POOP = SAMPLE_POOP * 2
                SAMPLE_EVERY = SAMPLE_EVERY * 2
            # train iteration
            inputs = torch.tensor(get_batch(), dtype=torch.float, device=device).permute(0, 2, 1)
            #(batch, seq, dmodel)

            optimizer.zero_grad()
            '''if STACK_LOSS:
                loss, stack_loss = model(inputs[:, :-1, :], labels=inputs[:, 1:, :], stack_labels=True)
                for sloss in zip(stack_loss):
                    sloss.backward()
            else:
                loss = model(inputs[:, :-1, :], labels=inputs[:, 1:, :])'''
            loss = model(inputs[:, :-1, :], labels=inputs[:, 1:, :])
            loss.backward()
            optimizer.step()

            v_loss = loss.item()
            avg_loss = (avg_loss[0] * 0.999 + v_loss, avg_loss[1] * 0.999 + 1.0)
            log("[{counter} | {time:2.2f}] loss={loss:3.4f} avg={avg:3.4f} fixed={g:4.4f}".format(counter=counter,
                                                                                   time=time.time() - start_time,
                                                                                   loss=v_loss,
                                                                                   avg=avg_loss[0] / avg_loss[1],
                                                                                    g=(avg_loss[0] / avg_loss[1])/(1024*720)))
            if counter % LOSS_FRAME == 1:
                frame_loss = avg_loss
            counter += 1
    except KeyboardInterrupt:
        log("process interrupted. saving and shutting down...")
        save()


if __name__ == '__main__':
    main()