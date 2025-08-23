# Colab-friendly RF classification runner with optional DPU/CPU backends

import os, math, time, threading, sys
import numpy as np

# ───────── User params (edit as needed) ─────────
RF_INPUT_PATH = "/workspace/rf_input.npy"      # required
RF_CLASSES_PATH = "/workspace/rf_classes.npy"  # optional, for accuracy
RF_SNRS_PATH = "/workspace/rf_snrs.npy"        # optional

# CPU (Keras) fallback model: should output LOGITS (no softmax)
MODEL_PATH = "/workspace/SRNET_Q_8IN_T1.h5"  # set to None if not using CPU model
BATCH_SIZE = 32
NUM_FRAMES = 256            # how many samples to run
NUM_THREADS = 1             # threading mainly for DPU; on CPU it won't speed up much

# DPU mode: requires xir/vart and a compiled .xmodel
USE_DPU = False
XMODEL_PATH = "/workspace/vai_c_output/rf_F32_t1.xmodel"  # if USE_DPU=True

# Label names (optional, used for readable top-1)
MODS = ['BPSK','QPSK','GMSK','FM','OOK','OQPSK','8PSK','16QAM','AM-SSB-WC','AM-DSB-SC']
# ────────────────────────────────────────────────


def softmax_stable(x):
    """Vectorized, numerically stable softmax for 2D array (N,C)."""
    x = x.astype(np.float32)
    m = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)

def load_data():
    X = np.load(RF_INPUT_PATH)  # expected (N,1024,1,2) float32
    if NUM_FRAMES is not None:
        X = X[:NUM_FRAMES]
    y = None
    snr = None
    if os.path.exists(RF_CLASSES_PATH):
        y = np.load(RF_CLASSES_PATH)  # (N,) ints 0..9
        y = y[:len(X)]
    if os.path.exists(RF_SNRS_PATH):
        snr = np.load(RF_SNRS_PATH)
        snr = snr[:len(X)]
    return X, y, snr

# ========= CPU (Keras) backend =========
class CpuRunner:
    def __init__(self, model_path, batch_size=32):
        import tensorflow as tf
        self.tf = tf
        self.model = tf.keras.models.load_model(model_path)
        self.bs = batch_size

    def predict_logits(self, X):
        """Return logits (N,C)."""
        preds = []
        for i in range(0, len(X), self.bs):
            batch = X[i:i+self.bs]
            logits = self.model(batch, training=False).numpy()
            preds.append(logits)
        return np.concatenate(preds, axis=0)

# ========= DPU (VART) backend =========
class DpuRunner:
    def __init__(self, xmodel_path, num_threads=1):
        import xir, vart
        self.xir = xir
        self.vart = vart
        g = self.xir.Graph.deserialize(xmodel_path)
        subgraphs = self._get_dpu_subgraphs(g)
        assert len(subgraphs) == 1, "Expected exactly one DPU subgraph."
        self.runners = [self.vart.Runner.create_runner(subgraphs[0], "run") for _ in range(num_threads)]
        self.num_threads = num_threads
        # infer shapes from first runner
        it = self.runners[0].get_input_tensors()[0]
        ot = self.runners[0].get_output_tensors()[0]
        self.input_shape = tuple(it.dims)           # e.g. (batch,H,W,C)
        self.output_shape = tuple(ot.dims)          # e.g. (batch,classes)
        self.batch_size = self.input_shape[0]
        self.out_elems = int(ot.get_data_size() // self.batch_size)

    def _get_dpu_subgraphs(self, graph):
        root = graph.get_root_subgraph()
        if root.is_leaf:
            return []
        childs = root.toposort_child_subgraph()
        return [cs for cs in childs if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"]

    def predict_logits(self, X):
        """Run DPU in batches, returns logits approximated from int8 outputs if necessary."""
        N = len(X)
        bs = self.batch_size
        logits_all = np.zeros((N, self.out_elems), dtype=np.float32)

        idx = 0
        while idx < N:
            run_bs = min(bs, N - idx)
            # prepare input/output buffers (match DPU tensor dtypes)
            in_buf = [np.zeros(self.input_shape, dtype=np.float32, order="C")]
            out_buf = [np.empty(self.output_shape, dtype=np.int8, order="C")]  # many xmodels output int8

            # fill input
            in_buf[0][:run_bs, ...] = X[idx:idx+run_bs].reshape((run_bs,) + self.input_shape[1:])

            # run (single runner for simplicity; could round-robin across self.runners)
            job_id = self.runners[0].execute_async(in_buf, out_buf)
            self.runners[0].wait(job_id)

            # reshape and dequantize heuristic (if int8): scale unknown → use as logits proxy
            out = out_buf[0][:run_bs].reshape(run_bs, self.out_elems).astype(np.float32)
            logits_all[idx:idx+run_bs] = out  # treat int8 scores as logits proxy

            idx += run_bs

        return logits_all

def evaluate_top1(logits, labels):
    if labels is None: 
        return None
    pred = np.argmax(logits, axis=1)
    acc = (pred == labels).mean()
    return float(acc), pred

def main():
    X, y, snr = load_data()
    print(f"Loaded X: {X.shape}  y: {None if y is None else y.shape}  snr: {None if snr is None else snr.shape}")

    t0 = time.time()

    if USE_DPU:
        print("Backend: DPU (VART)")
        runner = DpuRunner(XMODEL_PATH, num_threads=NUM_THREADS)
        logits = runner.predict_logits(X)
    else:
        print("Backend: CPU (Keras)")
        if MODEL_PATH is None or not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Set MODEL_PATH to a valid Keras logits model (.h5 or SavedModel).")
        runner = CpuRunner(MODEL_PATH, batch_size=BATCH_SIZE)
        logits = runner.predict_logits(X)

    probs = softmax_stable(logits)
    t1 = time.time()
    fps = len(X) / (t1 - t0 + 1e-9)

    # Optional accuracy
    acc_info = ""
    acc_val = None
    if y is not None:
        acc_val, preds = evaluate_top1(logits, y)
        acc_info = f" | top-1 acc = {acc_val*100:.2f}%"

    print(f"Done {len(X)} samples in {t1 - t0:.3f}s | FPS = {fps:.2f}{acc_info}")

    # Show a few predictions
    pred_idx = np.argmax(probs, axis=1)
    for i in range(min(5, len(pred_idx))):
        name = MODS[pred_idx[i]] if 0 <= pred_idx[i] < len(MODS) else str(pred_idx[i])
        lbl  = (None if y is None else MODS[y[i]] if 0 <= y[i] < len(MODS) else int(y[i]))
        print(f"sample {i}: pred={name}  true={lbl}")

main()
