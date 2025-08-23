# Colab-friendly RF classifier runner: VART if available, else TF/Keras CPU.
import os, math, sys, threading, numpy as np

# ---------------- Softmax (stable) ----------------
def softmax_stable(x, scale=1.0):
    x = np.asarray(x, dtype=np.float32) * float(scale)
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)

# ---------------- Data ----------------
RF_INPUT_PATH   = "/workspace/rf_input.npy"
RF_CLASSES_PATH = "/workspace/rf_classes.npy"  # int labels OR one-hot; both supported
RF_SNRS_PATH    = "/workspace/rf_snrs.npy"

X = np.load(RF_INPUT_PATH)        # (N, 1024, 1, 2) float32
Y = np.load(RF_CLASSES_PATH)      # (N,) int OR (N,10) one-hot
Z = np.load(RF_SNRS_PATH)         # (N,) float (SNR dB)
N, H, W, C = X.shape
assert (H, W, C) == (1024, 1, 2), f"Unexpected input shape: {X.shape}"

# Convert labels to integer indices if one-hot
if Y.ndim == 2 and Y.shape[1] > 1:
    y_idx = np.argmax(Y, axis=1).astype(np.int64)
else:
    y_idx = Y.astype(np.int64)

class_names = ['BPSK','QPSK','GMSK','FM','OOK','OQPSK','8PSK','16QAM','AM-SSB-WC','AM-DSB-SC']
num_classes = len(class_names)

# ---------------- Try VART (DPU) ----------------
USE_VART = False
try:
    import xir, vart  # only present in Vitis-AI containers
    USE_VART = True
except Exception:
    USE_VART = False

# ---------------- VART path ----------------
def get_dpu_subgraph(graph):
    root = graph.get_root_subgraph()
    assert root is not None
    subs = root.toposort_child_subgraph()
    dpu_subs = [s for s in subs if s.has_attr("device") and s.get_attr("device").upper()=="DPU"]
    assert len(dpu_subs) == 1, "Expected exactly 1 DPU subgraph."
    return dpu_subs[0]

def run_vart(xmodel_path, max_samples=None, verbose_every=1):
    g = xir.Graph.deserialize(xmodel_path)
    dpu_sub = get_dpu_subgraph(g)
    runner = vart.Runner.create_runner(dpu_sub, "run")

    in_t = runner.get_input_tensors()[0]
    out_t = runner.get_output_tensors()[0]
    in_dims  = tuple(in_t.dims)          # e.g., (B,1024,1,2)
    out_dims = tuple(out_t.dims)         # e.g., (B,10)
    batch    = in_dims[0]
    out_fix  = out_t.get_attr("fix_point") if out_t.has_attr("fix_point") else 0
    out_scale = 1.0 / (2**out_fix) if out_fix is not None else 1.0
    out_size  = int(out_t.get_data_size() / batch)

    total = min(N, max_samples) if max_samples else N
    top1 = 0
    count = 0
    while count < total:
        run_size = min(batch, total - count)
        in_buf  = [np.empty(in_dims,  dtype=np.float32, order="C")]
        out_buf = [np.empty(out_dims, dtype=np.int8,   order="C")]

        for j in range(run_size):
            in_buf[0][j, ...] = X[count + j]

        jid = runner.execute_async(in_buf, out_buf)
        runner.wait(jid)

        logits_q = out_buf[0].reshape(run_size, out_size)  # int8 logits
        probs = softmax_stable(logits_q, scale=out_scale)  # dequantize+softmax
        pred = np.argmax(probs, axis=1)

        for j in range(run_size):
            i   = count + j
            ok  = int(pred[j] == y_idx[i])
            top1 += ok
            if (i % verbose_every) == 0:
                print(f"{i}: SNR={Z[i]:.1f} dB  Top1={class_names[pred[j]]:>10s} "
                      f"{probs[j,pred[j]]:5.2f}  Actual={class_names[y_idx[i]]}")

        count += run_size

    acc = top1 / float(total)
    print(f"\n[DPU] Samples: {total}  Batch: {batch}  Top1: {acc:.4f}")

# ---------------- TF/Keras CPU path ----------------
def run_tf(keras_model_path, max_samples=None, batch=256, verbose_every=1):
    import tensorflow as tf
    model = tf.keras.models.load_model(keras_model_path, compile=False)
    # Expecting logits (no softmax). If your model ends with softmax, it still works;
    # we’ll detect it by shape and apply softmax only if needed.
    total = min(N, max_samples) if max_samples else N
    top1 = 0
    count = 0
    while count < total:
        run_size = min(batch, total - count)
        xb = X[count:count+run_size]
        logits = model(xb, training=False).numpy()
        # If already probabilities, clip & renorm; else compute softmax.
        if np.all((logits >= 0) & (logits <= 1)) and np.allclose(np.sum(logits, axis=1), 1, atol=1e-3):
            probs = logits
        else:
            probs = softmax_stable(logits, scale=1.0)
        pred = np.argmax(probs, axis=1)

        for j in range(run_size):
            i   = count + j
            ok  = int(pred[j] == y_idx[i])
            top1 += ok
            if (i % verbose_every) == 0:
                print(f"{i}: SNR={Z[i]:.1f} dB  Top1={class_names[pred[j]]:>10s} "
                      f"{probs[j,pred[j]]:5.2f}  Actual={class_names[y_idx[i]]}")
        count += run_size

    acc = top1 / float(total)
    print(f"\n[CPU/TF] Samples: {total}  Batch: {batch}  Top1: {acc:.4f}")

# ---------------- Choose backend & run ----------------
# If you’re in a Vitis-AI container with VART, set your compiled xmodel here:
XMODEL_PATH = "/workspace/vai_c_output/rf_F32_t1.xmodel"   # change if applicable

# If no VART, set your TF/Keras logits model path:
KERAS_MODEL_PATH = "/workspace/SRNET_Q_8IN_T1.h5"  # change to your .h5/.keras

if USE_VART and os.path.exists(XMODEL_PATH):
    run_vart(XMODEL_PATH, max_samples=None, verbose_every=50)
else:
    print("[Info] VART not available or xmodel not found -> using TF/Keras CPU.")
    assert os.path.exists(KERAS_MODEL_PATH), f"Model not found: {KERAS_MODEL_PATH}"
    run_tf(KERAS_MODEL_PATH, max_samples=None, batch=512, verbose_every=50)
