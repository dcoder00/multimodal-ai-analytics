import argparse
import time
import sys
from pathlib import Path

import numpy as np
import cv2

# Try to import CuPy (optional)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False


def has_opencv_cuda():
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


def build_cupy_brightness_contrast_kernel():
    """
    Returns a compiled CuPy RawKernel for brightness/contrast on uint8 grayscale images:
        out[x] = clip(alpha * inp[x] + beta, 0, 255)
    """
    if not CUPY_AVAILABLE:
        return None

    src = r'''
    extern "C" __global__
    void adjust_bc_u8(const unsigned char* __restrict__ inp,
                      unsigned char* __restrict__ out,
                      const float alpha,
                      const float beta,
                      const int n) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n) {
            float v = alpha * (float)inp[i] + beta;
            v = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
            out[i] = (unsigned char)(v + 0.5f);
        }
    }
    '''
    try:
        module = cp.RawModule(code=src, backend='nvcc')
        return module.get_function('adjust_bc_u8')
    except Exception as e:
        print(f"[warn] Failed to compile CuPy kernel: {e}")
        return None


def brightness_contrast_gpu_u8(gray_gpu_mat, alpha=1.2, beta=10, bc_kernel=None):
    """
    Brightness/Contrast on GPU using either:
      - OpenCV CUDA (if available: convertScaleAbs)
      - CuPy custom kernel (if available)
    Inputs:
      gray_gpu_mat: cv2.cuda_GpuMat (single-channel uint8)
    Returns:
      cv2.cuda_GpuMat
    """
    # Prefer OpenCV CUDA if present
    try:
        # convertScaleAbs is CPU in classic OpenCV; in CUDA we mimic via custom kernel or use cv2.cuda.multiply/add
        # Approach: convert to float32 on GPU, scale, add, clip, convert back
        gray_f = cv2.cuda.cvtColor(gray_gpu_mat, cv2.COLOR_GRAY2BGR)  # trick to use some ops if needed
    except cv2.error:
        gray_f = None

    # Strategy 1: Use CuPy kernel directly on the underlying data buffer
    if CUPY_AVAILABLE and bc_kernel is not None:
        # Download pointer-free path: wrap memory via download/upload is simplest & still GPU->GPU with CuPy
        # Convert GpuMat to CuPy without CPU hop is non-trivial; simplest robust path is .download() then re-upload
        # But we want to stay on GPU; OpenCV Python API doesn’t expose device ptr safely, so do a minimal hop.
        arr = gray_gpu_mat.download()
        d_in = cp.asarray(arr)  # HxW on device
        d_out = cp.empty_like(d_in)
        n = d_in.size
        threads = 256
        blocks = (n + threads - 1) // threads
        bc_kernel((blocks,), (threads,), (d_in.ravel(), d_out.ravel(), np.float32(alpha), np.float32(beta), np.int32(n)))
        out_arr = cp.asnumpy(d_out)  # pull back to host
        out_gpu = cv2.cuda_GpuMat()
        out_gpu.upload(out_arr)
        return out_gpu

    # Strategy 2: Emulate brightness/contrast via CUDA-friendly ops using CPU fallback if needed
    # We’ll implement a simple CPU fallback (fast enough for control), then re-upload.
    arr = gray_gpu_mat.download()
    out = cv2.convertScaleAbs(arr, alpha=alpha, beta=beta)
    out_gpu = cv2.cuda_GpuMat()
    out_gpu.upload(out)
    return out_gpu


def gaussian_blur_gpu(gray_gpu_mat, ksize=(5,5), sigma=0):
    # CUDA Gaussian filter
    try:
        gauss = cv2.cuda.createGaussianFilter(gray_gpu_mat.type(), gray_gpu_mat.type(), ksize, sigma)
        return gauss.apply(gray_gpu_mat)
    except cv2.error:
        # fallback to CPU
        arr = gray_gpu_mat.download()
        out = cv2.GaussianBlur(arr, ksize, sigma)
        out_gpu = cv2.cuda_GpuMat()
        out_gpu.upload(out)
        return out_gpu


def clahe_gpu(gray_gpu_mat, clip_limit=2.0, tile_grid_size=(8,8)):
    # CUDA CLAHE available as cv2.cuda.createCLAHE in contrib builds
    try:
        clahe = cv2.cuda.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(gray_gpu_mat)
    except cv2.error:
        # CPU fallback
        arr = gray_gpu_mat.download()
        clahe_cpu = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        out = clahe_cpu.apply(arr)
        out_gpu = cv2.cuda_GpuMat()
        out_gpu.upload(out)
        return out_gpu


def preprocess_frame_gpu(frame_bgr, bc_kernel, alpha, beta, blur_ksize, blur_sigma, clahe_clip, clahe_tile):
    # Upload to GPU
    gpu = cv2.cuda_GpuMat()
    gpu.upload(frame_bgr)

    # Convert to gray
    gray = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2GRAY)

    # Brightness/Contrast (GPU/CuPy, fallback CPU)
    gray = brightness_contrast_gpu_u8(gray, alpha=alpha, beta=beta, bc_kernel=bc_kernel)

    # Gaussian blur (GPU, fallback)
    gray = gaussian_blur_gpu(gray, ksize=blur_ksize, sigma=blur_sigma)

    # CLAHE (GPU, fallback)
    gray = clahe_gpu(gray, clip_limit=clahe_clip, tile_grid_size=clahe_tile)

    # Convert back to BGR for writing
    out_gpu = cv2.cuda.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return out_gpu


def preprocess_frame_cpu(frame_bgr, alpha, beta, blur_ksize, blur_sigma, clahe_clip, clahe_tile):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    gray = cv2.GaussianBlur(gray, ksize=blur_ksize, sigmaX=blur_sigma)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    gray = clahe.apply(gray)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return out


def main():
    parser = argparse.ArgumentParser(description="GPU-Accelerated Visual Preprocessing Pipeline")
    parser.add_argument("--input", type=str, help="Path to input video")
    parser.add_argument("--webcam", type=int, help="Webcam index (e.g., 0)", default=None)
    parser.add_argument("--out", type=str, default="output.avi", help="Output video path")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS if input metadata missing")
    parser.add_argument("--alpha", type=float, default=1.2, help="Contrast multiplier")
    parser.add_argument("--beta", type=float, default=10.0, help="Brightness addend")
    parser.add_argument("--blur", type=int, default=5, help="Gaussian kernel size (odd int)")
    parser.add_argument("--sigma", type=float, default=0.0, help="Gaussian sigma")
    parser.add_argument("--clahe", type=float, default=2.0, help="CLAHE clip limit")
    parser.add_argument("--tile", type=int, default=8, help="CLAHE tile grid size (tile x tile)")
    parser.add_argument("--max_frames", type=int, default=0, help="Process only first N frames (0=all)")
    parser.add_argument("--benchmark", action="store_true", help="Print CPU vs GPU throughput")
    args = parser.parse_args()

    if (args.input is None) == (args.webcam is None):
        print("Choose exactly one: --input <video> OR --webcam <index>")
        sys.exit(1)

    cap = cv2.VideoCapture(args.input) if args.input else cv2.VideoCapture(args.webcam)
    if not cap.isOpened():
        print("Failed to open video source.")
        sys.exit(1)

    # Get frame properties
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = src_fps if src_fps and src_fps > 1 else args.fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.out, fourcc, out_fps, (w, h))
    if not out.isOpened():
        print("Failed to open output writer.")
        sys.exit(1)

    # CUDA / CuPy availability
    cuda_ok = has_opencv_cuda()
    if cuda_ok:
        cv2.cuda.setDevice(0)
    bc_kernel = build_cupy_brightness_contrast_kernel() if CUPY_AVAILABLE and cuda_ok else None

    print(f"[info] OpenCV CUDA: {'YES' if cuda_ok else 'NO'} | CuPy: {'YES' if CUPY_AVAILABLE else 'NO'} | BC kernel: {'YES' if bc_kernel else 'NO'}")
    print(f"[info] Processing... (alpha={args.alpha}, beta={args.beta}, blur={args.blur}, sigma={args.sigma}, clahe={args.clahe}, tile={args.tile}x{args.tile})")

    # Benchmark accumulators
    gpu_frames = cpu_frames = 0
    gpu_t0 = gpu_t1 = cpu_t0 = cpu_t1 = 0.0

    frame_count = 0
    blur_ksize = (max(1, args.blur) | 1, max(1, args.blur) | 1)  # ensure odd
    tile_sz = (args.tile, args.tile)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if cuda_ok:
            t0 = time.perf_counter()
            try:
                out_gpu = preprocess_frame_gpu(frame, bc_kernel, args.alpha, args.beta, blur_ksize, args.sigma, args.clahe, tile_sz)
                processed = out_gpu.download()
            except cv2.error:
                # any unexpected issue -> fallback to CPU for this frame
                processed = preprocess_frame_cpu(frame, args.alpha, args.beta, blur_ksize, args.sigma, args.clahe, tile_sz)
            t1 = time.perf_counter()
            gpu_t0 += t1 - t0
            gpu_frames += 1
        else:
            t0 = time.perf_counter()
            processed = preprocess_frame_cpu(frame, args.alpha, args.beta, blur_ksize, args.sigma, args.clahe, tile_sz)
            t1 = time.perf_counter()
            cpu_t0 += t1 - t0
            cpu_frames += 1

        out.write(processed)

        if args.max_frames and frame_count >= args.max_frames:
            break

    cap.release()
    out.release()

    if args.benchmark:
        if gpu_frames:
            print(f"[benchmark] GPU frames: {gpu_frames}, avg/frame: {gpu_t0/gpu_frames*1000:.2f} ms, FPS: {gpu_frames/gpu_t0:.2f}")
        if cpu_frames:
            print(f"[benchmark] CPU frames: {cpu_frames}, avg/frame: {cpu_t0/cpu_frames*1000:.2f} ms, FPS: {cpu_frames/cpu_t0:.2f}")
    print(f"[done] Wrote: {args.out}")


if __name__ == "__main__":
    main()
