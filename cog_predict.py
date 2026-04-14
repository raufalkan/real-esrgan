import os
import sys
import time
import tempfile
import subprocess
import threading
import queue
import cv2
import numpy as np
import torch

sys.path.insert(0, "/src")

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.utils import RealESRGANer
from cog import BasePredictor, Input, Path

WEIGHTS = "/src/weights"


class Predictor(BasePredictor):

    def setup(self):
        t0 = time.time()
        self.half = torch.cuda.is_available()
        print("[setup] GPU:", torch.cuda.get_device_name(0) if self.half else "CPU")

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23, num_grow_ch=32,
            scale=4
        )
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=f"{WEIGHTS}/RealESRGAN_x4plus.pth",
            model=model,
            tile=512,       # OOM önleme; tile=0'dan minimal overhead
            tile_pad=32,
            pre_pad=0,
            half=self.half,
        )
        print(f"[setup] Model yüklendi: {time.time()-t0:.1f}s")

    def analyze_video(self, video_path):
        """Videoyu analiz et, en iyi parametreleri otomatik belirle."""
        cap = cv2.VideoCapture(video_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 24
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        blur_scores = []
        sample_positions = [int(total * p) for p in [0.2, 0.35, 0.5, 0.65, 0.8]]

        for pos in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())

        cap.release()

        avg_blur = np.mean(blur_scores) if blur_scores else 100

        # Scale: 1080p+ için de 2x uygula (kalite için)
        if width <= 720:
            scale = 4
        else:
            scale = 2

        # Sharpen: sadece gerçekten bulanık videolara uygula
        # RealESRGAN zaten noise temizler, net videoya ekleme
        if avg_blur < 50:
            sharpen = 0.5
        elif avg_blur < 150:
            sharpen = 0.25
        else:
            sharpen = 0.0

        print(f"[analyze] {width}x{height} @ {fps:.1f}fps, {total} kare")
        print(f"[analyze] Blur={avg_blur:.1f} → scale={scale}, sharpen={sharpen}")

        return scale, sharpen, fps, total, width, height

    def sharpen_frame(self, img, strength):
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
        return cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)

    def predict(
        self,
        video: Path = Input(description="Upscale edilecek video (MP4, MOV, AVI)"),
    ) -> Path:

        video_path = str(video)
        workdir    = tempfile.mkdtemp()
        out_path   = os.path.join(workdir, "output.mp4")

        # 1. Video analiz
        scale, sharpen, fps, total, width, height = self.analyze_video(video_path)
        self.upsampler.scale = scale

        out_w = width * scale
        out_h = height * scale

        # 2. Ses var mı kontrol et
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
            capture_output=True, text=True
        )
        has_audio = "audio" in probe.stdout

        # 3. ffmpeg reader — pipe üzerinden ham kare akışı (disk I/O yok)
        reader_cmd = [
            "ffmpeg", "-i", video_path,
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-loglevel", "error",
            "pipe:1"
        ]
        reader_proc = subprocess.Popen(reader_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        # 4. ffmpeg writer — işlenmiş kareleri doğrudan encode et
        writer_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{out_w}x{out_h}",
            "-r", str(fps),
            "-i", "pipe:0",
        ]
        if has_audio:
            # Ses dosyası ayırmadan doğrudan kaynaktan mux
            writer_cmd += ["-i", video_path, "-map", "0:v", "-map", "1:a", "-c:a", "copy"]
        writer_cmd += [
            "-c:v", "libx264",
            "-crf", "16",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            out_path
        ]
        writer_proc = subprocess.Popen(writer_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        # 5. Arka plan okuma thread'i — GPU işlerken sonraki kare hazır
        frame_bytes = width * height * 3
        read_q = queue.Queue(maxsize=8)

        def reader_thread():
            for _ in range(total):
                raw = reader_proc.stdout.read(frame_bytes)
                if not raw or len(raw) < frame_bytes:
                    break
                img = np.frombuffer(raw, np.uint8).reshape([height, width, 3])
                read_q.put(img)
            read_q.put(None)  # sentinel

        t = threading.Thread(target=reader_thread, daemon=True)
        t.start()

        # 6. GPU işleme döngüsü
        t_start = time.time()
        processed = 0

        while True:
            img = read_q.get()
            if img is None:
                break

            if processed % 100 == 0 and processed > 0:
                elapsed = time.time() - t_start
                fps_proc = processed / elapsed
                eta = (total - processed) / max(fps_proc, 0.001)
                print(f"[predict] {processed}/{total} kare — {fps_proc:.2f} fps — ETA: {eta:.0f}s")

            try:
                output, _ = self.upsampler.enhance(img, outscale=scale)
            except RuntimeError as e:
                print(f"CUDA hatası kare {processed}: {e}")
                raise

            if sharpen > 0:
                output = self.sharpen_frame(output, strength=sharpen)

            writer_proc.stdin.write(output.astype(np.uint8).tobytes())
            processed += 1

        # 7. Temizlik
        reader_proc.stdout.close()
        reader_proc.wait()
        writer_proc.stdin.close()
        writer_proc.wait()
        t.join()

        elapsed = time.time() - t_start
        print(f"[predict] Tamamlandı: {processed} kare, {elapsed:.1f}s ({processed/elapsed:.2f} fps)")
        return Path(out_path)
