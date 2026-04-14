import os
import sys
import time
import tempfile
import subprocess
import cv2
import numpy as np
import torch

sys.path.insert(0, "/src")

from basicsr.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.utils import RealESRGANer
from cog import BasePredictor, Input, Path

WEIGHTS = "/src/weights"


class Predictor(BasePredictor):

    def setup(self):
        t0 = time.time()
        self.half = torch.cuda.is_available()
        print("[setup] GPU:", torch.cuda.get_device_name(0) if self.half else "CPU")

        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_conv=16,
            upscale=4, act_type="prelu"
        )
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=f"{WEIGHTS}/realesr-animevideov3.pth",
            model=model,
            tile=0,        # H100 80GB — tile gerekmez
            tile_pad=0,
            pre_pad=0,
            half=self.half,
        )
        print(f"[setup] Model yüklendi: {time.time()-t0:.1f}s")

    def sharpen(self, img, strength=0.3):
        """Upscale sonrası hafif unsharp mask — blur onarımı."""
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
        return cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)

    def predict(
        self,
        video: Path = Input(description="Upscale edilecek video (MP4, MOV, AVI)"),
        scale: float = Input(description="Ölçek (1-4)", default=4, ge=1, le=4),
        sharpen_strength: float = Input(
            description="Blur onarım gücü (0=kapalı, 0.3=hafif, 0.6=güçlü)",
            default=0.3, ge=0.0, le=1.0
        ),
        denoise: bool = Input(
            description="Gürültü temizleme (eski/gürültülü videolar için)",
            default=False
        ),
    ) -> Path:

        workdir = tempfile.mkdtemp()
        frames_in  = os.path.join(workdir, "frames_in")
        frames_out = os.path.join(workdir, "frames_out")
        os.makedirs(frames_in, exist_ok=True)
        os.makedirs(frames_out, exist_ok=True)

        video_path = str(video)
        out_path   = os.path.join(workdir, "output.mp4")

        # 1. Ses ayır
        audio_path = os.path.join(workdir, "audio.aac")
        has_audio = False
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
            capture_output=True, text=True
        )
        if "audio" in probe.stdout:
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "copy", audio_path
            ], check=True)
            has_audio = True

        # 2. FPS al
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"[predict] {total} kare, {fps:.2f} FPS")

        # 3. Kareleri çıkar — maksimum kalite PNG
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-qscale:v", "1", "-qmin", "1",
            f"{frames_in}/%08d.png"
        ], check=True)

        # 4. Her kareyi işle
        frame_files = sorted(os.listdir(frames_in))
        t_start = time.time()

        for i, fname in enumerate(frame_files):
            if i % 50 == 0:
                elapsed = time.time() - t_start
                eta = (elapsed / max(i, 1)) * (len(frame_files) - i)
                print(f"[predict] Kare {i+1}/{len(frame_files)} — ETA: {eta:.0f}s")

            img = cv2.imread(os.path.join(frames_in, fname), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            # Opsiyonel: upscale öncesi denoise
            if denoise:
                img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)

            # Upscale
            try:
                output, _ = self.upsampler.enhance(img, outscale=scale)
            except RuntimeError as e:
                print(f"CUDA hatası kare {fname}: {e}")
                raise

            # Blur onarımı — unsharp mask
            if sharpen_strength > 0:
                output = self.sharpen(output, strength=sharpen_strength)

            cv2.imwrite(os.path.join(frames_out, fname), output)

        print(f"[predict] İşleme tamamlandı: {time.time()-t_start:.1f}s")

        # 5. Kareleri videoya birleştir — H264 en yüksek kalite
        video_noaudio = os.path.join(workdir, "video_noaudio.mp4")
        subprocess.run([
            "ffmpeg", "-y",
            "-r", str(fps),
            "-i", f"{frames_out}/%08d.png",
            "-c:v", "libx264",
            "-crf", "16",          # 0=lossless, daha düşük=daha kaliteli
            "-preset", "fast",     # slow yerine fast — H100'de encoding bottleneck değil
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            video_noaudio
        ], check=True)

        # 6. Sesi geri ekle
        if has_audio:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", video_noaudio,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                out_path
            ], check=True)
        else:
            os.rename(video_noaudio, out_path)

        print(f"[predict] Tamamlandı: {out_path}")
        return Path(out_path)