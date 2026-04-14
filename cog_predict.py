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
            tile=0,
            tile_pad=0,
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

        # Ortadan 5 kare al, analiz için
        blur_scores = []
        noise_scores = []
        sample_positions = [int(total * p) for p in [0.2, 0.35, 0.5, 0.65, 0.8]]

        for pos in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Blur skoru — düşük = bulanık
            blur_scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())

            # Noise skoru — yüksek = gürültülü
            noise = cv2.meanStdDev(gray - cv2.GaussianBlur(gray, (5,5), 0))[1][0][0]
            noise_scores.append(noise)

        cap.release()

        avg_blur  = np.mean(blur_scores) if blur_scores else 100
        avg_noise = np.mean(noise_scores) if noise_scores else 5

        # Scale belirle
        if width <= 480:
            scale = 4
        elif width <= 720:
            scale = 4
        elif width <= 1080:
            scale = 2
        else:
            scale = 1

        # Sharpen belirle
        if avg_blur < 50:
            sharpen = 0.6    # Çok bulanık
        elif avg_blur < 150:
            sharpen = 0.35   # Orta bulanık
        else:
            sharpen = 0.15   # Zaten net

        # Denoise belirle
        denoise = avg_noise > 8.0

        print(f"[analyze] {width}x{height}, {total} kare, {fps:.1f}fps")
        print(f"[analyze] Blur skoru: {avg_blur:.1f} | Noise skoru: {avg_noise:.2f}")
        print(f"[analyze] Karar → scale={scale}, sharpen={sharpen}, denoise={denoise}")

        return scale, sharpen, denoise, fps, total

    def sharpen_frame(self, img, strength):
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
        return cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)

    def predict(
        self,
        video: Path = Input(description="Upscale edilecek video (MP4, MOV, AVI)"),
    ) -> Path:

        video_path = str(video)
        workdir    = tempfile.mkdtemp()
        frames_in  = os.path.join(workdir, "frames_in")
        frames_out = os.path.join(workdir, "frames_out")
        os.makedirs(frames_in, exist_ok=True)
        os.makedirs(frames_out, exist_ok=True)
        out_path   = os.path.join(workdir, "output.mp4")

        # 1. Video analiz et — parametreleri otomatik belirle
        scale, sharpen, denoise, fps, total = self.analyze_video(video_path)

        # scale değişebileceği için upsampler'ı güncelle
        self.upsampler.scale = scale

        # 2. Ses ayır
        audio_path = os.path.join(workdir, "audio.aac")
        has_audio  = False
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

        # 3. Kareleri çıkar
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

            # Denoise (gerekirse)
            if denoise:
                img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)

            # Upscale
            try:
                output, _ = self.upsampler.enhance(img, outscale=scale)
            except RuntimeError as e:
                print(f"CUDA hatası: {e}")
                raise

            # Sharpen
            if sharpen > 0:
                output = self.sharpen_frame(output, strength=sharpen)

            cv2.imwrite(os.path.join(frames_out, fname), output)

        print(f"[predict] İşleme: {time.time()-t_start:.1f}s")

        # 5. Video birleştir
        video_noaudio = os.path.join(workdir, "video_noaudio.mp4")
        subprocess.run([
            "ffmpeg", "-y",
            "-r", str(fps),
            "-i", f"{frames_out}/%08d.png",
            "-c:v", "libx264",
            "-crf", "16",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            video_noaudio
        ], check=True)

        # 6. Ses ekle
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