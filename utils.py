import os
import time
import requests
from tqdm import tqdm
import yt_dlp

def download_youtube_video(url: str, out_path: str) -> str:
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "outtmpl": out_path,
        "quiet": False,
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return out_path

def download_video(
    url: str,
    out_path: str,
    timeout: int = 30,
    retries: int = 3,
    chunk_size: int = 8192,
) -> str:
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    if os.path.exists(out_path):
        print(f"Already exists, skipping: {out_path}")
        return out_path
    
    if "youtube.com" in url:
        print(f"Downloading YouTube video: {url}")
        download_youtube_video(url, out_path)
        return out_path

    # Write to a temp file first — if download fails halfway,
    # the corrupt file gets deleted and out_path is never created
    tmp_path = out_path + ".part"

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()

                total = int(response.headers.get("content-length", 0))

                with open(tmp_path, "wb") as f, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=os.path.basename(out_path),
                    leave=True,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

            os.rename(tmp_path, out_path)
            print(f"Saved to {out_path}")
            return out_path

        except (requests.RequestException, OSError) as e:
            last_error = e
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if attempt < retries:
                wait = 2 ** attempt  # exponential backoff: 2s, 4s, 8s
                print(f"Attempt {attempt}/{retries} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(
        f"Failed to download {url} after {retries} attempts. "
        f"Last error: {last_error}"
    ) from last_error