"""
C1（正弦信号的抽样）

(1) 连续信号 x1(t) = sin(2π f t)，f = 10 Hz。
    采样频率依次取 fs ∈ {40, 20, 30, 50} Hz，对时间范围 t ∈ [0, 0.2] s：
    - 在同一张图中绘制连续波形（plot）与离散样本（stem）。
    - 图像会保存到 outputs/ 下。

(2) 选做：钢琴中音“1”（C4，Do）频率约 f = 261.63 ≈ 262 Hz。
    采样频率 fs = 8000 Hz，产生 1 s 波形并播放；另外演示一小段乐音（C大调简谱 1-2-3-5-3-2-1）。
    - 播放依赖 simpleaudio；若缺失，将自动仅保存 WAV 文件到 outputs/。

运行
----
直接运行本脚本即可在 outputs/ 目录下生成图片与音频：
  - sampling_*.png：四个采样频率的对比图
  - tone_do.wav / melody_demo.wav：单音与旋律音频

可选参数（命令行）：
  --no-sound  仅生成文件，不播放声音
  --no-show   不弹出绘图窗口（默认保存 PNG）
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Union

import numpy as np
import matplotlib
# 自动选择非交互后端以避免无显示环境报错（必要时仍可 plt.show()）
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    import simpleaudio as sa  # 轻量级音频播放
    HAVE_SA = True
except Exception:
    sa = None
    HAVE_SA = False

# Windows 备选：winsound 播放 WAV 文件
try:
    import winsound  # type: ignore
    HAVE_WINSOUND = True
except Exception:
    winsound = None  # type: ignore
    HAVE_WINSOUND = False

import wave

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


def save_fig(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(IMAGE_DIR, filename)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def write_wav_mono_16k(filename: str, x: np.ndarray, fs: int) -> str:
    """将 -1..1 浮点单声道写入 16-bit PCM WAV。返回文件路径。"""
    x = np.asarray(x, dtype=np.float64)
    # 峰值归一化，避免超过 [-1,1]
    peak = np.max(np.abs(x)) or 1.0
    y = np.int16(np.clip(x / peak, -1, 1) * 32767)
    path = os.path.join(AUDIO_DIR, filename)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(fs))
        wf.writeframes(y.tobytes())
    print(f"[saved] {path}")
    return path


def resample_linear(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """保留的简易重采样函数（当前默认直接生成 44.1k 音频，通常无需调用）。"""
    if fs_in == fs_out:
        return np.asarray(x, dtype=float)
    t_in = np.arange(len(x)) / fs_in
    t_out = np.arange(int(round(len(x) * fs_out / fs_in))) / fs_out
    return np.interp(t_out, t_in, x).astype(float)


def maybe_play(x: np.ndarray, fs: int, *, enable_sound: bool = True, wav_path: str | None = None) -> None:
    if not enable_sound:
        return
    if HAVE_SA:
        audio = np.int16(np.clip(x, -1, 1) * 32767)
        try:
            sa.play_buffer(audio, 1, 2, fs)
        except Exception as e:
            print(f"[warn] simpleaudio 播放失败：{e}")
    elif HAVE_WINSOUND and wav_path:
        try:
            # 异步播放，不阻塞
            winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            print(f"[warn] winsound 播放失败：{e}")
    else:
        print("[info] 未安装 simpleaudio，跳过播放（已保存 WAV 文件）。建议 pip install simpleaudio")


@dataclass
class SineSpec:
    f: float  # 频率 (Hz)
    fs: float  # 采样率 (Hz)
    duration: float  # 时长 (s)


def generate_continuous_sine(f: float, t_end: float, oversample: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """生成“准连续”的参考波形（用高过采样率离散化）。"""
    t = np.linspace(0, t_end, int(t_end * oversample) + 1, endpoint=True)
    x = np.sin(2 * np.pi * f * t)
    return t, x


def sample_sine(f: float, fs: float, t_end: float) -> Tuple[np.ndarray, np.ndarray]:
    n = np.arange(0, int(np.floor(t_end * fs)) + 1)
    ts = n / fs
    x = np.sin(2 * np.pi * f * ts)
    return ts, x


def plot_sampling(f: float, fs_list: Iterable[float], t_end: float = 0.2, show: bool = False) -> None:
    t_cont, x_cont = generate_continuous_sine(f, t_end)
    for fs in fs_list:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(t_cont, x_cont, 'C0-', lw=2, label=f"连续: f={f}Hz")
        ts, xs = sample_sine(f, fs, t_end)
        markerline, stemlines, baseline = ax.stem(ts, xs, basefmt=' ', linefmt='C1-', markerfmt='C1o', label=f"样本: fs={fs}Hz")
        plt.setp(stemlines, linewidth=1.2)
        ax.set_xlim(0, t_end)
        ax.set_xlabel("时间 t (s)")
        ax.set_ylabel("幅度")
        ax.grid(True, ls='--', alpha=0.5)
        ax.set_title(f"正弦抽样对比 f={f} Hz, fs={fs} Hz")
        ax.legend(loc='upper right')
        save_fig(fig, f"sampling_f{f}_fs{int(fs)}.png")
        if show:
            plt.show()


def tone(f: float, fs: int, duration: float, amplitude: float = 0.8) -> np.ndarray:
    t = np.arange(int(fs * duration)) / fs
    x = amplitude * np.sin(2 * np.pi * f * t)
    # 5 ms 线性起落，避免爆音
    fade = int(0.005 * fs)
    if fade > 0:
        env = np.ones_like(x)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        x = x * env
    return x


# 钢琴十二平均律：以 A4=440Hz，C4（中音1/Do）= 261.63Hz
PIANO_KEY_FREQ = {
    # C大调音阶 C4..C5（简谱 1..7..1）
    'C4': 261.63,  # 1 (Do)
    'D4': 293.66,  # 2 (Re)
    'E4': 329.63,  # 3 (Mi)
    'F4': 349.23,  # 4 (Fa)
    'G4': 392.00,  # 5 (So)
    'A4': 440.00,  # 6 (La)
    'B4': 493.88,  # 7 (Ti)
    'C5': 523.25,  # 1 (Do)
}

SOLFEGE_TO_KEY = {
    '1': 'C4', '2': 'D4', '3': 'E4', '4': 'F4', '5': 'G4', '6': 'A4', '7': 'B4', '1^': 'C5'
}


Token = Union[str, Tuple[str, float]]  # 如 '1' 或 ('5', 2.0) 表示持续 2 拍


def melody_from_solfege(score: List[Token], beat: float, fs: int) -> np.ndarray:
    """根据简谱合成旋律。
    - score: 由音符字符串或 (音符, 倍数) 组成的列表；支持休止符 'r'/'0'/'-'.
    - beat: 基本音长（秒）。
    """
    frames: List[np.ndarray] = []
    for tok in score:
        if isinstance(tok, tuple):
            s, mult = tok[0], float(tok[1])
        else:
            s, mult = tok, 1.0

        if s in {"r", "0", "-"}:  # 休止
            frames.append(np.zeros(int(round(fs * beat * mult))))
            continue

        key = SOLFEGE_TO_KEY[s]
        f = PIANO_KEY_FREQ[key]
        frames.append(tone(f, fs, beat * mult))

    return np.concatenate(frames) if frames else np.zeros(0, dtype=float)


def main(show: bool = False, enable_sound: bool = True):
    # 优先选择系统常见中文字体，避免标题/标签出现方块
    preferred_fonts = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "Noto Sans CJK SC"]
    for name in preferred_fonts:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False

    # 题(1)：采样演示
    f = 10.0
    fs_list = [40, 20, 30, 50]
    plot_sampling(f, fs_list, t_end=0.2, show=show)

    # 题(2)：单音 + 旋律
    f_do = 261.63  # ≈262Hz
    fs = 44100  # 直接以 44.1 kHz 生成，保证兼容性

    x_do = tone(f_do, fs, 1.0)
    path_do = write_wav_mono_16k("tone_do.wav", x_do, fs)
    maybe_play(x_do, fs, enable_sound=enable_sound, wav_path=path_do)

    # 一闪一闪亮晶晶（前半句）：1 1 5 5 6 6 5(二拍)
    # 完整“一闪一闪亮晶晶”前两句：
    # 上半句：1 1 5 5 6 6 5(二拍)
    # 下半句：4 4 3 3 2 2 1(二拍)
    twinkle_score: List[Token] = [
        '1','1','5','5','6','6',('5',2.0),
        '4','4','3','3','2','2',('1',2.0)
    ]
    # 每拍 0.5 s，总长约 7 s
    mel = melody_from_solfege(twinkle_score, beat=0.5, fs=fs)
    path_mel = write_wav_mono_16k("twinkle_demo.wav", mel, fs)
    maybe_play(mel, fs, enable_sound=enable_sound, wav_path=path_mel)

    # 绘制单音波形（上：0~0.2s 更清晰；下：前 15ms 细节）
    t = np.arange(len(x_do)) / fs
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharey=True)
    axes[0].plot(t, x_do, 'C0-', lw=0.8)
    axes[0].set_title(f"中音 Do 单音波形 f≈{f_do:.2f}Hz, fs={fs}Hz, 1s")
    axes[0].set_xlim(0, 0.2)
    axes[0].grid(True, ls='--', alpha=0.5)

    axes[1].plot(t, x_do, 'C0-', lw=1.2)
    axes[1].set_xlim(0, 0.015)
    axes[1].set_xlabel("时间 t (s)")
    axes[1].grid(True, ls='--', alpha=0.5)

    save_fig(fig, "tone_do_waveform.png")
    if show:
        plt.show()

    # 额外：为旋律生成波形图（上：全长；下：前 80 ms 细节）
    def plot_wave_sections(x: np.ndarray, fs: int, title: str, fname: str,
                           sections: List[Tuple[float, float]]):
        t = np.arange(len(x)) / fs
        # 为提升可读性，仅绘制至多 ~5000 个点
        step = max(1, len(x) // 5000)
        t_ds, x_ds = t[::step], x[::step]
        fig, axes = plt.subplots(len(sections), 1, figsize=(9, 2.0 * len(sections) + 1.0), sharex=False)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        for ax, (a, b) in zip(axes, sections):
            ax.plot(t_ds, x_ds, 'C0-', lw=0.9)
            ax.set_xlim(max(0, a), min(t[-1], b))
            ax.grid(True, ls='--', alpha=0.5)
        axes[0].set_title(title)
        axes[-1].set_xlabel("时间 t (s)")
        save_fig(fig, fname)

    # 把 2 秒旋律切成三段显示，避免太密难以分辨
    total = len(mel)/fs
    # 1 秒为一个窗口，6 个分段展示
    sections = [(i, min(i+1.0, total)) for i in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] if i < total]
    plot_wave_sections(
        mel, fs, "一闪一闪亮晶晶（完整前两句）波形（分段视图）",
        "twinkle_waveform.png",
        sections=sections
    )

    # 额外：为 do..xi（C4..B4）分别绘制 1s 波形图
    note_map = {
        'do(C4)': 'C4', 're(D4)': 'D4', 'mi(E4)': 'E4', 'fa(F4)': 'F4',
        'so(G4)': 'G4', 'la(A4)': 'A4', 'xi(B4)': 'B4'
    }
    for label, key in note_map.items():
        f_note = PIANO_KEY_FREQ[key]
        x = tone(f_note, fs, 1.0)
        t = np.arange(len(x)) / fs
        fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharey=True)
        axes[0].plot(t, x, 'C0-', lw=0.8)
        axes[0].set_title(f"{label} 单音波形 f={f_note:.2f}Hz, fs={fs}Hz")
        axes[0].set_xlim(0, 0.2)
        axes[0].grid(True, ls='--', alpha=0.5)
        axes[1].plot(t, x, 'C0-', lw=1.2)
        axes[1].set_xlim(0, 0.015)
        axes[1].set_xlabel("时间 t (s)")
        axes[1].grid(True, ls='--', alpha=0.5)
        fname = f"wave_tone_{key}.png"
        save_fig(fig, fname)

    # 将 outputs 根目录下旧的 wav/png 迁移到新子目录（一次性维护，忽略失败）
    try:
        for name in os.listdir(OUTPUT_DIR):
            p = os.path.join(OUTPUT_DIR, name)
            if os.path.isfile(p):
                if name.lower().endswith('.wav'):
                    os.replace(p, os.path.join(AUDIO_DIR, name))
                elif name.lower().endswith('.png'):
                    os.replace(p, os.path.join(IMAGE_DIR, name))
    except Exception as _:
        pass

    # 音频目录清理：只保留最终输出 tone_do.wav 与 twinkle_demo.wav
    try:
        keep = {"tone_do.wav", "twinkle_demo.wav"}
        for name in os.listdir(AUDIO_DIR):
            if name.lower().endswith('.wav') and name not in keep:
                os.remove(os.path.join(AUDIO_DIR, name))
    except Exception:
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-sound", action="store_true", help="不播放声音，只保存 WAV")
    parser.add_argument("--no-show", action="store_true", help="不弹出图形窗口，仅保存 PNG")
    args = parser.parse_args()
    main(show=not args.no_show, enable_sound=not args.no_sound)
