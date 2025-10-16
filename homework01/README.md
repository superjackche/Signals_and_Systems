# 作业 C1：正弦信号的抽样与简易乐音合成

## 依赖
- Python 3.8+
- numpy, matplotlib（详见 `requirements.txt`）
- 可选：simpleaudio（用于播放声音；未安装也会照常生成 WAV 文件）

## 运行
### 用 uv（推荐，极快）
```powershell
# 进入仓库根目录
uv venv .venv ; uv pip install -r homework01\requirements.txt
uv run python homework01\c1_sampling_and_tones.py --no-show --no-sound
```

### 用系统已激活的 venv / pip
```powershell
python homework01\c1_sampling_and_tones.py --no-show --no-sound
```

运行后在 `homework01/outputs/` 下可见（已分目录）：
- images/：`sampling_f10_fs{20,30,40,50}.png`、`tone_do_waveform.png`、`twinkle_waveform.png`、`wave_tone_*.png`
- audio/：`tone_do.wav`、`twinkle_demo.wav`（均为 44.1 kHz，便于通用播放器播放）

若图标题出现“豆腐块”，脚本已内置中文字体优先级（Microsoft YaHei、SimHei 等），一般会自动消失；如依然存在，请在本机安装任一常见中文字体。

说明：示例旋律“一闪一闪亮晶晶”将每拍时长设置为 0.5 s（总长约 4 s），更易清晰试听；可在脚本中修改 `beat` 值来调整时长。

## 你会观察到什么？
- 当采样率 `fs=40 Hz`（大于两倍信号频率 `2f=20 Hz`）时，离散点落在连续曲线上，能无失真重建。
- 当 `fs=20 Hz`（恰等于奈奎斯特率）时，样本点恰好落在两极值附近，波形“看起来”像直线，重建最脆弱。
- 当 `fs=30 Hz` 与 `fs=50 Hz` 时（均 > 2f），无混叠；fs 越高，点越密，重构越稳健。
