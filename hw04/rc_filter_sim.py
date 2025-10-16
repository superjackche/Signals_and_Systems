import numpy as np
import matplotlib
# 使用非交互后端，避免弹窗
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from tabulate import tabulate
from pathlib import Path
import csv

"""RC 低通滤波器仿真 (Python 对应 MATLAB lsim 练习)

说明:
1. 原题 MATLAB 使用: alpha = 2*pi*fc, A=[1 alpha], B=[alpha] 构建一阶系统。
2. 真实电路: alpha = 1/(R*C), 截止频率 fc = alpha/(2*pi)。这里提供两种方式, 通过参数 use_physical 决定。
3. 本脚本: 
    - 复现 MATLAB 单一正弦输入结构, 并扩展批量频率、阶跃、方波、chirp、二阶级联。
    - 自动保存图像与结果文件到新建目录 (默认 rc_figs)。
    - 不在运行中直接给出主观评述, 仅保存数据, 方便根据输出再次判读。
4. 中文显示: 设置中文字体(若系统无相应字体, 会退回英文字体)。
"""

def first_order_rc(alpha: float):
    """生成一阶 RC 低通传递函数 H(s)=alpha/(s+alpha)."""
    return signal.TransferFunction([alpha], [1.0, alpha])


def cascade_second_order(alpha: float):
    """串联两个相同一阶得到二阶: (alpha/(s+alpha))^2."""
    num = [alpha * alpha]
    den = np.convolve([1.0, alpha], [1.0, alpha])  # [1, 2alpha, alpha^2]
    return signal.TransferFunction(num, den)


def sine_response(sys, f, fs, dur):
    """对单频正弦进行时域仿真并估计稳态幅相。"""
    t = np.arange(0, dur, 1/fs)
    x = np.sin(2*np.pi*f*t)
    tout, y, _ = signal.lsim(sys, U=x, T=t)
    steady_index = int(0.2*len(tout))  # 丢弃前 20% 视为暂态
    ts = tout[steady_index:]
    ys = y[steady_index:]
    w = 2*np.pi*f
    ejwt = np.exp(-1j*w*ts)
    A = 2/len(ts) * np.sum(ys * ejwt)
    amp = np.abs(A)
    phase = np.angle(A)
    return tout, x, y, amp, phase


def theoretical_mag_phase(alpha, f):
    w = 2*np.pi*f
    H = alpha / (1j*w + alpha)
    return np.abs(H), np.angle(H)


def main():
    # ========= 基本参数 =========
    fs = 10000        # 采样频率
    dur = 0.1         # 正弦仿真时长 100 ms
    fc_target = 100.0 # 目标截止频率(用于与 MATLAB 对比)
    use_physical = False  # False: alpha=2πfc (与题目 MATLAB 保持) True: 通过 R C 计算
    R = 1e3           # 仅当 use_physical=True 时生效
    C = 1.5915e-6     # 使 fc≈1/(2πRC) ≈ 100Hz

    if use_physical:
        alpha = 1.0 / (R * C)
        fc = alpha / (2*np.pi)
    else:
        fc = fc_target
        alpha = 2 * np.pi * fc  # 题目给出的写法

    # ========= 中文字体设置 =========
    plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','微软雅黑','Heiti TC','Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # ========= 输出目录 =========
    try:
        base_dir = Path(__file__).parent
    except Exception:
        base_dir = Path.cwd()
    print(f'[DEBUG] base_dir={base_dir.resolve()} cwd={Path.cwd().resolve()}')
    out_dir = Path.cwd() / 'rc_figs_out'
    out_dir.mkdir(parents=True, exist_ok=True)
    if not out_dir.exists():
        raise RuntimeError(f'无法创建输出目录: {out_dir}')
    # 写一个测试文件
    with open(out_dir / 'write_test.txt','w',encoding='utf-8') as tf:
        tf.write('write ok')
    print(f'[DEBUG] out_dir contents after create: {list(out_dir.iterdir())}')

    # ========= 单频 (复现 MATLAB 原始结构) =========
    single_freq = 100.0
    sys_single = first_order_rc(alpha)
    t_single = np.arange(0, dur, 1/fs)
    x_single = np.sin(2*np.pi*single_freq*t_single)
    t_single, y_single, _ = signal.lsim(sys_single, U=x_single, T=t_single)
    fig_base, ax_base = plt.subplots(figsize=(7,3))
    ax_base.plot(t_single*1000, x_single, label='输入 x(t)')
    ax_base.plot(t_single*1000, y_single, label='输出 y(t)')
    ax_base.set_title(f'单频正弦输入 (f={single_freq}Hz) 与输出 (fc={fc:.1f}Hz)')
    ax_base.set_xlabel('时间 (ms)')
    ax_base.set_ylabel('电压')
    ax_base.grid(True, ls=':')
    ax_base.legend()
    fig_base.tight_layout()
    fig_base.savefig(out_dir / 'single_frequency_time.png', dpi=160)
    plt.close(fig_base)

    # ========= 多频正弦批量仿真 =========
    freqs = [10, 100, 200, 500, 1000]
    sys1 = sys_single
    rows = []
    store_waveforms = {}
    for f in freqs:
        t, x, y, amp_est, phase_est = sine_response(sys1, f, fs, dur)
        mag_th, phase_th = theoretical_mag_phase(alpha, f)
        rel_err_amp = (amp_est - mag_th)/mag_th if mag_th != 0 else 0.0
        rows.append([f, amp_est, mag_th, rel_err_amp, phase_est, phase_th])
        store_waveforms[f] = (t, x, y)

    # 保存表格 CSV
    csv_path = out_dir / 'amplitude_phase_table.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['f_Hz','Amp_est','Amp_theory','RelErr_Amp','Phase_est_rad','Phase_theory_rad'])
        for r in rows:
            writer.writerow([f'{r[0]:.6f}', f'{r[1]:.8f}', f'{r[2]:.8f}', f'{r[3]:.6e}', f'{r[4]:.8f}', f'{r[5]:.8f}'])

    # 也保存为整齐文本
    table_txt = tabulate(rows, headers=['f(Hz)','Amp_est','Amp_theory','RelErr_Amp','Phase_est(rad)','Phase_theory(rad)'], floatfmt='.6f')
    with open(out_dir / 'amplitude_phase_table.txt','w',encoding='utf-8') as ft:
        ft.write('一阶系统幅频/相频结果\n')
        ft.write(f'采用 {"alpha=2πfc" if not use_physical else "alpha=1/RC"}, fc≈{fc:.3f}Hz\n\n')
        ft.write(table_txt)
        ft.write('\n')

    # 绘制各频率输入输出
    fig, axes = plt.subplots(len(freqs), 1, figsize=(8, 2*len(freqs)), sharex=True)
    for ax, f in zip(axes, freqs):
        t, x, y = store_waveforms[f]
        ax.plot(t*1000, x, label='输入')
        ax.plot(t*1000, y, label='输出')
        ax.set_ylabel(f'{f}Hz')
        ax.grid(True, ls=':')
    axes[-1].set_xlabel('时间 (ms)')
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle('不同频率正弦输入输出')
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(out_dir / 'multi_frequency_time.png', dpi=160)
    plt.close(fig)

    # ========= 阶跃 / 方波 / Chirp =========
    t_ext = np.linspace(0, 0.02, int(fs*0.02))  # 20 ms
    step_in = np.ones_like(t_ext)
    square_in = signal.square(2*np.pi*500*t_ext)
    chirp_in = signal.chirp(t_ext, f0=10, f1=2000, t1=t_ext[-1], method='logarithmic')
    _, step_out, _ = signal.lsim(sys1, U=step_in, T=t_ext)
    _, square_out, _ = signal.lsim(sys1, U=square_in, T=t_ext)
    _, chirp_out, _ = signal.lsim(sys1, U=chirp_in, T=t_ext)

    fig2, axs2 = plt.subplots(3,1, figsize=(8,8))
    axs2[0].plot(t_ext*1000, step_in, label='阶跃输入')
    axs2[0].plot(t_ext*1000, step_out, label='输出')
    axs2[0].set_title('阶跃响应')
    axs2[0].grid(True, ls=':')
    axs2[0].legend()
    axs2[1].plot(t_ext*1000, square_in, label='方波输入')
    axs2[1].plot(t_ext*1000, square_out, label='输出')
    axs2[1].set_title('500Hz 方波')
    axs2[1].grid(True, ls=':')
    axs2[1].legend()
    axs2[2].plot(t_ext*1000, chirp_in, label='Chirp 输入')
    axs2[2].plot(t_ext*1000, chirp_out, label='输出')
    axs2[2].set_title('Chirp (10→2000Hz)')
    axs2[2].grid(True, ls=':')
    axs2[2].legend()
    axs2[2].set_xlabel('时间 (ms)')
    fig2.tight_layout()
    fig2.savefig(out_dir / 'various_inputs.png', dpi=160)
    plt.close(fig2)

    # ========= 二阶级联比较 =========
    sys2 = cascade_second_order(alpha)
    t2, _, y2_1, amp1, phase1 = sine_response(sys1, 500, fs, dur)
    _, _, y2_2, amp2, phase2 = sine_response(sys2, 500, fs, dur)
    fig3, ax3 = plt.subplots(figsize=(8,3))
    ax3.plot(t2*1000, y2_1, label=f'一阶输出 (幅:{amp1:.3f})')
    ax3.plot(t2*1000, y2_2, label=f'二阶级联 (幅:{amp2:.3f})')
    ax3.set_title('500Hz: 一阶 vs 二阶级联')
    ax3.set_xlabel('时间 (ms)')
    ax3.set_ylabel('电压')
    ax3.grid(True, ls=':')
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(out_dir / 'first_vs_second_order_500Hz.png', dpi=160)
    plt.close(fig3)

    # ========= Bode 图 =========
    w = np.logspace(1, 5, 400)
    w, mag1, phase1_bode = signal.bode(sys1, w=w)
    w, mag2, phase2_bode = signal.bode(sys2, w=w)
    fig4, (axm, axp) = plt.subplots(2,1, figsize=(7,6), sharex=True)
    axm.semilogx(w/(2*np.pi), mag1, label='一阶 |H| (dB)')
    axm.semilogx(w/(2*np.pi), mag2, label='二阶 |H| (dB)')
    axm.axvline(fc, color='k', ls='--', alpha=0.6, label=f'fc≈{fc:.1f}Hz')
    axm.set_ylabel('幅度 (dB)')
    axm.grid(True, which='both', ls=':')
    axm.legend(fontsize=8)
    axp.semilogx(w/(2*np.pi), phase1_bode, label='一阶 相位')
    axp.semilogx(w/(2*np.pi), phase2_bode, label='二阶 相位')
    axp.set_ylabel('相位 (deg)')
    axp.set_xlabel('频率 (Hz)')
    axp.grid(True, which='both', ls=':')
    axp.legend(fontsize=8)
    fig4.tight_layout()
    fig4.savefig(out_dir / 'bode.png', dpi=160)
    plt.close(fig4)

    # ========= 元数据保存 =========
    meta_path = out_dir / 'meta_info.txt'
    with open(meta_path, 'w', encoding='utf-8') as fm:
        fm.write('RC 低通仿真参数与文件说明\n')
        fm.write(f'use_physical={use_physical}\n')
        fm.write(f'alpha={alpha:.6e}\nfc≈{fc:.3f} Hz\nfs={fs} Hz\n')
        if use_physical:
            fm.write(f'R={R} Ω, C={C} F\n')
        fm.write('图像文件: single_frequency_time.png, multi_frequency_time.png, various_inputs.png, first_vs_second_order_500Hz.png, bode.png\n')
        fm.write('数据: amplitude_phase_table.(csv|txt)\n')

    # 控制台仅简要提示
    print(f'仿真完成。结果与图像已保存至: {out_dir.resolve()}')

if __name__ == '__main__':
    main()
