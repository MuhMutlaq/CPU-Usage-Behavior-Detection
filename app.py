import tkinter as tk
from tkinter import font
import threading
import time
import os
import queue
import random
import numpy as np
import psutil
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# MODEL LOADING
try:
    import joblib
    _candidates= [
        os.path.join(os.path.dirname(__file__), "models", "XGBoost_ModelAnalysis.pkl"),
        "./models/XGBoost_ModelAnalysis.pkl",
        "./models/xgb_anomaly_v4_final.pkl",
        "./models/model_analysis.pkl",
    ]
    MODEL_PATH= next((p for p in _candidates if os.path.exists(p)), None)
    model= joblib.load(MODEL_PATH) if MODEL_PATH else None
    MODEL_LOADED= model is not None
    print(f"[✓] Model loaded  →  {MODEL_PATH}" if MODEL_LOADED
          else "[!] No model file found — running in demo mode.")
except Exception as e:
    MODEL_LOADED= False
    model= None
    print(f"[!] Model not found — demo mode. ({e})")

# THEME
BG_DARK= "#0d1117"
BG_CARD= "#161b22"
BG_CARD2= "#1c2128"
BG_HEADER= "#0d1117"
ACCENT_BLUE= "#2f81f7"
ACCENT_GREEN= "#3fb950"
ACCENT_RED= "#f85149"
ACCENT_ORANGE= "#f0883e"
ACCENT_PURPLE= "#d2a8ff"
ACCENT_CYAN= "#79c0ff"
TEXT_PRIMARY= "#e6edf3"
TEXT_SECONDARY= "#8b949e"
TEXT_MUTED= "#484f58"
BORDER_COLOR= "#30363d"
CHART_BG= "#0d1117"
CHART_GRID= "#21262d"

HISTORY_LEN= 60
UPDATE_MS= 1000

# FEATURE ORDER — MUST MATCH TRAINING PIPELINE EXACTLY
FEATURE_ORDER= [
    "cpu_utilization",
    "memory_usage",
    "temperature",
    "thermal_load",
    "stress_score",
    "cpu_mem_product",
    "mem_temp_product",
    "cpu_temp_product",
]

# FEATURE ENGINEERING  — identical to training script
def engineer_features(cpu: float, mem: float, temp: float) -> dict:
    """
    Compute all 8 model features from the 3 raw KS-separable signals.
    No other inputs needed — thread/process/ratio features were removed.
    """
    return {
        "cpu_utilization": cpu,
        "memory_usage": mem,
        "temperature": temp,
        "thermal_load": (cpu + mem + temp) / 3,
        "stress_score": (cpu / 100 * 0.4) + (mem / 100 * 0.4) + (temp / 100 * 0.2),
        "cpu_mem_product": cpu * mem,
        "mem_temp_product": mem * temp,
        "cpu_temp_product": cpu * temp,
    }


# ==============================================================================
#  AI PREDICTOR
# ==============================================================================
def predict_anomaly(features: dict):
    """
    Returns (is_anomaly: bool, confidence: float 0-1).
    Only passes the 8 FEATURE_ORDER columns to the model.
    All other keys in `features` are display-only and safely ignored.
    """
    if not MODEL_LOADED or model is None:
        is_anom= (features["cpu_utilization"] >= 80 and
                   features["memory_usage"] >= 85 and
                   features["temperature"] >= 75)
        conf= round(random.uniform(0.82, 0.97), 3) if is_anom \
               else round(random.uniform(0.01, 0.10), 3)
        return is_anom, conf

    # Build row with ONLY the 8 model features — no KeyError possible
    row= np.array([[features[f] for f in FEATURE_ORDER]])
    proba= float(model.predict_proba(row)[0][1])
    return proba >= 0.5, round(proba, 4)

# DATA COLLECTOR
class SystemDataCollector:
    def __init__(self):
        self._sim_temp_base= 52.0
        self._sim_temp_noise= 0.0
        self._prev_disk_r= 0
        self._prev_disk_w= 0
        self._prev_net_s= 0
        self._prev_net_r= 0
        self._prev_ctx= 0
        self._uptime_start= psutil.boot_time()
        psutil.cpu_percent(interval= None)

    def _temperature(self) -> float:
        try:
            temps= psutil.sensors_temperatures()
            if temps:
                for key in ("coretemp", "cpu_thermal", "k10temp", "acpitz"):
                    if key in temps and temps[key]:
                        return float(temps[key][0].current)
                first= list(temps.keys())[0]
                if temps[first]:
                    return float(temps[first][0].current)
        except Exception:
            pass
        cpu= psutil.cpu_percent(interval= None)
        target= 38 + cpu * 0.48
        self._sim_temp_base+= (target - self._sim_temp_base) * 0.07
        self._sim_temp_noise= self._sim_temp_noise * 0.7 + random.gauss(0, 0.5)
        return round(max(30, min(95, self._sim_temp_base + self._sim_temp_noise)), 1)

    def _disk_io(self) -> float:
        try:
            c = psutil.disk_io_counters()
            if c:
                total= c.read_bytes + c.write_bytes
                delta= max(0, total - self._prev_disk_r - self._prev_disk_w)
                self._prev_disk_r= c.read_bytes
                self._prev_disk_w= c.write_bytes
                return round(delta / 1_000_000, 2)
        except Exception:
            pass
        return round(random.uniform(0.5, 18), 2)

    def _network_latency(self) -> float:
        try:
            net= psutil.net_io_counters()
            ds= abs(net.bytes_sent - self._prev_net_s)
            dr= abs(net.bytes_recv - self._prev_net_r)
            self._prev_net_s= net.bytes_sent
            self._prev_net_r= net.bytes_recv
            load= min(1.0, (ds + dr) / 1_000_000)
            return round(8 + load * 85 + random.gauss(0, 4), 1)
        except Exception:
            return round(random.uniform(5, 120), 1)

    def _ctx_switches(self) -> int:
        try:
            cs= psutil.cpu_stats()
            delta= cs.ctx_switches - self._prev_ctx
            self._prev_ctx= cs.ctx_switches
            return max(100, min(2000, delta if delta > 0 else 0))
        except Exception:
            return 0

    def _thread_count(self) -> int:
        try:
            total= 0
            for p in psutil.process_iter(["num_threads"]):
                try:
                    total+= p.info["num_threads"] or 0
                except Exception:
                    pass
            return total
        except Exception:
            return 0

    def _power(self) -> float:
        cpu= psutil.cpu_percent(interval= None)
        return round(50 + cpu * 2.5 + random.gauss(0, 8), 1)

    def collect(self) -> dict:
        cpu= psutil.cpu_percent(interval= None)
        mem= psutil.virtual_memory().percent
        temp= self._temperature()

        model_feats= engineer_features(cpu, mem, temp)

        display= {
            "disk_io": self._disk_io(),
            "network_latency": self._network_latency(),
            "process_count": len(psutil.pids()),
            "thread_count": self._thread_count(),
            "context_switches": self._ctx_switches(),
            "cache_miss_rate": round(random.uniform(0.02, 0.18), 4),
            "power_consumption": self._power(),
            "uptime": (time.time() - self._uptime_start) / 60,
        }
        return {**model_feats, **display}

# GAUGE WIDGET
class GaugeWidget(tk.Frame):
    def __init__(self, parent, title, unit= "%", max_val= 100,
                 warn_at= 70, crit_at= 85, **kwargs):
        super().__init__(parent, bg= BG_CARD, **kwargs)
        self.title= title
        self.unit= unit
        self.max_val= max_val
        self.warn_at= warn_at
        self.crit_at= crit_at

        self.fig= Figure(figsize= (2, 2), dpi= 90, facecolor= BG_CARD)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(BG_CARD)
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.canvas= FigureCanvasTkAgg(self.fig, master= self)
        self.canvas.get_tk_widget().pack(fill= "both", expand=True)
        self._draw(0)

    def _color_for(self, val):
        pct= val if self.unit == "%" else val / self.max_val * 100
        if pct >= self.crit_at: return ACCENT_RED
        if pct >= self.warn_at: return ACCENT_ORANGE
        return ACCENT_GREEN

    def _arc_pts(self, t1, t2, r= 1.0, n= 200):
        t= np.linspace(np.radians(t1), np.radians(t2), n)
        return r * np.cos(t), r * np.sin(t)

    def _draw(self, value):
        ax = self.ax
        ax.clear()
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")

        pct= min(value / self.max_val, 1.0)
        color= self._color_for(value)
        S, W= 220, 280

        for r in np.linspace(0.80, 0.99, 14):
            x, y= self._arc_pts(S, S + W, r= r)
            ax.plot(x, y, color= BG_CARD2, lw= 1.8, solid_capstyle= "round", zorder= 1)

        if pct > 0.001:
            end= S + pct * W
            for r in np.linspace(0.80, 0.99, 14):
                x, y = self._arc_pts(S, end, r= r)
                ax.plot(x, y, color= color, lw= 1.8,
                        solid_capstyle= "round", zorder= 2, alpha= 0.92)
            x, y= self._arc_pts(S, end, r= 0.895, n= 300)
            ax.plot(x, y, color= color, lw= 2.8, solid_capstyle= "round", zorder= 3)

        ax.add_patch(plt.Circle((0, 0), 0.72, color= BG_DARK, zorder= 4))

        display= f"{value:.0f}" if self.unit == "%" else f"{value:.1f}"
        ax.text(0, 0.10, f"{display}{self.unit}",
                ha= "center", va= "center",
                fontsize= 14, fontweight= "bold", color= color, zorder= 6)
        ax.text(0, -0.30, self.title,
                ha= "center", va= "center",
                fontsize= 7.5, color= TEXT_SECONDARY, zorder= 6)
        self.canvas.draw_idle()

    def update(self, value):
        self._draw(value)

# SPARKLINE CHART
class SparklineChart(tk.Frame):
    def __init__(self, parent, title, color= ACCENT_BLUE,
                 y_max= 100, y_label= "", **kwargs):
        super().__init__(parent, bg= BG_CARD, **kwargs)
        self.title= title
        self.color= color
        self.y_max= y_max
        self.y_label= y_label
        self.data= [0.0] * HISTORY_LEN

        self.fig= Figure(figsize= (4, 1.55), dpi= 90, facecolor= BG_CARD)
        self.ax= self.fig.add_subplot(111)
        self.fig.subplots_adjust(left= 0.12, right= 0.98, top= 0.82, bottom= 0.22)
        self._setup_chart()
        self.canvas = FigureCanvasTkAgg(self.fig, master= self)
        self.canvas.get_tk_widget().pack(fill= "both", expand= True)
        self._draw()

    def _setup_chart(self):
        ax= self.ax
        ax.set_facecolor(CHART_BG)
        ax.tick_params(colors= TEXT_MUTED, labelsize= 6)
        ax.spines["bottom"].set_color(BORDER_COLOR)
        ax.spines["left"].set_color(BORDER_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, self.y_max)
        ax.set_xlim(0, HISTORY_LEN - 1)
        ax.set_yticks([0, self.y_max // 2, self.y_max])
        ax.xaxis.set_visible(False)
        ax.set_title(self.title, color= TEXT_SECONDARY, fontsize= 8, loc= "left", pad= 3)
        ax.grid(axis= "y", color= CHART_GRID, lw= 0.5, alpha= 0.6)

    def _draw(self):
        self.ax.clear()
        self._setup_chart()
        x= np.arange(len(self.data))
        y= np.array(self.data)
        self.ax.fill_between(x, y, alpha= 0.15, color= self.color)
        self.ax.plot(x, y, color= self.color, lw= 1.5, zorder= 3)
        if self.data:
            cur= self.data[-1]
            self.ax.text(HISTORY_LEN - 1,
                         min(cur + self.y_max * 0.08, self.y_max * 0.95),
                         f"{cur:.1f}{self.y_label}",
                         ha= "right", va= "bottom",
                         fontsize= 7, color= self.color, fontweight= "bold")
        self.canvas.draw_idle()

    def push(self, value):
        self.data.append(float(value))
        if len(self.data) > HISTORY_LEN:
            self.data.pop(0)
        self._draw()

# ANOMALY MODE ALERT POPUP
class AnomalyAlert(tk.Toplevel):
    def __init__(self, parent, metrics):
        super().__init__(parent)
        self.title("ANOMALY MODE BEHAVIOR DETECTED")
        self.configure(bg= BG_DARK)
        self.resizable(False, False)
        self.attributes("-topmost", True)
        w, h= 480, 340
        sw, sh= self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        self._build(metrics)
        self.grab_set()

    def _build(self, m):
        tk.Frame(self, bg= ACCENT_RED, height= 4).pack(fill= "x")

        tf= tk.Frame(self, bg= BG_DARK)
        tf.pack(fill= "x", padx= 24, pady= (18, 0))
        tk.Label(tf, text= "🚨", font= ("Segoe UI", 28),
                 bg= BG_DARK, fg= ACCENT_RED).pack(side= "left")
        tk.Label(tf, text= "ANOMALY MODE BEHAVIOR DETECTED",
                 font= ("Segoe UI", 17, "bold"),
                 bg= BG_DARK, fg= ACCENT_RED).pack(side= "left", pady= 4)

        tk.Label(self,
                 text= "The AI model classified the current system state\n"
                      "as ABNORMAL. Immediate attention may be required.",
                 font= ("Segoe UI", 10), bg= BG_DARK, fg= TEXT_SECONDARY,
                 justify= "left").pack(padx= 24, pady= (8, 0), anchor= "w")

        tk.Frame(self, bg= BORDER_COLOR, height= 1).pack(fill= "x", padx= 24, pady= 12)

        grid= tk.Frame(self, bg= BG_DARK)
        grid.pack(fill= "x", padx= 24)

        # Only show the 8 model features in the alert popup
        cells= [
            ("CPU Usage", f"{m['cpu_utilization']:.1f}%", m["cpu_utilization"] > 80),
            ("Memory", f"{m['memory_usage']:.1f}%", m["memory_usage"] > 85),
            ("Temperature", f"{m['temperature']:.1f}°C", m["temperature"] > 75),
            ("Thermal Load", f"{m['thermal_load']:.1f}", m["thermal_load"] > 80),
            ("Stress Score", f"{m['stress_score']:.3f}", m["stress_score"] > 0.80),
            ("CPU×MEM", f"{m['cpu_mem_product']:.0f}", m["cpu_mem_product"] > 7000),
        ]
        for i, (lbl, val, high) in enumerate(cells):
            r, c= divmod(i, 3)
            cell= tk.Frame(grid, bg= BG_CARD, padx= 10, pady= 6)
            cell.grid(row= r, column= c, padx= 4, pady= 4, sticky= "ew")
            grid.columnconfigure(c, weight= 1)
            tk.Label(cell, text= lbl, font= ("Segoe UI", 8),
                     bg= BG_CARD, fg= TEXT_MUTED).pack(anchor= "w")
            tk.Label(cell, text= val,
                     font= ("Segoe UI", 11, "bold"),
                     bg= BG_CARD,
                     fg= ACCENT_RED if high else TEXT_PRIMARY).pack(anchor= "w")

        tk.Label(self, text= f"Detected at {time.strftime('%Y-%m-%d %H:%M:%S')}",
                 font= ("Segoe UI", 8), bg= BG_DARK, fg= TEXT_MUTED).pack(pady= (10, 4))

        tk.Button(self, text= "ACKNOWLEDGE",
                  font= ("Segoe UI", 10, "bold"),
                  bg= ACCENT_RED, fg= "white", relief= "flat",
                  activebackground= "#c0392b", activeforeground= "white",
                  cursor= "hand2", padx= 20, pady= 8,
                  command= self.destroy).pack(pady= (4, 16))

# STAT CARD
class StatCard(tk.Frame):
    def __init__(self, parent, label, icon= "", color= TEXT_PRIMARY, **kwargs):
        super().__init__(parent, bg= BG_CARD2, padx= 12, pady= 8, **kwargs)
        self.color= color
        tk.Label(self, text= f"{icon}  {label}", font= ("Segoe UI", 8),
                 bg= BG_CARD2, fg= TEXT_MUTED).pack(anchor= "w")
        self.val_lbl= tk.Label(self, text= "—", font= ("Segoe UI", 13, "bold"),
                                bg= BG_CARD2, fg= color)
        self.val_lbl.pack(anchor= "w")

    def update(self, text, color= None):
        self.val_lbl.config(text= text, fg= color or self.color)

# LOG PANEL
class LogPanel(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg= BG_CARD, **kwargs)
        hdr= tk.Frame(self, bg= BG_CARD)
        hdr.pack(fill= "x", padx= 12, pady= (10, 4))
        tk.Label(hdr, text= "📋 Event Log", font= ("Segoe UI", 10, "bold"),
                 bg=BG_CARD, fg=TEXT_PRIMARY).pack(side= "left")

        self.text= tk.Text(self, bg= BG_DARK, fg= TEXT_SECONDARY,
                            font= ("Consolas", 8), relief= "flat",
                            state= "disabled", wrap= "word")
        self.text.pack(fill= "both", expand= True, padx= 8, pady= (0, 8))
        self.text.tag_config("anom", foreground= ACCENT_RED)
        self.text.tag_config("warn", foreground= ACCENT_ORANGE)
        self.text.tag_config("ok", foreground= ACCENT_GREEN)
        self.text.tag_config("info", foreground= ACCENT_BLUE)
        self.text.tag_config("ts", foreground= TEXT_MUTED)

    def log(self, msg, tag= "info"):
        self.text.config(state= "normal")
        self.text.insert("end", f"[{time.strftime('%H:%M:%S')}]  ", "ts")
        self.text.insert("end", msg + "\n", tag)
        self.text.see("end")
        if int(self.text.index("end-1c").split(".")[0]) > 200:
            self.text.delete("1.0", "10.0")
        self.text.config(state= "disabled")

# MAIN APPLICATION
class ServerMonitorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Server Anomaly Monitor — AI-Powered Real-Time Analysis")
        self.configure(bg= BG_DARK)
        self.minsize(1200, 760)

        self._running= True
        self._alert_cooldown= 0
        self._total_samples= 0
        self._total_anomalies= 0
        self._data_queue= queue.Queue()
        self._collector= SystemDataCollector()

        self._setup_fonts()
        self._build_ui()
        self._start_collector_thread()
        self._schedule_update()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # Fonts
    def _setup_fonts(self):
        try:
            self._font_title= font.Font(family= "Segoe UI", size= 13, weight= "bold")
            self._font_body= font.Font(family= "Segoe UI", size= 9)
            self._font_mono= font.Font(family= "Consolas",  size= 9)
        except Exception:
            self._font_title= font.Font(size= 13, weight= "bold")
            self._font_body= font.Font(size= 9)
            self._font_mono= font.Font(size= 9)

    # Root Layout
    def _build_ui(self):
        self._build_header()

        outer= tk.Frame(self, bg= BG_DARK)
        outer.pack(fill= "both", expand= True)

        self._vscroll= tk.Scrollbar(outer, orient= "vertical",
                                     bg= BG_CARD2, troughcolor= BG_DARK,
                                     activebackground= ACCENT_BLUE,
                                     highlightthickness= 0, bd= 0, width= 10)
        self._vscroll.pack(side= "right", fill= "y")

        self._scroll_canvas= tk.Canvas(outer, bg= BG_DARK, highlightthickness= 0, yscrollcommand= self._vscroll.set)
        self._scroll_canvas.pack(side= "left", fill= "both", expand= True)
        self._vscroll.config(command= self._scroll_canvas.yview)

        self._inner= tk.Frame(self._scroll_canvas, bg= BG_DARK)
        self._canvas_window= self._scroll_canvas.create_window((0, 0), window= self._inner, anchor= "nw")

        self._scroll_canvas.bind(
            "<Configure>",
            lambda e: self._scroll_canvas.itemconfig(self._canvas_window, width= e.width))
        self._inner.bind(
            "<Configure>",
            lambda e: self._scroll_canvas.configure(
                scrollregion= self._scroll_canvas.bbox("all")))

        def _wheel(event):
            if event.num == 4: self._scroll_canvas.yview_scroll(-1, "units")
            elif event.num == 5: self._scroll_canvas.yview_scroll( 1, "units")
            else: self._scroll_canvas.yview_scroll(
                    -1 if event.delta > 0 else 1, "units")

        self.bind_all("<MouseWheel>", _wheel)
        self.bind_all("<Button-4>", _wheel)
        self.bind_all("<Button-5>", _wheel)

        body= tk.Frame(self._inner, bg= BG_DARK)
        body.pack(fill= "both", expand= True, padx= 12, pady= (4, 8))
        body.columnconfigure(0, weight= 2, minsize= 220)
        body.columnconfigure(1, weight= 5)
        body.columnconfigure(2, weight= 3, minsize= 240)
        body.rowconfigure(0, weight= 1)

        self._build_left_panel(body)
        self._build_center_panel(body)
        self._build_right_panel(body)

    # Header
    def _build_header(self):
        hdr= tk.Frame(self, bg= BG_HEADER, pady= 10)
        hdr.pack(fill= "x")

        left= tk.Frame(hdr, bg= BG_HEADER)
        left.pack(side= "left", padx= 16)
        tk.Label(left, text= "⬡", font= ("Segoe UI", 20),
                 bg= BG_HEADER, fg= ACCENT_BLUE).pack(side= "left")
        tk.Label(left, text= "SERVER ANOMALY MODE MONITOR",
                 font= ("Segoe UI", 13, "bold"),
                 bg= BG_HEADER, fg= TEXT_PRIMARY).pack(side= "left")
        tk.Label(left, text= "AI-Powered Real-Time Analysis",
                 font= ("Segoe UI", 9),
                 bg= BG_HEADER, fg= TEXT_MUTED).pack(side= "left")

        right= tk.Frame(hdr, bg= BG_HEADER)
        right.pack(side= "right", padx= 16)

        mc= ACCENT_GREEN if MODEL_LOADED else ACCENT_ORANGE
        mt= "AI Model: Active" if MODEL_LOADED else "AI Model: Demo"
        tk.Label(right, text= mt, font= ("Segoe UI", 8, "bold"),
                 bg= mc, fg= "white", padx= 10, pady= 3).pack(side= "right", padx= 4)

        self._live_indicator= tk.Label(right, text= "● LIVE",
                                        font= ("Segoe UI", 8, "bold"),
                                        bg= ACCENT_GREEN, fg= "white",
                                        padx= 10, pady= 3)
        self._live_indicator.pack(side= "right", padx= 4)

        self._clock_lbl= tk.Label(right, text= "", font= ("Segoe UI", 8),
                                   bg= BG_HEADER, fg= TEXT_MUTED)
        self._clock_lbl.pack(side= "right", padx= 12)

        tk.Frame(self, bg= BORDER_COLOR, height= 1).pack(fill= "x")

    # Left Panel
    def _build_left_panel(self, parent):
        col= tk.Frame(parent, bg= BG_DARK)
        col.grid(row= 0, column= 0, sticky= "nsew", padx= (0, 6), pady= 4)

        tk.Label(col, text= "SYSTEM GAUGES", font= ("Segoe UI", 8, "bold"),
                 bg= BG_DARK, fg= TEXT_MUTED).pack(anchor= "w", pady= (4, 6))

        gf= tk.Frame(col, bg= BG_DARK)
        gf.pack(fill= "x")
        self.gauge_cpu= GaugeWidget(gf, "CPU", "%", 100, 70, 85)
        self.gauge_mem= GaugeWidget(gf, "MEM", "%", 100, 70, 88)
        self.gauge_temp= GaugeWidget(gf, "TEMP", "°C", 100, 70, 80)
        for g in (self.gauge_cpu, self.gauge_mem, self.gauge_temp):
            g.pack(fill= "x", pady= 3)

        # AI prediction card
        tk.Label(col, text= "AI PREDICTION", font= ("Segoe UI", 8, "bold"),
                 bg= BG_DARK, fg= TEXT_MUTED).pack(anchor= "w", pady= (14, 6))

        pred= tk.Frame(col, bg= BG_CARD, padx= 14, pady= 12)
        pred.pack(fill= "x")
        self._pred_icon= tk.Label(pred, text= "◉", font= ("Segoe UI", 22),
                                    bg= BG_CARD, fg= ACCENT_GREEN)
        self._pred_icon.pack()
        self._pred_label= tk.Label(pred, text= "NORMAL MODE",
                                    font= ("Segoe UI", 13, "bold"),
                                    bg= BG_CARD, fg= ACCENT_GREEN)
        self._pred_label.pack()
        self._conf_label= tk.Label(pred, text= "Confidence: —",
                                    font= ("Segoe UI", 8),
                                    bg= BG_CARD, fg= TEXT_MUTED)
        self._conf_label.pack(pady= (2, 0))

        # Session Stats
        tk.Label(col, text= "SESSION STATS", font= ("Segoe UI", 8, "bold"),
                 bg= BG_DARK, fg= TEXT_MUTED).pack(anchor= "w", pady= (14, 6))

        sf= tk.Frame(col, bg= BG_DARK)
        sf.pack(fill= "x")
        sf.columnconfigure(0, weight= 1)
        sf.columnconfigure(1, weight= 1)

        self._card_samples= StatCard(sf, "Samples", "📊", ACCENT_CYAN)
        self._card_anomalies= StatCard(sf, "Anomalies", "🔴", ACCENT_RED)
        self._card_rate= StatCard(sf, "Anom Rate", "%",  ACCENT_ORANGE)
        self._card_uptime= StatCard(sf, "Uptime", "⏱", ACCENT_PURPLE)

        self._card_samples.grid(row= 0, column= 0, padx= 2, pady= 2, sticky= "ew")
        self._card_anomalies.grid(row= 0, column= 1, padx= 2, pady= 2, sticky= "ew")
        self._card_rate.grid(row= 1, column= 0, padx= 2, pady= 2, sticky= "ew")
        self._card_uptime.grid(row= 1, column= 1, padx= 2, pady= 2, sticky= "ew")

    # Center Panel
    def _build_center_panel(self, parent):
        col= tk.Frame(parent, bg= BG_DARK)
        col.grid(row= 0, column= 1, sticky= "nsew", padx= 6, pady= 4)

        tk.Label(col, text= "REAL-TIME TELEMETRY", font= ("Segoe UI", 8, "bold"),
                 bg= BG_DARK, fg= TEXT_MUTED).pack(anchor= "w", pady= (4, 6))

        cf= tk.Frame(col, bg= BG_DARK)
        cf.pack(fill= "both", expand= True)
        for i in range(3): cf.columnconfigure(i, weight= 1)
        for i in range(2): cf.rowconfigure(i, weight= 1)

        self.chart_cpu= SparklineChart(cf, "CPU Utilization (%)", ACCENT_BLUE, 100, "%")
        self.chart_mem= SparklineChart(cf, "Memory Usage (%)", ACCENT_PURPLE, 100, "%")
        self.chart_temp= SparklineChart(cf, "Temperature (°C)", ACCENT_RED, 100, "°C")
        self.chart_net= SparklineChart(cf, "Network Latency (ms)", ACCENT_CYAN, 250, "ms")
        self.chart_disk= SparklineChart(cf, "Disk I/O (MB/s)", ACCENT_ORANGE, 50, "MB/s")
        self.chart_stress= SparklineChart(cf, "Stress Score (0–1)", ACCENT_GREEN, 1, "")

        self.chart_cpu.grid(row= 0, column= 0, padx= 3, pady= 3, sticky= "nsew")
        self.chart_mem.grid(row= 0, column= 1, padx= 3, pady= 3, sticky= "nsew")
        self.chart_temp.grid(row= 0, column= 2, padx= 3, pady= 3, sticky= "nsew")
        self.chart_net.grid(row= 1, column= 0, padx= 3, pady= 3, sticky= "nsew")
        self.chart_disk.grid(row= 1, column= 1, padx= 3, pady= 3, sticky= "nsew")
        self.chart_stress.grid(row= 1, column= 2, padx= 3, pady= 3, sticky= "nsew")

        # Probability Chart
        tk.Label(col, text= "CLIENT BEHAVIOR — ANOMALY MODE PROBABILITY LINE",
                 font= ("Segoe UI", 8, "bold"),
                 bg= BG_DARK, fg= TEXT_MUTED).pack(anchor= "w", pady= (14, 4))
        self._build_probability_chart(col)

    # Probability Chart
    def _build_probability_chart(self, parent):
        self._prob_data= [0.0] * HISTORY_LEN
        self._prob_anom_flags= [False] * HISTORY_LEN

        card= tk.Frame(parent, bg= BG_CARD)
        card.pack(fill= "x")

        info= tk.Frame(card, bg= BG_CARD2, padx= 12, pady= 6)
        info.pack(fill= "x")
        tk.Label(info, text= "📈 Anomaly Mode Probability Over Time",
                 font= ("Segoe UI", 9, "bold"),
                 bg= BG_CARD2, fg= TEXT_PRIMARY).pack(side= "left")

        ri= tk.Frame(info, bg= BG_CARD2)
        ri.pack(side= "right")
        self._prob_badge= tk.Label(ri, text= "P= 0.000",
                                    font= ("Segoe UI", 9, "bold"),
                                    bg= ACCENT_GREEN, fg= "white", padx= 8, pady= 2)
        self._prob_badge.pack(side= "right", padx= (6, 0))
        self._prob_status= tk.Label(ri, text= "NORMAL",
                                     font= ("Segoe UI", 8),
                                     bg= BG_CARD2, fg= TEXT_MUTED)
        self._prob_status.pack(side= "right")

        self._prob_fig= Figure(figsize= (6, 2.6), dpi= 90, facecolor= BG_CARD)
        self._prob_ax= self._prob_fig.add_subplot(111)
        self._prob_fig.subplots_adjust(left= 0.07, right= 0.97, top= 0.88, bottom= 0.18)
        self._prob_ax.set_facecolor(CHART_BG)

        self._prob_canvas= FigureCanvasTkAgg(self._prob_fig, master= card)
        self._prob_canvas.get_tk_widget().pack(fill= "both", expand= True, pady= (0, 4))

        leg= tk.Frame(card, bg= BG_CARD, padx= 12, pady= 4)
        leg.pack(fill= "x")
        for c, txt in [(ACCENT_CYAN, "● Probability line"),
                       (ACCENT_ORANGE, "● Rolling avg (10s)"),
                       (ACCENT_RED, "▐ Anomaly zone"),
                       (ACCENT_GREEN, "▐ Normal zone")]:
            tk.Label(leg, text= txt, font= ("Segoe UI", 7),
                     bg= BG_CARD, fg= c).pack(side= "left", padx= 8)

    def _update_probability_chart(self, prob, is_anom=False):
        self._prob_data.append(float(prob))
        self._prob_anom_flags.append(bool(is_anom))
        if len(self._prob_data) > HISTORY_LEN:
            self._prob_data.pop(0)
            self._prob_anom_flags.pop(0)

        bc= ACCENT_RED if prob >= 0.5 else ACCENT_GREEN
        self._prob_badge.config(text= f"P= {prob:.3f}", bg= bc)
        self._prob_status.config(
            text= "! ANOMALY MODE BEHAVIOR DETECTED" if prob >= 0.5 else "✔ NORMAL MODE", fg= bc)

        ax= self._prob_ax
        ax.clear()
        ax.set_facecolor(CHART_BG)

        n, THR= len(self._prob_data), 0.5
        x= np.arange(n)
        y= np.array(self._prob_data)

        ax.axhspan(THR, 1.0, alpha= 0.04, color= ACCENT_RED, zorder= 0)
        ax.axhspan(0.0, THR, alpha= 0.04, color= ACCENT_GREEN, zorder= 0)
        ax.set_ylim(-0.02, 1.08)
        ax.set_xlim(0, HISTORY_LEN - 1)
        ax.yaxis.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.set_ticklabels(["0", "25%", "50%", "75%", "100%"])
        ax.tick_params(axis= "y", colors= TEXT_MUTED, labelsize= 6.5, length= 2)
        ax.grid(axis= "y", color= CHART_GRID, lw= 0.6, alpha= 0.5, zorder= 1)
        for sp in ax.spines.values(): sp.set_color(BORDER_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(np.linspace(0, HISTORY_LEN - 1, 7))
        ax.set_xticklabels(
            [f"-{int((HISTORY_LEN - 1 - t))}s"
             for t in np.linspace(0, HISTORY_LEN - 1, 7)],
            fontsize= 6, color= TEXT_MUTED)
        ax.tick_params(axis= "x", length= 2, colors= TEXT_MUTED)

        ax.fill_between(x, y, THR, where= (y >= THR), color= ACCENT_RED,
                        alpha= 0.22, zorder= 2, interpolate= True)
        ax.fill_between(x, y, THR, where= (y <  THR), color= ACCENT_GREEN,
                        alpha= 0.12, zorder= 2, interpolate= True)
        ax.plot(x, y, color= ACCENT_CYAN, lw= 2.0, zorder= 5,
                solid_capstyle= "round", solid_joinstyle= "round")

        if n >= 3:
            w= min(10, n)
            roll= np.convolve(y, np.ones(w) / w, mode= "same")
            for i in range(w // 2):
                roll[i]= y[:i+1].mean()
                roll[n - 1 - i]= y[n - 1 - i:].mean()
            ax.plot(x, roll, color= ACCENT_ORANGE, lw= 1.4,
                    linestyle= "--", alpha= 0.85, zorder= 4)

        ax.axhline(THR, color= ACCENT_RED, lw= 1.0, linestyle= ":", alpha= 0.8, zorder= 3)
        ax.text(HISTORY_LEN - 1, THR + 0.03, "threshold",
                ha= "right", va= "bottom", fontsize= 6, color= ACCENT_RED, alpha= 0.8)

        for xi, flag in zip(x, self._prob_anom_flags):
            if flag:
                ax.axvline(xi, color= ACCENT_RED, lw= 0.6, alpha= 0.35, zorder= 3)

        anom_x= [xi for xi, f in zip(x, self._prob_anom_flags) if f]
        anom_y= [y[xi] for xi in anom_x]
        if anom_x:
            ax.scatter(anom_x, anom_y, color= ACCENT_RED, s= 18,
                       zorder= 6, edgecolors= "white", linewidths= 0.4)

        if n > 0:
            cc= ACCENT_RED if y[-1] >= THR else ACCENT_GREEN
            ax.scatter([n-1], [y[-1]], color= cc, s= 30, zorder= 7,
                       edgecolors= "white", linewidths= 0.6)
            ax.annotate(f"{y[-1]:.3f}",
                        xy= (n-1, y[-1]),
                        xytext= (max(0, n - 4), min(y[-1] + 0.10, 1.0)),
                        fontsize= 7, color= cc, fontweight= "bold",
                        arrowprops= dict(arrowstyle= "-", color= cc, lw= 0.8, alpha= 0.6),
                        zorder= 8)

        ax.text(HISTORY_LEN-1, 0.82, "ANOMALY MODE", ha="right", va="center",
                fontsize= 6.5, color= ACCENT_RED, alpha=0.55, fontweight="bold")
        ax.text(HISTORY_LEN-1, 0.18, "NORMAL MODE", ha= "right", va= "center",
                fontsize= 6.5, color= ACCENT_GREEN, alpha= 0.55, fontweight= "bold")
        ax.set_title(
            f"Client Behavior Anomaly Probability   "
            f"[events this window: {sum(self._prob_anom_flags)}]",
            color= TEXT_SECONDARY, fontsize= 8, loc= "left", pad= 4)
        self._prob_canvas.draw_idle()

    # Right Panel
    def _build_right_panel(self, parent):
        col= tk.Frame(parent, bg= BG_DARK)
        col.grid(row= 0, column= 2, sticky= "nsew", padx= (6, 0), pady= 4)

        tk.Label(col, text= "SYSTEM METRICS", font= ("Segoe UI", 8, "bold"),
                 bg= BG_DARK, fg= TEXT_MUTED).pack(anchor= "w", pady= (4, 6))

        ig= tk.Frame(col, bg= BG_DARK)
        ig.pack(fill= "x")
        ig.columnconfigure(0, weight= 1)
        ig.columnconfigure(1, weight= 1)

        self._card_procs= StatCard(ig, "Processes", "⚙", ACCENT_CYAN)
        self._card_threads= StatCard(ig, "Threads", "🔗", ACCENT_PURPLE)
        self._card_ctx= StatCard(ig, "Ctx Switches", "↔", ACCENT_ORANGE)
        self._card_net_lat= StatCard(ig, "Net Latency", "🌐", ACCENT_BLUE)
        self._card_disk= StatCard(ig, "Disk I/O", "💾", ACCENT_GREEN)
        self._card_pwr= StatCard(ig, "Power Est.", "⚡", ACCENT_ORANGE)

        for i, card in enumerate((self._card_procs, self._card_threads,
                                   self._card_ctx,   self._card_net_lat,
                                   self._card_disk,  self._card_pwr)):
            r, c= divmod(i, 2)
            card.grid(row= r, column= c, padx= 2, pady= 2, sticky= "ew")

        # AI feature scores — shows the 3 highest importance model features
        tk.Label(col, text= "AI FEATURE SCORES", font= ("Segoe UI", 8, "bold"),
                 bg= BG_DARK, fg= TEXT_MUTED).pack(anchor= "w", pady= (12, 6))

        feat_frame= tk.Frame(col, bg= BG_CARD, padx= 12, pady= 8)
        feat_frame.pack(fill= "x")

        self._feat_bars= {}
        for label, key, max_v in [
            ("Thermal Load", "thermal_load", 100),
            ("Stress Score", "stress_score", 1),
            ("CPU × MEM", "cpu_mem_product", 10000),
        ]:
            row= tk.Frame(feat_frame, bg= BG_CARD)
            row.pack(fill= "x", pady= 3)
            tk.Label(row, text= label, font= ("Segoe UI", 8),
                     bg= BG_CARD, fg= TEXT_SECONDARY,
                     width= 12, anchor= "w").pack(side= "left")
            bar_bg= tk.Frame(row, bg= BG_DARK, height= 6, width= 120)
            bar_bg.pack(side= "left", padx= 6)
            bar_bg.pack_propagate(False)
            bar_fill= tk.Frame(bar_bg, bg= ACCENT_BLUE, height= 6)
            bar_fill.place(x= 0, y= 0, height= 6)
            val_lbl= tk.Label(row, text= "—", font= ("Segoe UI", 8),
                               bg= BG_CARD, fg= TEXT_MUTED, width= 7)
            val_lbl.pack(side= "left")
            self._feat_bars[key] = (bar_bg, bar_fill, val_lbl, max_v)
        # Log
        tk.Label(col, text= "EVENT LOG", font= ("Segoe UI", 8, "bold"),
                 bg= BG_DARK, fg= TEXT_MUTED).pack(anchor= "w", pady= (12, 4))
        self.log_panel= LogPanel(col)
        self.log_panel.pack(fill= "both", expand= True)

    # ── feature bar update ────────────────────────────────────────────
    def _update_feat_bars(self, m):
        for key, (bg_f, fill, lbl, max_v) in self._feat_bars.items():
            val= m.get(key, 0)
            pct= min(val / max_v, 1.0)
            w= max(1, int(pct * (bg_f.winfo_width() or 120)))
            clr= ACCENT_RED if pct > 0.85 else ACCENT_ORANGE if pct > 0.65 else ACCENT_GREEN
            fill.config(bg= clr)
            fill.place(x= 0, y= 0, height= 6, width= w)
            lbl.config(text= f"{val:.2f}" if max_v <= 1 else f"{val:.0f}")

    # ── background thread ─────────────────────────────────────────────
    def _start_collector_thread(self):
        def worker():
            while self._running:
                try:
                    metrics= self._collector.collect()
                    is_anom, conf= predict_anomaly(metrics)
                    self._data_queue.put((metrics, is_anom, conf))
                except Exception as e:
                    print(f"[Collector Error] {e}")
                time.sleep(UPDATE_MS / 1000)
        threading.Thread(target= worker, daemon= True).start()

    # ── UI update loop ────────────────────────────────────────────────
    def _schedule_update(self):
        self._process_queue()
        self._clock_lbl.config(text= time.strftime("%A, %d %b %Y %H:%M:%S"))
        if self._running:
            self.after(UPDATE_MS, self._schedule_update)

    def _process_queue(self):
        try:
            while True:
                metrics, is_anom, conf= self._data_queue.get_nowait()
                self._apply_update(metrics, is_anom, conf)
        except queue.Empty:
            pass

    def _apply_update(self, m, is_anom, conf):
        self._total_samples+= 1
        if is_anom:
            self._total_anomalies+= 1
        # Gauges
        self.gauge_cpu.update(m["cpu_utilization"])
        self.gauge_mem.update(m["memory_usage"])
        self.gauge_temp.update(m["temperature"])
        # Sparklines
        self.chart_cpu.push(m["cpu_utilization"])
        self.chart_mem.push(m["memory_usage"])
        self.chart_temp.push(m["temperature"])
        self.chart_net.push(m["network_latency"])
        self.chart_disk.push(m["disk_io"])
        self.chart_stress.push(m["stress_score"])
        # Probability chart
        self._update_probability_chart(conf, is_anom)

        # Prediction card
        if is_anom:
            self._pred_icon .config(text= "!",        fg= ACCENT_RED)
            self._pred_label.config(text= "ABNORMAL MODE",  fg= ACCENT_RED)
            self._live_indicator.config(bg= ACCENT_RED, text= "! ANOMALY MODE")
        else:
            self._pred_icon .config(text= "◉",         fg= ACCENT_GREEN)
            self._pred_label.config(text= "NORMAL MODE",    fg= ACCENT_GREEN)
            self._live_indicator.config(bg= ACCENT_GREEN, text= "● LIVE")

        self._conf_label.config(
            text= f"Confidence: {conf*100:.1f}%  "
                 f"({'ANOMALY MODE' if is_anom else 'NORMAL MODE'})")

        # Session stats
        rate= self._total_anomalies / self._total_samples * 100
        upt_min= m["uptime"]

        self._card_samples.update(str(self._total_samples))
        self._card_anomalies.update(str(self._total_anomalies),
                                    ACCENT_RED if self._total_anomalies else ACCENT_GREEN)
        self._card_rate.update(f"{rate:.1f}%",
                                    ACCENT_RED if rate > 5 else ACCENT_GREEN)
        self._card_uptime.update(f"{int(upt_min//60)}h {int(upt_min%60)}m")
        self._card_procs.update(str(int(m["process_count"])))
        self._card_threads.update(str(int(m["thread_count"])))
        self._card_ctx.update(str(int(m["context_switches"])))
        self._card_net_lat.update(f"{m['network_latency']:.1f} ms")
        self._card_disk.update(f"{m['disk_io']:.1f} MB/s")
        self._card_pwr.update(f"{m['power_consumption']:.0f} W")
        self._update_feat_bars(m)

        # Log
        if is_anom:
            self.log_panel.log(
                f"ANOMALY MODE  cpu= {m['cpu_utilization']:.1f}%  "
                f"mem= {m['memory_usage']:.1f}%  "
                f"temp= {m['temperature']:.1f}°C  "
                f"conf= {conf*100:.1f}%", "anom")
        elif m["cpu_utilization"] > 75 or m["memory_usage"] > 80:
            self.log_panel.log(
                f"WARNING  cpu= {m['cpu_utilization']:.1f}%  "
                f"mem= {m['memory_usage']:.1f}%", "warn")
        elif self._total_samples % 10 == 0:
            self.log_panel.log(
                f"OK  stress= {m['stress_score']:.3f}  "
                f"thermal= {m['thermal_load']:.1f}", "ok")

        # Popup alert (30s cooldown)
        if is_anom and self._alert_cooldown <= 0:
            self._alert_cooldown= 30
            self.after(100, lambda: AnomalyAlert(self, m))
        else:
            self._alert_cooldown= max(0, self._alert_cooldown - 1)

    def _on_close(self):
        self._running= False
        plt.close("all")
        self.destroy()

if __name__ == "__main__":
    app= ServerMonitorApp()
    app.mainloop()