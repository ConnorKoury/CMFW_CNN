import sys
import numpy as np
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem
)
from PySide6.QtCore import QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

C = 3e8 # speed of light
# profile
NUM_RX_DEFAULT = 4
NUM_SAMPLES_DEFAULT = 256
NUM_CHIRPS_DEFAULT = 128
NUM_FRAMES_DEFAULT = 50

FS_DEFAULT = 10e6 # Sample Rate
SLOPE_DEFAULT = 66e12 # Frequency Slope (MHz/us)
FC_DEFAULT = 77e9 # Freq (GHz)
CHIRP_PERIOD_DEFAULT = (100 + 60) * 1e-6


# Processing functions
def read_dca1000_complex(file_path, num_rx, num_samples, num_chirps, num_frames = None):
    """Read DCA1000 binary and return data[frames, rx, chirp, sample]"""

    raw = np.fromfile(file_path, dtype=np.int16)

    iq = raw.reshape(-1, 2)
    complex_samples = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)

    samples_per_frame = num_rx * num_chirps * num_samples

    if num_frames is None:
        num_frames = complex_samples.size // samples_per_frame

    data = complex_samples.reshape(num_frames, num_chirps, num_samples, num_rx)
    data = np.transpose(data, (0, 3, 1, 2)) # [frame, rx, chirp, sample]
    return data


def compute_axes(num_range_bins, num_chirps, fs, slope, chirp_period, f_c, n_fft_range_full):
    # Range axis
    range_res = C * fs / (2 * slope * n_fft_range_full)
    ranges = np.arange(num_range_bins) * range_res

    # Doppler axis
    prf = 1.0 / chirp_period
    lam = C / f_c
    v_res = lam * prf / (2 * num_chirps)
    velocities = (np.arange(-num_chirps // 2, num_chirps // 2)) * v_res
    return ranges, velocities


def compute_range_doppler(frame_cube, rx_idx = 0, clutter_removal = True):
    """ Returns rd_power, rd_db """
    rx_data = frame_cube[rx_idx] # [chirp, sample]
    num_chirps, num_samples = rx_data.shape

    # static clutter removal before Doppler FFT
    if clutter_removal:
        # remove mean across chirps for each range sample
        rx_data = rx_data - np.mean(rx_data, axis=0, keepdims=True)

    # Windowing
    win_range = np.hanning(num_samples)
    win_dopp = np.hanning(num_chirps)
    data_win = (rx_data * win_range[None, :]) * win_dopp[:, None]

    # Range FFT
    range_fft = np.fft.fft(data_win, axis=1)

    # Keep only positive beat frequencies on range
    range_fft = range_fft[:, :num_samples // 2]

    # Doppler FFT
    rd = np.fft.fft(range_fft, axis=0)
    rd = np.fft.fftshift(rd, axes=0)  # center zero Doppler

    rd_power = np.abs(rd) ** 2
    rd_db = 10 * np.log10(rd_power + 1e-12)
    return rd_power, rd_db


def ca_cfar_2d(rd_power, num_train=(8, 4), num_guard=(4, 2), pfa=1e-3):
    """ 2D CA-CFAR over rd_power (doppler x range) and returns detections mask """
    n_dopp, n_range = rd_power.shape
    Tr, Td = num_train
    Gr, Gd = num_guard

    num_train_cells = ((2 * Tr + 2 * Gr + 1) * (2 * Td + 2 * Gd + 1) - (2 * Gr + 1) * (2 * Gd + 1))
    alpha = num_train_cells * (pfa ** (-1 / num_train_cells) - 1)

    detections = np.zeros_like(rd_power, dtype=bool)
    threshold_map = np.zeros_like(rd_power)

    for i in range(Tr + Gr, n_dopp - (Tr + Gr)):
        for j in range(Td + Gd, n_range - (Td + Gd)):
            i0 = i - (Tr + Gr)
            i1 = i + Tr + Gr + 1
            j0 = j - (Td + Gd)
            j1 = j + Td + Gd + 1
            window = rd_power[i0:i1, j0:j1]

            gi0 = i - Gr
            gi1 = i + Gr + 1
            gj0 = j - Gd
            gj1 = j + Gd + 1
            guard_cut = rd_power[gi0:gi1, gj0:gj1]

            noise_sum = window.sum() - guard_cut.sum()
            noise_mean = noise_sum / num_train_cells

            threshold = alpha * noise_mean
            threshold_map[i, j] = threshold

            if rd_power[i, j] > threshold:
                detections[i, j] = True

    return detections


# GUI
class RadarWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AWR2243 Range–Doppler / CFAR Viewer")

        self.data = None
        self.ranges = None
        self.velocities = None

        # playback state
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.playing = False
        self.play_interval_ms = 20 # change speed here

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left side controls
        controls = QVBoxLayout()
        layout.addLayout(controls, 0)

        # Files
        file_group = QGroupBox("Data File")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)

        self.file_label = QLabel("No file selected")
        browse_btn = QPushButton("Browse .bin")
        browse_btn.clicked.connect(self.browse_file)

        file_layout.addWidget(self.file_label)
        file_layout.addWidget(browse_btn)
        controls.addWidget(file_group)

        # Configs
        cfg_group = QGroupBox("Radar / Data Config")
        cfg_form = QFormLayout()
        cfg_group.setLayout(cfg_form)

        self.spin_rx = QSpinBox()
        self.spin_rx.setRange(1, 16)
        self.spin_rx.setValue(NUM_RX_DEFAULT)

        self.spin_samples = QSpinBox()
        self.spin_samples.setRange(16, 4096)
        self.spin_samples.setValue(NUM_SAMPLES_DEFAULT)

        self.spin_chirps = QSpinBox()
        self.spin_chirps.setRange(16, 4096)
        self.spin_chirps.setValue(NUM_CHIRPS_DEFAULT)

        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(0, 0)
        self.spin_frame.setValue(0)

        self.spin_rx_index = QSpinBox()
        self.spin_rx_index.setRange(0, NUM_RX_DEFAULT - 1)
        self.spin_rx_index.setValue(0)

        cfg_form.addRow("NUM_RX", self.spin_rx)
        cfg_form.addRow("NUM_SAMPLES", self.spin_samples)
        cfg_form.addRow("NUM_CHIRPS", self.spin_chirps)
        cfg_form.addRow("Frame index", self.spin_frame)
        cfg_form.addRow("Rx index", self.spin_rx_index)

        controls.addWidget(cfg_group)

        # Radar parameters
        param_group = QGroupBox("FMCW Parameters")
        param_form = QFormLayout()
        param_group.setLayout(param_form)

        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setDecimals(0)
        self.fs_spin.setRange(1e3, 200e6)
        self.fs_spin.setValue(FS_DEFAULT)

        self.slope_spin = QDoubleSpinBox()
        self.slope_spin.setDecimals(0)
        self.slope_spin.setRange(1e9, 200e12)
        self.slope_spin.setValue(SLOPE_DEFAULT)

        self.tchirp_spin = QDoubleSpinBox()
        self.tchirp_spin.setDecimals(8)
        self.tchirp_spin.setRange(1e-6, 1e-2)
        self.tchirp_spin.setValue(CHIRP_PERIOD_DEFAULT)

        self.fc_spin = QDoubleSpinBox()
        self.fc_spin.setDecimals(0)
        self.fc_spin.setRange(60e9, 90e9)
        self.fc_spin.setValue(FC_DEFAULT)

        param_form.addRow("Fs [Hz]", self.fs_spin)
        param_form.addRow("Slope [Hz/s]", self.slope_spin)
        param_form.addRow("T_chirp [s]", self.tchirp_spin)
        param_form.addRow("f_c [Hz]", self.fc_spin)

        controls.addWidget(param_group)

        # CFAR
        cfar_group = QGroupBox("CFAR Parameters")
        cfar_form = QFormLayout()
        cfar_group.setLayout(cfar_form)

        self.train_dopp_spin = QSpinBox()
        self.train_dopp_spin.setRange(1, 64)
        self.train_dopp_spin.setValue(8)

        self.train_range_spin = QSpinBox()
        self.train_range_spin.setRange(1, 64)
        self.train_range_spin.setValue(4)

        self.guard_dopp_spin = QSpinBox()
        self.guard_dopp_spin.setRange(1, 32)
        self.guard_dopp_spin.setValue(4)

        self.guard_range_spin = QSpinBox()
        self.guard_range_spin.setRange(1, 32)
        self.guard_range_spin.setValue(2)

        self.pfa_spin = QDoubleSpinBox()
        self.pfa_spin.setDecimals(6)
        self.pfa_spin.setRange(1e-6, 1e-1)
        self.pfa_spin.setSingleStep(1e-3)
        self.pfa_spin.setValue(1e-3)

        cfar_form.addRow("Train (Dopp)", self.train_dopp_spin)
        cfar_form.addRow("Train (Range)", self.train_range_spin)
        cfar_form.addRow("Guard (Dopp)", self.guard_dopp_spin)
        cfar_form.addRow("Guard (Range)", self.guard_range_spin)
        cfar_form.addRow("Pfa", self.pfa_spin)

        controls.addWidget(cfar_group)

        # Process and playback buttons
        btn_row = QHBoxLayout()
        process_btn = QPushButton("Compute RD + CFAR")
        process_btn.clicked.connect(self.process_current)
        btn_row.addWidget(process_btn)

        self.next_btn = QPushButton("Next frame")
        self.next_btn.clicked.connect(self.next_frame) 
        btn_row.addWidget(self.next_btn)

        self.play_btn = QPushButton("Play frames")
        self.play_btn.clicked.connect(self.toggle_play)
        btn_row.addWidget(self.play_btn)

        controls.addLayout(btn_row)

        # Table of detections
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Range [m]", "Vel [m/s]", "Power [dB]", "Indices"])
        controls.addWidget(self.table, 1)

        # Right side plot
        self.fig = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, 1)

    # Slots / helpers
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open DCA1000 ADC Binary", "", "Binary Files (*.bin);;All Files (*)"
        )
        if not file_path:
            return
        self.file_label.setText(file_path)
        self.load_data(file_path)

    def load_data(self, file_path):
        num_rx = self.spin_rx.value()
        num_samples = self.spin_samples.value()
        num_chirps = self.spin_chirps.value()

        self.data = read_dca1000_complex(
            file_path,
            num_rx=num_rx,
            num_samples=num_samples,
            num_chirps=num_chirps,
            num_frames=None, # infer
        )

        frames, rx, chirps, samples = self.data.shape
        self.spin_frame.setRange(0, frames - 1)
        self.spin_rx_index.setRange(0, rx - 1)
        self.spin_frame.setValue(0)
        self.spin_rx_index.setValue(0)

    def process_current(self):
        if self.data is None:
            return

        frame_idx = self.spin_frame.value()
        rx_idx = self.spin_rx_index.value()

        frame_cube = self.data[frame_idx] # [rx, chirp, sample]

        rd_power, rd_db = compute_range_doppler(frame_cube, rx_idx=rx_idx, clutter_removal=True)

        fs = self.fs_spin.value()
        slope = self.slope_spin.value()
        t_chirp = self.tchirp_spin.value()
        fc = self.fc_spin.value()

        num_samples_full = frame_cube.shape[2]
        num_dopp_bins, num_rng_bins = rd_db.shape

        self.ranges, self.velocities = compute_axes(num_rng_bins, num_dopp_bins, fs, slope, t_chirp, fc, n_fft_range_full=num_samples_full)

        Tr = self.train_dopp_spin.value()
        Td = self.train_range_spin.value()
        Gr = self.guard_dopp_spin.value()
        Gd = self.guard_range_spin.value()
        pfa = self.pfa_spin.value()

        detections = ca_cfar_2d(rd_power, num_train=(Tr, Td), num_guard=(Gr, Gd), pfa=pfa)

        self.update_plot(rd_db, detections)
        self.update_table(rd_power, detections)

    #  playback helpers
    def next_frame(self):
        """Advance to next frame and re-process"""
        if self.data is None:
            return
        frames = self.data.shape[0]
        curr = self.spin_frame.value()
        new = (curr + 1) % frames
        self.spin_frame.setValue(new)
        self.process_current()

    def toggle_play(self):
        """Start/stop automatic frame stepping"""
        if self.data is None:
            return
        if self.playing:
            self.timer.stop()
            self.playing = False
            self.play_btn.setText("Play frames")
        else:
            self.timer.start(self.play_interval_ms)
            self.playing = True
            self.play_btn.setText("Stop")

    # plotting
    def update_plot(self, rd_db, detections):
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        rd_db_plot = rd_db.T
        det_idx = np.argwhere(detections)

        extent = [
            self.velocities[0], self.velocities[-1], # x-axis velocity
            self.ranges[0], self.ranges[-1],         # y-axis range
        ]

        im = ax.imshow(
            rd_db_plot,
            origin="lower",
            extent=extent,
            aspect="auto",
        )
        ax.set_xlabel("Velocity (m/s)")
        ax.set_ylabel("Range (m)")
        ax.set_title("Range–Doppler (dB) with CFAR Detections")
        self.fig.colorbar(im, ax=ax, label="Power (dB)")

        if det_idx.size > 0:
            dopp_idx = det_idx[:, 0]
            rng_idx = det_idx[:, 1]
            ax.scatter(
                self.velocities[dopp_idx],
                self.ranges[rng_idx],
                s=20,
                facecolors="none",
                edgecolors="white",
                linewidths=0.8,
            )

        self.canvas.draw()

    def update_table(self, rd_power, detections):
        det_idx = np.argwhere(detections)
        if det_idx.size == 0:
            self.table.setRowCount(0)
            return

        # Sort by power descending
        det_list = [
            (di, rj, rd_power[di, rj])
            for di, rj in det_idx
        ]
        det_list.sort(key=lambda x: x[2], reverse=True)

        top = det_list[:50] # show up to 50
        self.table.setRowCount(len(top))

        for row, (di, rj, p_lin) in enumerate(top):
            rng = self.ranges[rj]
            vel = self.velocities[di]
            p_db = 10 * np.log10(p_lin + 1e-12)
            idx_str = f"({di}, {rj})"

            self.table.setItem(row, 0, QTableWidgetItem(f"{rng:.2f}"))
            self.table.setItem(row, 1, QTableWidgetItem(f"{vel:.2f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{p_db:.1f}"))
            self.table.setItem(row, 3, QTableWidgetItem(idx_str))


def main():
    app = QApplication(sys.argv)
    win = RadarWindow()
    win.resize(1200, 700)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
