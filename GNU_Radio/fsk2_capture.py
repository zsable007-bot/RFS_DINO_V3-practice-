#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: 2FSK capture
# Author: luojf
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import analog
from gnuradio import blocks
from gnuradio import channels
from gnuradio.filter import firdes
from gnuradio import digital
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import sip
import threading



class fsk2_capture(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "2FSK capture", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("2FSK capture")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "fsk2_capture")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.signal_dbfs = signal_dbfs = -27
        self.adc_bits = adc_bits = 12
        self.symbol_rate = symbol_rate = 1000
        self.snr_db = snr_db = 20
        self.signal_amp = signal_amp = 10**(signal_dbfs/20)
        self.samp_rate = samp_rate = 1000000
        self.rx_gain_db = rx_gain_db = 0
        self.capture_seconds = capture_seconds = 2
        self.adc_scale = adc_scale = 2**(adc_bits-1)
        self.slot_samples = slot_samples = int(samp_rate*0.1)
        self.samples_per_symbol = samples_per_symbol = int(samp_rate/symbol_rate)
        self.rx_gain = rx_gain = 10**(rx_gain_db/20)
        self.ramp_samples = ramp_samples = int(samp_rate*0.002)
        self.noise_voltage = noise_voltage = signal_amp*10**(-snr_db/20)
        self.iq_phase_deg = iq_phase_deg = 0
        self.iq_mag_db = iq_mag_db = 0
        self.fsk_deviation_hz = fsk_deviation_hz = 30000
        self.freq_offset_hz = freq_offset_hz = 5000
        self.display_decim = display_decim = 1000
        self.dc_q = dc_q = 0
        self.dc_i = dc_i = 0
        self.center_hz = center_hz = 50000
        self.capture_samples = capture_samples = int(samp_rate*capture_seconds)
        self.adc_max = adc_max = 1.0-1.0/adc_scale

        ##################################################
        # Blocks
        ##################################################

        self.qtgui_waterfall_sink_x_0 = qtgui.waterfall_sink_c(
            512, #size
            window.WIN_HANN, #wintype
            0, #fc
            samp_rate, #bw
            "CW - Waterfall", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_waterfall_sink_x_0.set_update_time(0.10)
        self.qtgui_waterfall_sink_x_0.enable_grid(False)
        self.qtgui_waterfall_sink_x_0.enable_axis_labels(True)



        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        colors = [0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_waterfall_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_waterfall_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_waterfall_sink_x_0.set_color_map(i, colors[i])
            self.qtgui_waterfall_sink_x_0.set_line_alpha(i, alphas[i])

        self.qtgui_waterfall_sink_x_0.set_intensity_range(-100, -20)

        self._qtgui_waterfall_sink_x_0_win = sip.wrapinstance(self.qtgui_waterfall_sink_x_0.qwidget(), Qt.QWidget)

        self.top_grid_layout.addWidget(self._qtgui_waterfall_sink_x_0_win, 2, 0, 1, 1)
        for r in range(2, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_time_sink_x_1 = qtgui.time_sink_f(
            1000, #size
            samp_rate/display_decim, #samp_rate
            "Burst envelope", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_1.set_update_time(0.10)
        self.qtgui_time_sink_x_1.set_y_axis(-0.1, 1.1)

        self.qtgui_time_sink_x_1.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_1.enable_tags(True)
        self.qtgui_time_sink_x_1.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_1.enable_autoscale(False)
        self.qtgui_time_sink_x_1.enable_grid(False)
        self.qtgui_time_sink_x_1.enable_axis_labels(True)
        self.qtgui_time_sink_x_1.enable_control_panel(False)
        self.qtgui_time_sink_x_1.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_time_sink_x_1.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_x_1.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_1.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_1.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_1.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_1.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_1.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_1_win = sip.wrapinstance(self.qtgui_time_sink_x_1.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_time_sink_x_1_win, 3, 0, 1, 1)
        for r in range(3, 4):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_time_sink_x_0 = qtgui.time_sink_c(
            200, #size
            samp_rate, #samp_rate
            "CW - I and Q", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0.set_y_axis(-0.12, 0.12)

        self.qtgui_time_sink_x_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0.enable_tags(True)
        self.qtgui_time_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0.enable_autoscale(False)
        self.qtgui_time_sink_x_0.enable_grid(False)
        self.qtgui_time_sink_x_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink_x_0.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_0.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_time_sink_x_0_win, 0, 0, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            1024, #size
            window.WIN_HANN, #wintype
            0, #fc
            samp_rate, #bw
            "CW - Spectrum", #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis((-140), 10)
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(False)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_freq_sink_x_0_win, 1, 0, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.digital_glfsr_source_x_0 = digital.glfsr_source_b(15, True, 0, 11)
        self.channels_iqbal_gen_0 = channels.iqbal_gen(iq_mag_db, iq_phase_deg, 1)
        self.channels_channel_model_0 = channels.channel_model(
            noise_voltage=noise_voltage,
            frequency_offset=(freq_offset_hz/samp_rate),
            epsilon=1.0,
            taps=[1+0j],
            noise_seed=20261004,
            block_tags=False)
        self.blocks_vector_source_x_0 = blocks.vector_source_f([0,1,0,0,1,0,0,1,0,0], True, 1, [])
        self.blocks_uchar_to_float_0 = blocks.uchar_to_float()
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_short_to_float_0_0 = blocks.short_to_float(1, adc_scale)
        self.blocks_short_to_float_0 = blocks.short_to_float(1, adc_scale)
        self.blocks_repeat_1 = blocks.repeat(gr.sizeof_float*1, samples_per_symbol)
        self.blocks_repeat_0 = blocks.repeat(gr.sizeof_float*1, slot_samples)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_const_vxx_1_0 = blocks.multiply_const_cc(signal_amp)
        self.blocks_multiply_const_vxx_1 = blocks.multiply_const_ff((2*fsk_deviation_hz))
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(rx_gain)
        self.blocks_moving_average_xx_0 = blocks.moving_average_ff(ramp_samples, (1.0/ramp_samples), 4000, 1)
        self.blocks_keep_one_in_n_0 = blocks.keep_one_in_n(gr.sizeof_float*1, display_decim)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, capture_samples)
        self.blocks_float_to_short_0_0 = blocks.float_to_short(1, adc_scale)
        self.blocks_float_to_short_0 = blocks.float_to_short(1, adc_scale)
        self.blocks_float_to_complex_1 = blocks.float_to_complex(1)
        self.blocks_float_to_complex_0 = blocks.float_to_complex(1)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, 'D:\\workspace\\preparation\\dinov3_11\\data\\2FSK\\2FSK_0001.cf32', False)
        self.blocks_file_sink_0.set_unbuffered(True)
        self.blocks_complex_to_float_0 = blocks.complex_to_float(1)
        self.blocks_add_const_vxx_1 = blocks.add_const_ff((center_hz-fsk_deviation_hz))
        self.blocks_add_const_vxx_0 = blocks.add_const_cc(complex(dc_i, dc_q))
        self.analog_rail_ff_0_0 = analog.rail_ff((-1), adc_max)
        self.analog_rail_ff_0 = analog.rail_ff((-1), adc_max)
        self.analog_frequency_modulator_fc_0 = analog.frequency_modulator_fc((2*3.141592653589793/samp_rate))


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_frequency_modulator_fc_0, 0), (self.blocks_multiply_const_vxx_1_0, 0))
        self.connect((self.analog_rail_ff_0, 0), (self.blocks_float_to_short_0, 0))
        self.connect((self.analog_rail_ff_0_0, 0), (self.blocks_float_to_short_0_0, 0))
        self.connect((self.blocks_add_const_vxx_0, 0), (self.blocks_complex_to_float_0, 0))
        self.connect((self.blocks_add_const_vxx_1, 0), (self.analog_frequency_modulator_fc_0, 0))
        self.connect((self.blocks_complex_to_float_0, 0), (self.analog_rail_ff_0, 0))
        self.connect((self.blocks_complex_to_float_0, 1), (self.analog_rail_ff_0_0, 0))
        self.connect((self.blocks_float_to_complex_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.blocks_float_to_complex_1, 0), (self.blocks_throttle_0, 0))
        self.connect((self.blocks_float_to_short_0, 0), (self.blocks_short_to_float_0, 0))
        self.connect((self.blocks_float_to_short_0_0, 0), (self.blocks_short_to_float_0_0, 0))
        self.connect((self.blocks_head_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.blocks_head_0, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.blocks_head_0, 0), (self.qtgui_time_sink_x_0, 0))
        self.connect((self.blocks_head_0, 0), (self.qtgui_waterfall_sink_x_0, 0))
        self.connect((self.blocks_keep_one_in_n_0, 0), (self.qtgui_time_sink_x_1, 0))
        self.connect((self.blocks_moving_average_xx_0, 0), (self.blocks_float_to_complex_0, 0))
        self.connect((self.blocks_moving_average_xx_0, 0), (self.blocks_keep_one_in_n_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.channels_iqbal_gen_0, 0))
        self.connect((self.blocks_multiply_const_vxx_1, 0), (self.blocks_add_const_vxx_1, 0))
        self.connect((self.blocks_multiply_const_vxx_1_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.channels_channel_model_0, 0))
        self.connect((self.blocks_repeat_0, 0), (self.blocks_moving_average_xx_0, 0))
        self.connect((self.blocks_repeat_1, 0), (self.blocks_multiply_const_vxx_1, 0))
        self.connect((self.blocks_short_to_float_0, 0), (self.blocks_float_to_complex_1, 0))
        self.connect((self.blocks_short_to_float_0_0, 0), (self.blocks_float_to_complex_1, 1))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_head_0, 0))
        self.connect((self.blocks_uchar_to_float_0, 0), (self.blocks_repeat_1, 0))
        self.connect((self.blocks_vector_source_x_0, 0), (self.blocks_repeat_0, 0))
        self.connect((self.channels_channel_model_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.channels_iqbal_gen_0, 0), (self.blocks_add_const_vxx_0, 0))
        self.connect((self.digital_glfsr_source_x_0, 0), (self.blocks_uchar_to_float_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "fsk2_capture")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_signal_dbfs(self):
        return self.signal_dbfs

    def set_signal_dbfs(self, signal_dbfs):
        self.signal_dbfs = signal_dbfs
        self.set_signal_amp(10**(self.signal_dbfs/20))

    def get_adc_bits(self):
        return self.adc_bits

    def set_adc_bits(self, adc_bits):
        self.adc_bits = adc_bits
        self.set_adc_scale(2**(self.adc_bits-1))

    def get_symbol_rate(self):
        return self.symbol_rate

    def set_symbol_rate(self, symbol_rate):
        self.symbol_rate = symbol_rate
        self.set_samples_per_symbol(int(self.samp_rate/self.symbol_rate))

    def get_snr_db(self):
        return self.snr_db

    def set_snr_db(self, snr_db):
        self.snr_db = snr_db
        self.set_noise_voltage(self.signal_amp*10**(-self.snr_db/20))

    def get_signal_amp(self):
        return self.signal_amp

    def set_signal_amp(self, signal_amp):
        self.signal_amp = signal_amp
        self.set_noise_voltage(self.signal_amp*10**(-self.snr_db/20))
        self.blocks_multiply_const_vxx_1_0.set_k(self.signal_amp)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_capture_samples(int(self.samp_rate*self.capture_seconds))
        self.set_ramp_samples(int(self.samp_rate*0.002))
        self.set_samples_per_symbol(int(self.samp_rate/self.symbol_rate))
        self.set_slot_samples(int(self.samp_rate*0.1))
        self.analog_frequency_modulator_fc_0.set_sensitivity((2*3.141592653589793/self.samp_rate))
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)
        self.channels_channel_model_0.set_frequency_offset((self.freq_offset_hz/self.samp_rate))
        self.qtgui_freq_sink_x_0.set_frequency_range(0, self.samp_rate)
        self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)
        self.qtgui_time_sink_x_1.set_samp_rate(self.samp_rate/self.display_decim)
        self.qtgui_waterfall_sink_x_0.set_frequency_range(0, self.samp_rate)

    def get_rx_gain_db(self):
        return self.rx_gain_db

    def set_rx_gain_db(self, rx_gain_db):
        self.rx_gain_db = rx_gain_db
        self.set_rx_gain(10**(self.rx_gain_db/20))

    def get_capture_seconds(self):
        return self.capture_seconds

    def set_capture_seconds(self, capture_seconds):
        self.capture_seconds = capture_seconds
        self.set_capture_samples(int(self.samp_rate*self.capture_seconds))

    def get_adc_scale(self):
        return self.adc_scale

    def set_adc_scale(self, adc_scale):
        self.adc_scale = adc_scale
        self.set_adc_max(1.0-1.0/self.adc_scale)
        self.blocks_float_to_short_0.set_scale(self.adc_scale)
        self.blocks_float_to_short_0_0.set_scale(self.adc_scale)
        self.blocks_short_to_float_0.set_scale(self.adc_scale)
        self.blocks_short_to_float_0_0.set_scale(self.adc_scale)

    def get_slot_samples(self):
        return self.slot_samples

    def set_slot_samples(self, slot_samples):
        self.slot_samples = slot_samples
        self.blocks_repeat_0.set_interpolation(self.slot_samples)

    def get_samples_per_symbol(self):
        return self.samples_per_symbol

    def set_samples_per_symbol(self, samples_per_symbol):
        self.samples_per_symbol = samples_per_symbol
        self.blocks_repeat_1.set_interpolation(self.samples_per_symbol)

    def get_rx_gain(self):
        return self.rx_gain

    def set_rx_gain(self, rx_gain):
        self.rx_gain = rx_gain
        self.blocks_multiply_const_vxx_0.set_k(self.rx_gain)

    def get_ramp_samples(self):
        return self.ramp_samples

    def set_ramp_samples(self, ramp_samples):
        self.ramp_samples = ramp_samples
        self.blocks_moving_average_xx_0.set_length_and_scale(self.ramp_samples, (1.0/self.ramp_samples))

    def get_noise_voltage(self):
        return self.noise_voltage

    def set_noise_voltage(self, noise_voltage):
        self.noise_voltage = noise_voltage
        self.channels_channel_model_0.set_noise_voltage(self.noise_voltage)

    def get_iq_phase_deg(self):
        return self.iq_phase_deg

    def set_iq_phase_deg(self, iq_phase_deg):
        self.iq_phase_deg = iq_phase_deg
        self.channels_iqbal_gen_0.set_phase(self.iq_phase_deg)

    def get_iq_mag_db(self):
        return self.iq_mag_db

    def set_iq_mag_db(self, iq_mag_db):
        self.iq_mag_db = iq_mag_db
        self.channels_iqbal_gen_0.set_magnitude(self.iq_mag_db)

    def get_fsk_deviation_hz(self):
        return self.fsk_deviation_hz

    def set_fsk_deviation_hz(self, fsk_deviation_hz):
        self.fsk_deviation_hz = fsk_deviation_hz
        self.blocks_add_const_vxx_1.set_k((self.center_hz-self.fsk_deviation_hz))
        self.blocks_multiply_const_vxx_1.set_k((2*self.fsk_deviation_hz))

    def get_freq_offset_hz(self):
        return self.freq_offset_hz

    def set_freq_offset_hz(self, freq_offset_hz):
        self.freq_offset_hz = freq_offset_hz
        self.channels_channel_model_0.set_frequency_offset((self.freq_offset_hz/self.samp_rate))

    def get_display_decim(self):
        return self.display_decim

    def set_display_decim(self, display_decim):
        self.display_decim = display_decim
        self.blocks_keep_one_in_n_0.set_n(self.display_decim)
        self.qtgui_time_sink_x_1.set_samp_rate(self.samp_rate/self.display_decim)

    def get_dc_q(self):
        return self.dc_q

    def set_dc_q(self, dc_q):
        self.dc_q = dc_q
        self.blocks_add_const_vxx_0.set_k(complex(self.dc_i, self.dc_q))

    def get_dc_i(self):
        return self.dc_i

    def set_dc_i(self, dc_i):
        self.dc_i = dc_i
        self.blocks_add_const_vxx_0.set_k(complex(self.dc_i, self.dc_q))

    def get_center_hz(self):
        return self.center_hz

    def set_center_hz(self, center_hz):
        self.center_hz = center_hz
        self.blocks_add_const_vxx_1.set_k((self.center_hz-self.fsk_deviation_hz))

    def get_capture_samples(self):
        return self.capture_samples

    def set_capture_samples(self, capture_samples):
        self.capture_samples = capture_samples
        self.blocks_head_0.set_length(self.capture_samples)

    def get_adc_max(self):
        return self.adc_max

    def set_adc_max(self, adc_max):
        self.adc_max = adc_max
        self.analog_rail_ff_0.set_hi(self.adc_max)
        self.analog_rail_ff_0_0.set_hi(self.adc_max)




def main(top_block_cls=fsk2_capture, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
