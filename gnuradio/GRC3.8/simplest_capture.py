#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Simplest Capture
# Author: Luc Wachter
# GNU Radio version: 3.8.1.0

from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import osmosdr
import time


class simplest_capture(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Simplest Capture")

        ##################################################
        # Variables
        ##################################################
        self.seconds = seconds = 5
        self.samp_rate = samp_rate = 2e6

        ##################################################
        # Blocks
        ##################################################
        self.osmosdr_source_0 = osmosdr.source(
            args="numchan=" + str(1) + " " + 'driver=lime,soapy=0'
        )
        self.osmosdr_source_0.set_sample_rate(samp_rate)
        self.osmosdr_source_0.set_center_freq(13.56e6, 0)
        self.osmosdr_source_0.set_freq_corr(0, 0)
        self.osmosdr_source_0.set_gain(16, 0)
        self.osmosdr_source_0.set_if_gain(16, 0)
        self.osmosdr_source_0.set_bb_gain(16, 0)
        self.osmosdr_source_0.set_antenna('', 0)
        self.osmosdr_source_0.set_bandwidth(0, 0)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, int(seconds * samp_rate))
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, '/home/luc/HEIG/TB/test.bin', False)
        self.blocks_file_sink_0.set_unbuffered(False)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_head_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.osmosdr_source_0, 0), (self.blocks_head_0, 0))


    def get_seconds(self):
        return self.seconds

    def set_seconds(self, seconds):
        self.seconds = seconds
        self.blocks_head_0.set_length(int(self.seconds * self.samp_rate))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_head_0.set_length(int(self.seconds * self.samp_rate))
        self.osmosdr_source_0.set_sample_rate(self.samp_rate)





def main(top_block_cls=simplest_capture, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    tb.wait()


if __name__ == '__main__':
    main()
