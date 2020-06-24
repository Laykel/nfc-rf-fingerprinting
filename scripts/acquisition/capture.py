#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Based on a GNU Radio Python Flow Graph
# Author: Luc Wachter
# GNU Radio version: 3.8.1.0

import os
import time
import signal
import osmosdr
from argparse import ArgumentParser
from gnuradio import blocks
from gnuradio import gr


class simplest_capture(gr.top_block):
    def __init__(self, capture_length, output):
        gr.top_block.__init__(self, "Simplest Capture")

        # Variables
        seconds = capture_length
        samp_rate = 768e3

        # Blocks
        source = osmosdr.source(
            args="numchan=" + str(1) + " " + 'airspyhf=0'
        )
        source.set_sample_rate(samp_rate)
        source.set_center_freq(13.56e6, 0)
        source.set_freq_corr(0, 0)
        source.set_gain(16, 0)
        source.set_if_gain(16, 0)
        source.set_bb_gain(16, 0)
        source.set_antenna('', 0)
        source.set_bandwidth(0, 0)

        head = blocks.head(gr.sizeof_gr_complex*1, int(seconds * samp_rate))

        sink = blocks.file_sink(gr.sizeof_gr_complex*1, output, False)
        sink.set_unbuffered(False)

        # Connections
        self.connect((source, 0), (head, 0))
        self.connect((head, 0), (sink, 0))


def main():
    parser = ArgumentParser(description="GNURadio-based capture script using airspyhf+")
    parser.add_argument("--time", help="The capture length in seconds", default=5, type=int)
    parser.add_argument("--path", help="The path to the output file relative to the script's location")

    args = parser.parse_args()
    SECONDS = args.time
    PATH = os.path.join(os.getcwd(), args.path)

    tb = simplest_capture(SECONDS, PATH)
    tb.start()
    time.sleep(SECONDS + 2)
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
