#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import osmosdr
from argparse import ArgumentParser
from gnuradio import blocks
from gnuradio import gr

"""
Uses the osmosdr sink block to read the output of an AirspyHF+ device, reads a specified amount of samples
(time in seconds * sample rate) and writes the output in a specified file.

Based on a GNU Radio Python Flow Graph
Requires GNU Radio version: 3.7.x or 3.8.x
Author: Luc Wachter
"""


class SimplestCapture(gr.top_block):
    def __init__(self, capture_length, sample_rate, output):
        gr.top_block.__init__(self, "Simplest Capture")

        # Variables
        seconds = capture_length

        # Blocks
        source = osmosdr.source(
            args="numchan=" + str(1) + " " + 'airspyhf=0'
        )
        source.set_sample_rate(sample_rate)
        source.set_center_freq(13.56e6, 0)
        source.set_freq_corr(0, 0)
        source.set_gain(16, 0)
        source.set_if_gain(16, 0)
        source.set_bb_gain(16, 0)
        source.set_antenna('', 0)
        source.set_bandwidth(0, 0)

        head = blocks.head(gr.sizeof_gr_complex*1, int(seconds * sample_rate))

        sink = blocks.file_sink(gr.sizeof_gr_complex*1, output, False)
        sink.set_unbuffered(False)

        # Connections
        self.connect((source, 0), (head, 0))
        self.connect((head, 0), (sink, 0))


def main():
    parser = ArgumentParser(description="GNURadio-based capture script using airspyhf+")
    parser.add_argument("--time", help="The capture length in seconds", default=5, type=int)
    parser.add_argument("--samplerate", help="The (theoretical) number of to capture samples per second",
                        default=768000, type=int)
    parser.add_argument("path", help="The path to the output file relative to the script's location")

    args = parser.parse_args()
    SECONDS = args.time
    SAMPLE_RATE = args.samplerate
    PATH = os.path.join(os.getcwd(), args.path)

    tb = SimplestCapture(SECONDS, SAMPLE_RATE, PATH)
    tb.start()
    time.sleep(SECONDS + 2)
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
