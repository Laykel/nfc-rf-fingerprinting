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
(time in seconds * sample rate) at a given frequency and writes the output in a specified file.

Based on a GNU Radio Python Flow Graph
Requires GNU Radio version: 3.7.x or 3.8.x
Author: Luc Wachter
"""


class SimplestCapture(gr.top_block):
    def __init__(self, capture_length, sample_rate, center_frequency, output):
        gr.top_block.__init__(self, "Simplest Capture")

        # Blocks
        source = osmosdr.source(
            args="numchan=" + str(1) + " " + 'airspyhf=0'
        )
        source.set_sample_rate(sample_rate)
        source.set_center_freq(center_frequency, 0)
        source.set_freq_corr(0, 0)
        source.set_gain(16, 0)
        source.set_if_gain(16, 0)
        source.set_bb_gain(16, 0)
        source.set_antenna('RX', 0)
        source.set_bandwidth(0, 0)
        # print(source.get_antennas())

        head = blocks.head(gr.sizeof_gr_complex, int(capture_length * sample_rate))

        sink = blocks.file_sink(gr.sizeof_gr_complex, output, False)
        sink.set_unbuffered(False)

        # Connections
        self.connect((source, 0), (head, 0))
        self.connect((head, 0), (sink, 0))


def main():
    # Manage program arguments
    parser = ArgumentParser(description="GNURadio-based capture script using airspyhf+")

    parser.add_argument("path", help="The path to the output file relative to the script's location")
    parser.add_argument("--time", help="The capture length in seconds", default=3, type=int)
    parser.add_argument("--samplerate", help="The number of samples to capture per second",
                        default=768000, type=int)
    parser.add_argument("--freq", help="The center frequency (in Hertz)", default=int(13.56e6), type=int)

    args = parser.parse_args()
    seconds = args.time
    sample_rate = args.samplerate
    center_freq = args.freq
    path = os.path.join(os.getcwd(), args.path)

    # Execute flow graph with given parameters
    tb = SimplestCapture(seconds, sample_rate, center_freq, path)

    # TODO Find out why neither the standard "Run to completion" nor the following code work on GR 4.7
    tb.start()
    time.sleep(seconds + 2)
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
