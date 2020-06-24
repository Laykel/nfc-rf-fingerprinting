#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: NFC Simplest Capture GRC3.7
# Author: Luc Wachter
# GNU Radio version: 3.7.13.5
##################################################

from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import osmosdr
import time


class nfc_simplest_capture_37(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "NFC Simplest Capture GRC3.7")

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 768e3
        self.frequency = frequency = 13.56e6

        ##################################################
        # Blocks
        ##################################################
        self.osmosdr_source_0 = osmosdr.source( args="numchan=" + str(1) + " " + 'airspyhf=0' )
        self.osmosdr_source_0.set_sample_rate(samp_rate)
        self.osmosdr_source_0.set_center_freq(frequency, 0)
        self.osmosdr_source_0.set_freq_corr(0, 0)
        self.osmosdr_source_0.set_dc_offset_mode(0, 0)
        self.osmosdr_source_0.set_iq_balance_mode(0, 0)
        self.osmosdr_source_0.set_gain_mode(False, 0)
        self.osmosdr_source_0.set_gain(20, 0)
        self.osmosdr_source_0.set_if_gain(20, 0)
        self.osmosdr_source_0.set_bb_gain(20, 0)
        self.osmosdr_source_0.set_antenna('', 0)
        self.osmosdr_source_0.set_bandwidth(0, 0)

        self.blocks_file_sink_0_0 = blocks.file_sink(gr.sizeof_gr_complex*1, 'C:\\Users\\Luc\\Desktop\\dev\\TB\\TestSignals\\768k-tests\\tag1-Nexus6p.nfc', False)
        self.blocks_file_sink_0_0.set_unbuffered(False)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.osmosdr_source_0, 0), (self.blocks_file_sink_0_0, 0))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.osmosdr_source_0.set_sample_rate(self.samp_rate)

    def get_frequency(self):
        return self.frequency

    def set_frequency(self, frequency):
        self.frequency = frequency
        self.osmosdr_source_0.set_center_freq(self.frequency, 0)


def main(top_block_cls=nfc_simplest_capture_37, options=None):

    tb = top_block_cls()
    tb.start()
    try:
        raw_input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
