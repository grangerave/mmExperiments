# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 09:02:48 2021
@author: aanferov

A library for shared code used in millimeter wave experiments

"""
import time, os
import numpy as np
import matplotlib.pyplot as plt
from instruments.PNAX import N5242A
from experiments.fridgeExperiment import FridgeExperiment, waitForCycle

#for tek70001a
from experiments.PulseExperiments.sequencer import Sequencer
from experiments.PulseExperiments.pulse_classes import Gauss, Square, Idle
from instruments.awg.Tek70001 import write_Tek70001_sequence
from instruments.awg.Tek70001 import Tek70001

# --------------- Helper functions ---------------

#generic save function
def saveCSV(data,filename,dataFolder,dataSubFolder=None,dataSubSubFolder=None,dataPath='S:\\_Data\\'):
    path=dataPath+'\\'+dataFolder+'\\'+(dataSubFolder is not None and (str(dataSubFolder) + '\\') or '')+(dataSubSubFolder is not None and (str(dataSubSubFolder) + '\\s') or '')+filename
    print('Saving as: '+path)
    np.savetxt(path,data,delimiter=',')

#formatted date
def dateStr():return time.strftime('%y%m%d')

#function to setup VNA for pulse
def setVNApulsed(VNA):
    pass

#function to unset pulses on VNA and re-enable CW
def setVNAcw(VNA):
    pass

def floor10(x):
    #return an int rounded down to the nearest 10. needs x>0
    return int(round(x-5,-1))

# --------------- Parent Experiment Class ---------------
# support for turning instruments on/off when waiting for cycle
# wrapper for experiment class
# call acquireInCycle() instead of acquire()

class mmPulseExperiment(FridgeExperiment):
    def __init__(self,InstrumentDict, path='', prefix='Test', dataFolder='data',seqFolder='sequences',config_file=None, progress=None,debug=False,**kwargs):
        super().__init__(path=path,dataFolder=dataFolder,prefix=prefix,config_file=config_file,progress=progress,**kwargs)
        self.seqFolder=seqFolder
        #cache reference to instruments
        self.PNAX = InstrumentDict['VNA']
        self.amcMixer = InstrumentDict['amcMixer']
        self.amcMixer.bk.Remote()
        self.tek = InstrumentDict['tek']

        if debug:
            print(self.PNAX.get_id())

    def on(self,quiet=False,stabilize_time=None,tek=False):
        if stabilize_time is None: stabilize_time=self.cfg.hardware.stabilize_time
        if not quiet: print('Turning Instruments ON')
        self.PNAX.set_output(True)
        self.PNAX.set_sweep_mode('CONT')
        self.amcMixer.on(stabilize=stabilize_time)
        if tek:
            self.tek.run()

    def off(self,quiet=False):
        if not quiet: print('Turning Instruments OFF')
        self.tek.stop()
        self.PNAX.set_sweep_mode('HOLD')
        self.PNAX.set_output(False)
        self.amcMixer.off()

    def prep(self,setVNApulse=True,setFreqs=True):
        #update config values with device index
        try:
            device_n = self.cfg.expt.device_n
        except:
            print('device_n not defined! Assuming 0')
            device_n=0
        for key, value in self.cfg.device.readout.items():  #support 1 level of dict in readout
            if isinstance(value, list):
                self.cfg.device.readout.update({key: value[device_n]})
        for key, value in self.cfg.device.qubit.items():    #support 3 levels of dict in qubit (excess)
            if isinstance(value, list):
                self.cfg.device.qubit.update({key: value[device_n]})
            elif isinstance(value, dict):
                for key2, value2 in value.items():
                    for key3, value3 in value2.items():
                        if isinstance(value3, list):
                            value2.update({key3: value3[device_n]})
        
        #setup the pulses
        if setVNApulse: self.setVNApulsed()
        if setFreqs: self.setup_freq_and_power()

    def setVNApulsed(self):
        self.PNAX.set_timeout(120)
        
        #enable subpoint averaging
        self.PNAX.write('SENS:PULSE0:SUBP 1')

        #disable leveling for source 1 (not technically needed since source 1 is levelled before switch)
        self.PNAX.write("source1:power1:alc:mode openloop")

        #trigger on point
        self.PNAX.write("SENS:SWE:TRIG:MODE POINT")
        #Turning off the "Reduce IF BW at Low Frequencies" because the PNA-X adjusts the BW automatically to correct for roll-off at low frequencies
        self.PNAX.write("SENS:BWID:TRAC OFF")

        #turn on the pulses
        for n in range(5):
            self.PNAX.write("SENS:PULS%d 1"%n)
            #turning off the inverting
            if n>0: self.PNAX.write("SENS:PULS%d:INV 0"%n)
        
        #Pulse timing
        #awg delay in ns
        #self.awg_trig_delay=(self.cfg['hardware']['period']-floor10(self.cfg['hardware']['awg_trigger_time'])%self.cfg['hardware']['period']-floor10(self.cfg['hardware']['awg_offset']))

        self.PNAX.write("SENS:PULS:PERiod %.12f" % (self.cfg['hardware']['period']*1e-9))
        self.PNAX.write("SENS:PULS0:DEL %.12f" % (self.cfg['hardware']['ADC_delay']*1e-9) )
        #(1) readout pulse
        self.PNAX.write("SENS:PULS1:WIDT %.12f" % (self.cfg['device']['readout']['width']*1e-9))
        self.PNAX.write("SENS:PULS1:DEL %.12f" % (self.cfg['device']['readout']['delay']*1e-9))
        #(2) unused- copy pulse1 for debug
        self.PNAX.write("SENS:PULS2:WIDT %.12f" % (self.cfg['device']['readout']['width']*1e-9))
        self.PNAX.write("SENS:PULS2:DEL %.12f" % (self.cfg['device']['readout']['delay']*1e-9))
        #(3) tek trigger
        self.PNAX.write("SENS:PULS3:WIDT %.12f" % (self.cfg['hardware']['awg_trigger']['width']*1e-9))
        #self.PNAX.write("SENS:PULS3:DEL %.12f" % (self.awg_trig_delay*1e-9))
        self.update_awg_offset(self.cfg['hardware']['awg_offset'])
        #(4) unused
        self.PNAX.write("SENS:PULS4:WIDT %.12f" % (self.cfg['hardware']['pulse4']['width']*1e-9))
        self.PNAX.write("SENS:PULS4:DEL %.12f" % (self.cfg['hardware']['pulse4']['delay']*1e-9))

    def update_awg_offset(self,awg_offset):
        #awg_offset in ns
        self.cfg.hardware.awg_offset = awg_offset
        self.awg_trig_delay=(self.cfg['hardware']['period']-floor10(self.cfg['hardware']['awg_trigger_time'])%self.cfg['hardware']['period']-floor10(self.cfg['hardware']['awg_offset']))
        self.PNAX.write("SENS:PULS3:DEL %.12f" % (self.awg_trig_delay*1e-9))

    def updateGain(self,gain,quiet=False):
        divN = 1
        awg_gain = gain
        while awg_gain < 0.25:
            divN = divN*2
            awg_gain = awg_gain*2
        print(f'Gain: {gain} Amplitude domain: 1/{divN} AWG amp: {awg_gain} Output: {awg_gain/divN}')
        #store in expt
        self.cfg.expt.divN = divN
        self.cfg.expt.awg_gain = awg_gain

    def setup_freq_and_power(self):
        """
        #assume initial part of prep() was already run
        #needs to have expt dictionary set in cfg with the following:
        expt = dict(
            device_n: device index
            nPtavg: point averages (fastest)
            nAvgs: points in sweep averages (fast)
            #optional:
                ifbw: VNA ifbw
                read_power: VNA readout power
                probe_LO_freq: amc mixer LO frequency
        )
        """
        self.PNAX.set_span(0)
        if 'ifbw' in self.cfg.expt.keys():
            self.PNAX.set_ifbw(self.cfg.expt.ifbw)
        else:
            self.PNAX.set_ifbw(self.cfg.device.readout.ifbw)
        #TODO configure VNA if path
        if 'read_power' in self.cfg.expt.keys():
            self.PNAX.set_power(self.cfg.expt.read_power)
        else:
            self.PNAX.set_power(self.cfg.device.readout.power)
        self.PNAX.set_center_frequency(self.cfg.device.readout.freq*1e9)
        #TODO need to handle readout frequency sideband
        self.PNAX.set_average_state(True)
        self.PNAX.set_averages_and_group_count(self.cfg.expt.nPtavg)
        self.PNAX.set_sweep_points(self.cfg.expt.nAvgs)
        
        #amcMixer freq (in GHz)
        if 'probe_lo_freq' in self.cfg.expt.keys():
            self.lo_freq_qubit = self.cfg.expt.probe_lo_freq
        else:
            if self.cfg.device.qubit.upper_sideband:
                self.lo_freq_qubit = self.cfg.device.qubit.f_ge - self.cfg.device.qubit.if_freq
            else:   #use lower sideband
                self.lo_freq_qubit = self.cfg.device.qubit.f_ge + self.cfg.device.qubit.if_freq
        self.amcMixer.set_frequency(self.lo_freq_qubit*1e9)


    def load_pulse(self,pulses=None,delay=0.0,type=None,sigma=None,sigma_cutoff=3,amp=1.0,ramp=0.1,phase=0,pulse_name=None,quiet=False):
        #override for more complicated pulses
        self.awg_dt = self.cfg.hardware.awg_info.tek70001a.dt
        self.sequencer = Sequencer(list(self.cfg.hardware.awg_channels.keys()),self.cfg.hardware.awg_channels,self.cfg.hardware.awg_info,{})
        
        if pulses is not None:
            #feed the pulses into the sequencer
            for pulse in pulses:
                self.sequencer.append('Ch1',pulse)
        else:
            pulse = None
            #define the pulses
            if type == 'square':
                #Square(max_amp, flat_len, ramp_sigma_len, cutoff_sigma, freq, phase, phase_t0 = 0, dt=None)
                pulse = Square(amp,sigma,ramp,1,self.cfg.device.qubit.if_freq,phase,dt=self.awg_dt)
            elif type == 'gauss':
                pulse = Gauss(amp,sigma,sigma_cutoff,self.cfg.device.qubit.if_freq,phase)
            else:
                print('Error: could not interpret pulse type!')
            if pulse is not None: self.pulse_length = pulse.get_length() # in ns?

            #pulse is too long! may need to auto-adjust the trig delay (needs to be a multiple of 10)
            if floor10(self.cfg.hardware.awg_offset) - (self.cfg.hardware.awg_trigger_time % 10) < self.pulse_length+delay:
                print('Warning: pulse is longer than awg_offset!')

            self.sequencer.new_sequence(floor10(self.cfg.hardware.awg_offset) - (self.cfg.hardware.awg_trigger_time % 10) -pulse.get_length()-delay)
            self.sequencer.append('Ch1', pulse)

        self.sequencer.end_sequence(0.1)#pads pulse with 0.1ns of 0s
        self.multiple_sequences=self.sequencer.complete()

        if pulse_name is None:
            pulse_name='Pulse_%s_s%1.1fns_%0.2fx_d%dns_%.2fGHz'%(type,sigma,amp,delay,self.cfg.device.qubit.if_freq)
        write_Tek70001_sequence([self.multiple_sequences[0]['Ch1']],os.path.join(self.path, self.seqFolder), pulse_name,awg=self.tek,quiet=quiet)
        self.tek.prep_experiment()
        #note need to do tek.run after this

    def load_pulse_and_run(self,**kwargs):
        self.load_pulse(**kwargs)
        self.tek.set_enabled(1,'on')
        self.tek.run()
        time.sleep(self.cfg.hardware.awg_load_time)

    def plot_pulses(self,stagger=0.25):
        plt.figure(figsize=(18,4))
        plt.subplot(111, title=f"Pulse Timing", xlabel="t (ns)")
        plt.plot(np.arange(0,len(self.multiple_sequences[0]['Ch1']))*self.awg_dt,self.multiple_sequences[0]['Ch1'])
        readout_ptx=[0.,self.cfg.hardware.awg_offset+self.cfg.device.readout.delay,
            self.cfg.hardware.awg_offset+self.cfg.device.readout.delay,
            self.cfg.hardware.awg_offset+self.cfg.device.readout.delay+self.cfg.device.readout.width,
            self.cfg.hardware.awg_offset+self.cfg.device.readout.delay+self.cfg.device.readout.width,
            self.cfg.hardware.awg_info.tek70001a.dt * self.cfg.hardware.awg_info.tek70001a.min_samples]
        readout_pty=[x + stagger for x in [0,0,.5,.5,0,0]]
        plt.plot(readout_ptx,readout_pty)
        plt.xlabel('t (ns)')    
        plt.show()