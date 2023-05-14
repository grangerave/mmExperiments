# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 2022
@author: aanferov

Experiment Classes based on mmwave Pulse Experiment

"""
import time, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, trange

from experiments.mmWave.mmExperiments import mmTwoToneExperiment

class TwoToneAmplitude(mmTwoToneExperiment):
    """
    Twotone Amplitude Experiment where amplitude is set by qmultiplier attenuator
    Requires Experimental Config
    expt = dict(
        start: start atten (V), 
        step: stop atten (V),
        expts: number of experiments,
        nPtavg: point averages (fastest),
        nAvgs: points in sweep averages (fast),
        reps: start-finish experiment repeats (very slow)
        )
    """

    def __init__(self, InstrumentDict, path='', prefix='AmplitudeRabiSegment', config_file=None, progress=None,**kwargs):
        super().__init__(InstrumentDict, path=path, prefix=prefix, config_file=config_file, progress=progress,**kwargs)
        self.pulses_plotted=False

    #override
    def acquire(self, progress=False,plot_pulse=False,start_on=False,leave_on=False):
        if 'stop' in self.cfg.expt:
            if 'step' in self.cfg.expt:
                self.cfg.expt["expts"] = 1+int(np.ceil(abs(self.cfg.expt["stop"]-self.cfg.expt["start"])/self.cfg.expt["step"]))
            else:
                self.cfg.expt["step"] = abs(self.cfg.expt["stop"]-self.cfg.expt["start"])/self.cfg.expt["expts"]
        xpts=self.cfg.expt["start"] + self.cfg.expt["step"]*np.arange(self.cfg.expt["expts"])
        #trim xpts
        xpts = [x for x in xpts if x <=0.5]

        if 'reps' not in self.cfg.expt: self.cfg.expt.reps = 1 
        elif self.cfg.expt.reps == 0: self.cfg.expt.reps = 1

        self.prep()

        if 'sigma' not in self.cfg.expt: self.cfg.expt.sigma = self.cfg.device.qubit.pulses.pi_ge.sigma
        #default values
        if 'delay' not in self.cfg.expt: self.cfg.expt.delay = 0.0
        if 'phase' not in self.cfg.expt: self.cfg.expt.phase = 0.0
        if 'sigma_cutoff' not in self.cfg.expt: self.cfg.expt.sigma_cutoff=3

        #figure out first domain
        divN0 = 1
        awg_gain = xpts[0]
        while awg_gain < 0.25:
            divN0 = divN0*2
            awg_gain = xpts[0]*divN0 #awg_gain*2
        print(f'first Gain: {xpts[0]} Amplitude domain: 1/{divN0} AWG amp: {awg_gain} Output: {awg_gain/divN0}')
        #store in config
        self.cfg.expt.divN0 = divN0
        self.cfg.expt.awg_gain = awg_gain

        #turn on and stabilize
        if not start_on:
            self.on(quiet = not progress,tek=True)

        data={"xpts":np.array(xpts), "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for i in tqdm(range(self.cfg.expt["reps"]),disable=not progress):
            divN = self.cfg.expt.divN0
            #load the first pulse
            self.tek.set_amplitude(1,self.cfg.expt.awg_gain)
            self.load_pulse_and_run(type=self.cfg.expt.pulse_type,delay=self.cfg.expt.delay,sigma=self.cfg.expt.sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
                amp=1/divN,phase=self.cfg.expt.phase)

            if plot_pulse and not self.pulses_plotted: 
                self.plot_pulses()
                self.pulses_plotted=True

            data_shot={"avgi":[], "avgq":[], "amps":[], "phases":[]}

            for a in tqdm(xpts, disable=not progress,desc='%d/%d'%(i+1,self.cfg.expt['reps']),leave=False):
                #update amplitude
                awg_gain=np.round(a*divN,3) # min step is 1mV
                if awg_gain>0.5:
                    divN=divN//2
                    awg_gain=np.round(a*divN,3)
                    self.tek.stop()
                    self.tek.set_amplitude(1,awg_gain)
                    self.load_pulse_and_run(type=self.cfg.expt.pulse_type,delay=self.cfg.expt.delay,sigma=self.cfg.expt.sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
                        amp=1/divN,phase=self.cfg.expt.phase,quiet=True)
                else:
                    self.tek.set_amplitude(1,awg_gain)
                #wait for tek amplitude to set
                time.sleep(self.cfg.hardware.awg_update_time)

                self.PNAX.set_sweep_mode('SING')
                #set format = polar?
                response = self.PNAX.read_data()
                avgi = np.mean(response[1])
                avgq = np.mean(response[2])
                amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
                phase = np.angle(avgi+1j*avgq) # Calculating the phase

                #data["xpts"].append(a)
                data_shot["avgi"].append(avgi)
                data_shot["avgq"].append(avgq)
                data_shot["amps"].append(amp)
                data_shot["phases"].append(phase)
            
            for k, a in data_shot.items():
                data[k].append(a)
            
        #final averaging
        for k in ['avgi','avgq','amps','phases']:
            data[k] = np.mean(data[k],axis=0)

        #for k, a in data.items():
        #    data[k]=np.array(a)
        
        self.data=data
        #turn off if finished
        if not leave_on:
            self.off(quiet = not progress)

        return data

    def analyze(self, data=None, fit=False,verbose=True, **kwargs):
        if data is None:
            data=self.data
            
        if fit:
            #TODO implement 
            pass
            
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 
        plt.figure(figsize=(10,8))
        plt.subplot(211, title="Amplitude Rabi", ylabel="I (Receiver B)")
        plt.plot(data["xpts"], data["avgi"],'o-')
        if fit:
            pass
        plt.subplot(212, xlabel="Gain (mVpp)", ylabel="Q (Receiver B)")
        plt.plot(data["xpts"], data["avgq"],'o-')
        if fit:
            pass
        plt.tight_layout()
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)