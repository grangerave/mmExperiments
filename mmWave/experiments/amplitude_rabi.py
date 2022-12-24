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

from experiments.mmWave.mmExperiments import mmPulseExperiment

class AmplitudeRabiSegmentExperiment(mmPulseExperiment):
    """
    Amplitude Rabi Experiment where amplitude is set by awg when possible
    Requires Experimental Config
    expt = dict(
        start: start gain (mVpp), 
        step: stop gain (mVpp),
        expts: number of experiments,
        nPtavg: point averages (fastest),
        nAvgs: points in sweep averages (fast),
        pulse_type: 'gauss' or 'square',
        sigma: gaussian sigma for pulse length [ns] (default: from pi_ge in config)
        )
    """

    def __init__(self, InstrumentDict, path='', prefix='AmplitudeRabiSegment', config_file=None, progress=None,**kwargs):
        super().__init__(InstrumentDict, path=path, prefix=prefix, config_file=config_file, progress=progress,**kwargs)

    #override
    def acquire(self, progress=False,plot_pulse=False,start_on=False,leave_on=False):
        xpts=self.cfg.expt["start"] + self.cfg.expt["step"]*np.arange(self.cfg.expt["expts"])
        #trim xpts
        xpts = [x for x in xpts if x <=0.5]

        if 'sigma' not in self.cfg.expt: self.cfg.expt.sigma = self.cfg.device.qubit.pulses.pi_ge.sigma
        #default values
        if 'delay' not in self.cfg.expt: self.cfg.expt.delay = 0.0
        if 'phase' not in self.cfg.expt: self.cfg.expt.phase = 0.0
        if 'sigma_cutoff' not in self.cfg.expt: self.cfg.expt.sigma_cutoff=3

        self.prep()

        #figure out first domain
        divN = 1
        awg_gain = xpts[0]
        while awg_gain < 0.25:
            divN = divN*2
            awg_gain = xpts[0]*divN
        print(f'First amplitude domain is 1/{divN}')

        #load the first pulse
        self.load_pulse(type=self.cfg.expt.pulse_type,delay=self.cfg.expt.delay,sigma=self.cfg.expt.sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
            amp=1/divN,phase=self.cfg.expt.phase)

        if plot_pulse: self.plot_pulses()

        if not start_on:
            self.on(quiet = not progress)
        else:
            self.tek.run()
        time.sleep(self.cfg.hardware.awg_load_time)

        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}

        for a in tqdm(xpts, disable=not progress):
            #update amplitude
            awg_gain=np.round(a*divN,3) # min step is 1mV
            if awg_gain>0.5:
                divN=divN//2
                awg_gain=np.round(a*divN,3)
                self.load_pulse_and_run(type='gauss',delay=self.cfg.expt.delay,sigma=self.cfg.expt.sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
                    amp=1/divN,phase=self.cfg.expt.phase)

            self.PNAX.set_sweep_mode('SING')
            response = self.PNAX.read_data()
            avgi = np.mean(response[1])
            avgq = np.mean(response[2])
            amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phase = np.angle(avgi+1j*avgq) # Calculating the phase

            data["xpts"].append(a)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)
        
        for k, a in data.items():
            data[k]=np.array(a)
        
        self.data=data
        #turn off if finished
        if not leave_on:
            self.off(quiet = not progress)

        return data

    def plot_pulses(self):
        plt.figure(figsize=(18,4))
        plt.subplot(111, title=f"Pulse Timing", xlabel="t (ns)")
        plt.plot(np.arange(0,len(self.multiple_sequences[0]['Ch1']))*self.awg_dt,self.multiple_sequences[0]['Ch1'])
        readout_ptx=[0.,self.cfg.hardware.awg_offset+self.cfg.device.readout.delay,
            self.cfg.hardware.awg_offset+self.cfg.device.readout.delay,
            self.cfg.hardware.awg_offset+self.cfg.device.readout.delay+self.cfg.device.readout.width,
            self.cfg.hardware.awg_offset+self.cfg.device.readout.delay+self.cfg.device.readout.width,
            self.cfg.hardware.awg_offset+self.cfg.device.readout.delay+2*self.cfg.device.readout.width]
        readout_pty=[0,0,.5,.5,0,0]
        plt.plot(readout_ptx,readout_pty)
        plt.xlabel('t (ns)')    
        plt.show()

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
        plt.figure(figsize=(18,6))
        plt.subplot(111, title=f"Amplitude Rabi", xlabel="AWG amplitude (mVpp)", ylabel="Amps (Reciever B)")
        
        plt.plot(data["xpts"][1:-1], data["amps"][1:-1],'o-')
        
        if fit:
            pass
            
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)