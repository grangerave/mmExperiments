# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 2022
@author: aanferov

Experiment Classes based on mmwave Pulse Experiment

"""
import time, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, trange

from experiments.mmWave.mmExperiments import mmPulseExperiment

class PulseProbeExperiment(mmPulseExperiment):
    """
    Pulse Probe Experiment where a probe tone is swept while monitoring readout
    Requires Experimental Config
    expt = dict(
        start: frequency start (GHz), 
        step: frequency step (GHz),
        expts: number of experiments,
        nPtavg: point averages (fastest),
        nAvgs: points in sweep averages (fast),
        reps: start-finish experiment repeats (very slow),
        pulse_type: 'gauss' or 'square',
        sigma: gaussian sigma / square pulse length [ns] (default: from pi_ge in config)
        (optional)
            gain: pulse gain. defaults to device.qubit.pulses.pi_ge.gain
            ramp: sigma for ends of square pulse [ns]
        )
    """

    def __init__(self, InstrumentDict, path='', prefix='PulseProbe', config_file=None, progress=None,**kwargs):
        super().__init__(InstrumentDict, path=path, prefix=prefix, config_file=config_file, progress=progress,**kwargs)
        self.pulses_plotted=False

    #override
    def acquire(self, progress=False,plot_pulse=False,start_on=False,leave_on=False):
        xpts=self.cfg.expt["start"] + self.cfg.expt["step"]*np.arange(self.cfg.expt["expts"])

        if 'reps' not in self.cfg.expt: self.cfg.expt.reps = 1 
        elif self.cfg.expt.reps == 0: self.cfg.expt.reps = 1

        self.prep()

        if 'sigma' not in self.cfg.expt: self.cfg.expt.sigma = self.cfg.device.qubit.pulses.pi_ge.sigma
        if 'gain' not in self.cfg.expt: self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_ge.gain
        #default values
        if 'delay' not in self.cfg.expt: self.cfg.expt.delay = 0.0
        if 'phase' not in self.cfg.expt: self.cfg.expt.phase = 0.0
        if 'sigma_cutoff' not in self.cfg.expt: self.cfg.expt.sigma_cutoff=3

        

        #turn on and stabilize
        if not start_on:
            self.on(quiet = not progress,tek=False)

        #figure out pulse domain
        divN = 1
        awg_gain = self.cfg.expt.gain
        while awg_gain < 0.25:
            divN = divN*2
            awg_gain = xpts[0]*divN
        print(f'Amplitude domain is 1/{divN}')
        
        self.tek.set_amplitude(1,awg_gain)
        #load the first pulse
        self.load_pulse_and_run(type=self.cfg.expt.pulse_type,delay=self.cfg.expt.delay,sigma=self.cfg.expt.sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
            amp=1/divN,phase=self.cfg.expt.phase)

        if plot_pulse and not self.pulses_plotted: 
            self.plot_pulses()
            self.pulses_plotted=True

        data={"xpts":np.array(xpts), "avgi":[], "avgq":[], "amps":[], "phases":[],"lo_qubit":[]}
        if self.cfg.device.qubit.upper_sideband:
            #already numpy array
            data["lo_qubit"]=data["xpts"] - self.cfg.device.qubit.if_freq
        else:
            data["lo_qubit"]=data["xpts"] + self.cfg.device.qubit.if_freq

        for i in tqdm(range(self.cfg.expt["reps"]),disable=not progress):
            data_shot={"avgi":[], "avgq":[], "amps":[], "phases":[]}

            for f in tqdm(xpts, disable=not progress,desc='%d/%d'%(i+1,self.cfg.expt['reps']),leave=False):
                if f>1e9: f=f/1e9   #correct for freq in Hz
                #update freq
                if self.cfg.device.qubit.upper_sideband:
                    self.lo_freq_qubit = f - self.cfg.device.qubit.if_freq
                else:   #use lower sideband
                    self.lo_freq_qubit = f + self.cfg.device.qubit.if_freq
                self.amcMixer.set_frequency(self.lo_freq_qubit*1e9,acknowledge=False)

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

    def analyze(self, data=None, fit=False, findpeaks=False,verbose=True, **kwargs):
        if data is None:
            data=self.data
            
        if fit:
            #TODO implement 
            pass

        if findpeaks:
            maxpeak=data["xpts"][np.argmax(data["amps"])]
            minpeak=data["xpts"][np.argmin(data["amps"])]
            data['maxpeaks'] = [maxpeak]
            data['minpeaks'] = [minpeak]
            
        return data

    def display(self, data=None, fit=False, findpeaks=True,**kwargs):
        if data is None:
            data=self.data 
        plt.figure(figsize=(16,10))
        plt.subplot(211, title="Pulse Probe", ylabel="I (Receiver B)")
        plt.plot(data["xpts"], data["avgi"],'o-')
        if findpeaks:
            for peak in data['maxpeaks']:
                plt.axvline(peak, linestyle='--', color='0.2')
                print(f'Max Peak [GHz]: {peak}')
        plt.subplot(212, xlabel="Pulse Frequency (GHz)", ylabel="Q (Receiver B)")
        plt.plot(data["xpts"], data["avgq"],'o-')
        if findpeaks:
            for peak in data['maxpeaks']:
                plt.axvline(peak, linestyle='--', color='0.2')
                #print(f'Max Peak [GHz]: {peak}')
        plt.tight_layout()
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)