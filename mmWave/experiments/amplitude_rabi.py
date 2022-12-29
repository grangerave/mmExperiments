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
        reps: start-finish experiment repeats (very slow),
        pulse_type: 'gauss' or 'square',
        sigma: gaussian sigma for pulse length [ns] (default: from pi_ge in config)
        )
    """

    def __init__(self, InstrumentDict, path='', prefix='AmplitudeRabiSegment', config_file=None, progress=None,**kwargs):
        super().__init__(InstrumentDict, path=path, prefix=prefix, config_file=config_file, progress=progress,**kwargs)
        self.pulses_plotted=False

    #override
    def acquire(self, progress=False,plot_pulse=False,start_on=False,leave_on=False):
        if 'stop' in self.cfg.expt:
            self.cfg.expt["expts"] = 1+int(np.ceil(abs(self.cfg.expt["stop"]-self.cfg.expt["start"])/self.cfg.expt["step"]))
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
            awg_gain = xpts[0]*divN0
        print(f'First amplitude domain is 1/{divN0}')

        #turn on and stabilize
        if not start_on:
            self.on(quiet = not progress,tek=True)

        data={"xpts":np.array(xpts), "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for i in tqdm(range(self.cfg.expt["reps"]),disable=not progress):
            divN = divN0
            #load the first pulse
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
                    self.load_pulse_and_run(type='gauss',delay=self.cfg.expt.delay,sigma=self.cfg.expt.sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
                        amp=1/divN,phase=self.cfg.expt.phase,quiet=True)

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


class AmplitudeFreqRabiSegmentExperiment(mmPulseExperiment):
    """
    Amplitude-Frequency Rabi Experiment where amplitude is set by awg when possible
    Requires Experimental Config
    expt = dict(
        start_gain: start gain (mVpp), 
        step_gain: stop gain (mVpp),
        start_f: start freq (GHz),
        step_f: step freq (GHz),
        expts_gain: number of gain points,
        expts_f: frequency points,
        nPtavg: point averages (fastest),
        nAvgs: points in sweep averages (fast),
        pulse_type: 'gauss' or 'square',
        sigma: gaussian sigma for pulse length [ns] (default: from pi_ge in config)
        )
    """

    def __init__(self, InstrumentDict, path='', prefix='FreqAmplitudeRabiSegment', config_file=None, progress=None,**kwargs):
        super().__init__(InstrumentDict, path=path, prefix=prefix, config_file=config_file, progress=progress,**kwargs)
        self.pulses_plotted=False
        self.maxTargetTemp=0.90

    #override
    def acquire(self, progress=False,sub_progress=False,plot_pulse=False,start_on=False,leave_on=False):
        fpts= self.cfg.expt["start_f"]+ self.cfg.expt["step_f"]*np.arange(self.cfg.expt["expts_f"])
        xpts=self.cfg.expt["start_gain"] + self.cfg.expt["step_gain"]*np.arange(self.cfg.expt["expts_gain"])
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

        if not start_on:
            self.on()

        data={"xpts":[],"fpts":[],"avgi":[], "avgq":[], "amps":[], "phases":[]}

        for f in tqdm(fpts,disable=not progress):
            if f>1e9: f=f/1e9   #correct for freq in Hz
            #update freq
            if self.cfg.device.qubit.upper_sideband:
                self.lo_freq_qubit = f - self.cfg.device.qubit.if_freq
            else:   #use lower sideband
                self.lo_freq_qubit = f + self.cfg.device.qubit.if_freq
            self.amcMixer.set_frequency(self.lo_freq_qubit*1e9)

            while True:
                #avoid collecting data if already warm
                self.waitForCycle(**self.fridge_config)
                #collect data
                result = self.acquire_pt(xpts,plot_pulse=plot_pulse,progress=sub_progress)
                if self.waitForCycle(**self.fridge_config):
                    break   #the temperature is good. Proceed
            data["avgi"].append(result["avgi"])
            data["avgq"].append(result["avgq"])
            data["amps"].append(result["amps"])
            data["phases"].append(result["phases"])

        
        data["xpts"] = xpts #1D array
        data["fpts"] = fpts #1D array

        for k, a in data.items():
            data[k]=np.array(a)
        
        self.data=data
        #turn off if finished
        if not leave_on:
            self.off(quiet = not progress)

        return data

    def acquire_pt(self,xpts,plot_pulse=False,progress=True):
        #figure out first domain
        divN0 = 1
        awg_gain = xpts[0]
        while awg_gain < 0.25:
            divN0 = divN0*2
            awg_gain = xpts[0]*divN0
        print(f'First amplitude domain is 1/{divN0}')

        data={"xpts":np.array(xpts), "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for i in tqdm(range(self.cfg.expt["reps"]),disable=not progress):
            divN = divN0
            #load the first pulse
            self.load_pulse_and_run(type=self.cfg.expt.pulse_type,delay=self.cfg.expt.delay,sigma=self.cfg.expt.sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
                amp=1/divN,phase=self.cfg.expt.phase,quiet=True)

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
                    self.load_pulse_and_run(type='gauss',delay=self.cfg.expt.delay,sigma=self.cfg.expt.sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
                        amp=1/divN,phase=self.cfg.expt.phase,quiet=True)

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
        x_sweep = data['xpts']
        y_sweep = data['fpts']
        avgi = data['avgi']
        avgq = data['avgq']

        plt.figure(figsize=(10,8))
        plt.subplot(211, title="Amplitude Rabi", ylabel="Frequency [GHz]")
        plt.imshow(
            np.flip(avgi, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.colorbar(label='I (Receiver B)')
        plt.clim(vmin=None, vmax=None)
        # plt.axvline(1684.92, color='k')
        # plt.axvline(1684.85, color='r')

        plt.subplot(212, xlabel="Gain (mVpp)", ylabel="Frequency [GHz]")
        plt.imshow(
            np.flip(avgq, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.colorbar(label='Q (Receiver B)')
        plt.clim(vmin=None, vmax=None)
        
        if fit: pass

        plt.tight_layout()
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)