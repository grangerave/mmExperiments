# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 2023
@author: aanferov

Experiment Classes based on mmwave Pulse Experiment

"""
import time, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, trange

from experiments.mmWave.mmExperiments import mmPulseExperiment

class T1Experiment(mmPulseExperiment):
    """
    T1 Experiment where a new sequence is written for every point
    Requires Experimental Config
    expt = dict(
        start: start time [ns] (can be negative), 
        step: step time [ns],
        expts: number of experiments,
        nPtavg: point averages (fastest),
        nAvgs: points in sweep averages (fast),
        reps: start-finish experiment repeats (extremely slow),
        pulse_type: 'gauss' or 'square',
        sigma: pulse length [ns] (default: from pi_ge in config)
        gain: pulse amplitude [mVpp]
        )
    """

    def __init__(self, InstrumentDict, path='', prefix='T1', config_file=None, progress=None,**kwargs):
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

        if 'reps' not in self.cfg.expt: self.cfg.expt.reps = 1 
        elif self.cfg.expt.reps == 0: self.cfg.expt.reps = 1

        self.prep()

        if 'sigma' not in self.cfg.expt: self.cfg.expt.sigma = self.cfg.device.qubit.pulses.pi_ge.sigma
        if 'gain' not in self.cfg.expt: self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_ge.gain
        #default values
        if 'phase' not in self.cfg.expt: self.cfg.expt.phase = 0.0
        if 'sigma_cutoff' not in self.cfg.expt: self.cfg.expt.sigma_cutoff=3
        if 'ramp' not in self.cfg.expt: self.cfg.expt.ramp = 0.1

        #figure out pulse domain
        divN = 1
        awg_gain = self.cfg.expt.gain
        while awg_gain < 0.25:
            divN = divN*2
            awg_gain = awg_gain*2
        print(f'Gain: {self.cfg.expt.gain} Amplitude domain: 1/{divN} AWG amp: {awg_gain} Output: {awg_gain/divN}')
        #store in config
        self.cfg.expt.divN = divN
        self.cfg.expt.awg_gain = awg_gain

        #turn on and stabilize
        if not start_on:
            self.on(quiet = not progress,stabilize_time=2,tek=False)

        #load waveforms
        self.tek.pre_load()
        for exp_n,delay in tqdm(enumerate(xpts),disable=not progress,desc='Loading Waveforms'):
            self.write_pulse_batch(exp_n,type=self.cfg.expt.pulse_type,delay=delay,sigma=self.cfg.expt.sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
                    amp=1/divN,ramp=self.cfg.expt.ramp,phase=self.cfg.expt.phase,quiet=True)
            
        data={"xpts":np.array(xpts), "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for i in tqdm(range(self.cfg.expt["reps"]),disable=not progress):

            data_shot={"avgi":[], "avgq":[], "amps":[], "phases":[]}
            for exp_n,delay in tqdm(enumerate(xpts), disable=not progress,desc='%d/%d'%(i+1,self.cfg.expt['reps']),leave=False):
                #update delay
                #self.tek.stop()
                self.tek.set_amplitude(1,awg_gain)
                #self.load_pulse_and_run(type=self.cfg.expt.pulse_type,delay=delay,sigma=self.cfg.expt.sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
                #    amp=1/divN,phase=self.cfg.expt.phase,quiet=True)
                self.load_experiment_and_run(exp_n)
                
                #plot pulses if first time
                if plot_pulse and not self.pulses_plotted: 
                    self.plot_pulses()
                    self.pulses_plotted=True

                self.PNAX.set_sweep_mode('SING')
                #set format = polar?
                response = self.read_data_fast()#self.PNAX.read_data()
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

    def display(self, data=None, fit=True, ampPhase=True,**kwargs):
        if data is None:
            data=self.data 
        plt.figure(figsize=(10,8))
        
        if ampPhase:
            plt.subplot(211, title="T1", ylabel="Amp (Receiver B)")
            plt.plot(data["xpts"],data["amps"],'o-')
        else:
            plt.subplot(211, title="T1", ylabel="I (Receiver B)")
            plt.plot(data["xpts"], data["avgi"],'o-')
        if fit:
            pass
        
        if ampPhase:
            plt.subplot(212, xlabel="Time (ns)", ylabel="Phase (Receiver B)")
            plt.plot(data["xpts"],data["phases"],'o-')
        else:
            plt.subplot(212, xlabel="Time (ns)", ylabel="Q (Receiver B)")
            plt.plot(data["xpts"], data["avgq"],'o-')
        if fit:
            pass
        plt.tight_layout()
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)


class T1AmplitudeExperiment(mmPulseExperiment):
    """
    T1-Amplitude Experiment where amplitude is set by awg when possible
    Requires Experimental Config
    expt = dict(
        start_time: start time [ns] (can be negative), 
        step_time: stop time [ns],
        start_gain: start gain [mVpp],
        step_gain: step gain [mVpp],
        expts_time: number of time points,
        expts_gain: gain points,
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
        if 'stop_time' in self.cfg.expt:
            if 'step_time' in self.cfg.expt:
                self.cfg.expt["expts_time"] = 1+int(np.ceil(abs(self.cfg.expt["stop_time"]-self.cfg.expt["start_time"])/self.cfg.expt["step_time"]))
            else:
                self.cfg.expt["step_time"] = abs(self.cfg.expt["stop_time"]-self.cfg.expt["start_time"])/self.cfg.expt["expts_time"]
        xpts=self.cfg.expt["start_time"] + self.cfg.expt["step_time"]*np.arange(self.cfg.expt["expts_time"])
        if 'stop_gain' in self.cfg.expt:
            if 'step_gain' in self.cfg.expt:
                self.cfg.expt["expts_gain"] = 1+int(np.ceil(abs(self.cfg.expt["stop_gain"]-self.cfg.expt["start_gain"])/self.cfg.expt["step_gain"]))
            else:
                self.cfg.expt["step_gain"] = abs(self.cfg.expt["stop_gain"]-self.cfg.expt["start_gain"])/self.cfg.expt["expts_gain"]
        gpts=self.cfg.expt["start_gain"] + self.cfg.expt["step_gain"]*np.arange(self.cfg.expt["expts_gain"])
        #trim gain pts
        gpts = [x for x in gpts if x <=0.5]

        if 'reps' not in self.cfg.expt: self.cfg.expt.reps = 1
        elif self.cfg.expt.reps == 0: self.cfg.expt.reps = 1
        
        self.prep()

        if 'sigma' not in self.cfg.expt: self.cfg.expt.sigma = self.cfg.device.qubit.pulses.pi_ge.sigma
        #default values
        if 'phase' not in self.cfg.expt: self.cfg.expt.phase = 0.0
        if 'sigma_cutoff' not in self.cfg.expt: self.cfg.expt.sigma_cutoff=3

        #figure out first domain
        divN = 1
        awg_gain = gpts[0]
        while awg_gain < 0.25:
            divN = divN*2
            awg_gain = gpts[0]*divN #awg_gain*2
        print(f'first Gain: {gpts[0]} Amplitude domain: 1/{divN} AWG amp: {awg_gain} Output: {awg_gain/divN}')
        #store in config
        self.cfg.expt.divN = divN
        self.cfg.expt.awg_gain = awg_gain

        #turn on and stabilize
        if not start_on:
            self.on(quiet = not progress,tek=True)

        #load the first pulse
        self.tek.set_amplitude(1,self.cfg.expt.awg_gain)
        self.load_pulse_and_run(type=self.cfg.expt.pulse_type,delay=xpts[0],sigma=self.cfg.expt.sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
            amp=1/divN,phase=self.cfg.expt.phase)

        if plot_pulse and not self.pulses_plotted: 
            self.plot_pulses()
            self.pulses_plotted=True

        data={"xpts":[],"gpts":[],"avgi":[], "avgq":[], "amps":[], "phases":[]}

        for a in tqdm(gpts, disable=not progress):
            #update amplitude in cfg
            self.cfg.expt.awg_gain=np.round(a*self.cfg.expt.divN,3) # min step is 1mV
            if self.cfg.expt.awg_gain>0.5:
                self.cfg.expt.divN=self.cfg.expt.divN//2
                self.cfg.expt.awg_gain=np.round(a*self.cfg.expt.divN,3)
                self.tek.stop()#prevent amplitude from increasing
                self.tek.set_amplitude(1,self.cfg.expt.awg_gain)
            else:
                self.tek.set_amplitude(1,self.cfg.expt.awg_gain)

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
        data["gpts"] = gpts #1D array

        for k, a in data.items():
            data[k]=np.array(a)
        
        self.data=data
        #turn off if finished
        if not leave_on:
            self.off(quiet = not progress)

        return data

    def acquire_pt(self,xpts,plot_pulse=False,progress=True):
        data={"xpts":np.array(xpts), "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for i in tqdm(range(self.cfg.expt["reps"]),disable=not progress):
            data_shot={"avgi":[], "avgq":[], "amps":[], "phases":[]}
            for delay in tqdm(xpts, disable=not progress,desc='%d/%d'%(i+1,self.cfg.expt['reps']),leave=False):
                #update delay
                # amplitude is handled in parent loop
                self.load_pulse_and_run(type=self.cfg.expt.pulse_type,delay=delay,sigma=self.cfg.expt.sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
                    amp=1/self.cfg.expt.divN,phase=self.cfg.expt.phase,quiet=True)
                #wait for tek amplitude to set
                time.sleep(self.cfg.hardware.awg_update_time)
                
                #plot pulses if first time
                if plot_pulse and not self.pulses_plotted: 
                    self.plot_pulses()
                    self.pulses_plotted=True

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

    def analyze(self, data=None, fit=False,verbose=True, **kwargs):
        if data is None:
            data=self.data
            
        if fit:
            #TODO implement 
            pass
            
        return data

    def display(self, data=None, fit=True, ampPhase = False, **kwargs):
        if data is None:
            data=self.data 
        x_sweep = data['xpts']
        y_sweep = data['gpts']
        if ampPhase:
            avgi = data['amps']
            avgq = data['phases']
        else:
            avgi = data['avgi']
            avgq = data['avgq']

        plt.figure(figsize=(10,8))
        plt.subplot(211, title="Amplitude Rabi", ylabel="Pulse Amp [mVpp]")
        plt.imshow(
            np.flip(avgi, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto',
            interpolation='none')
        if ampPhase:
            plt.colorbar(label='Amplitude (Receiver B)')
        else:
            plt.colorbar(label='I (Receiver B)')
        plt.clim(vmin=None, vmax=None)
        # plt.axvline(1684.92, color='k')
        # plt.axvline(1684.85, color='r')

        plt.subplot(212, xlabel="time (ns)", ylabel="Pulse Amp [mVpp]")
        plt.imshow(
            np.flip(avgq, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto',
            interpolation='none')
        if ampPhase:
            plt.colorbar(label='Phase (Receiver B)')
        else:
            plt.colorbar(label='Q (Receiver B)')
        plt.clim(vmin=None, vmax=None)
        
        if fit: pass

        plt.tight_layout()
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)