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

class LengthAmplitudeRabiSegmentExperiment(mmPulseExperiment):
    """
    Length-Amplitude Rabi Experiment where amplitude is set by awg when possible
    Requires Experimental Config
    expt = dict(
        start_gain: start gain (mVpp), 
        step_gain: stop gain (mVpp),
        start_sigma: start sigma [ns], 
        step_sigma: stop sigma [ns],
        expts_gain: number of gain points,
        expts_sigma: number of sigma points,
        nPtavg: point averages (fastest),
        nAvgs: points in sweep averages (fast),
        pulse_type: 'gauss' or 'square'
        )
    """

    def __init__(self, InstrumentDict, path='', prefix='LengthAmplitudeRabiSegment', config_file=None, progress=None,**kwargs):
        super().__init__(InstrumentDict, path=path, prefix=prefix, config_file=config_file, progress=progress,**kwargs)
        self.pulses_plotted=False
        self.maxTargetTemp=0.90

    #override
    def acquire(self, progress=False,sub_progress=False,plot_pulse=False,start_on=False,leave_on=False):
        if 'stop_sigma' in self.cfg.expt:
            if 'step_gain' in self.cfg.expt:
                self.cfg.expt["expts_sigma"] = 1+int(np.ceil(abs(self.cfg.expt["stop_sigma"]-self.cfg.expt["start_sigma"])/self.cfg.expt["step_sigma"]))
            else:
                self.cfg.expt["step_sigma"] = abs(self.cfg.expt["stop_sigma"]-self.cfg.expt["start_sigma"])/self.cfg.expt["expts_sigma"]
        xpts=self.cfg.expt["start_sigma"] + self.cfg.expt["step_sigma"]*np.arange(self.cfg.expt["expts_sigma"])
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

        #default values
        if 'delay' not in self.cfg.expt: self.cfg.expt.delay = 0.0
        if 'phase' not in self.cfg.expt: self.cfg.expt.phase = 0.0
        if 'sigma_cutoff' not in self.cfg.expt: self.cfg.expt.sigma_cutoff=3
        if 'ramp' not in self.cfg.expt: self.cfg.expt.ramp = 0.1

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
        self.load_pulse_and_run(type=self.cfg.expt.pulse_type,delay=self.cfg.expt.delay,sigma=xpts[0],sigma_cutoff=self.cfg.expt.sigma_cutoff,
            amp=1/divN,ramp=self.cfg.expt.ramp,phase=self.cfg.expt.phase)

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
            for sigma in tqdm(xpts, disable=not progress,desc='%d/%d'%(i+1,self.cfg.expt['reps']),leave=False):
                #update delay
                # amplitude is handled in parent loop
                self.load_pulse_and_run(type=self.cfg.expt.pulse_type,delay=self.cfg.expt.delay,sigma=sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
                    amp=1/self.cfg.expt.divN,phase=self.cfg.expt.phase,quiet=True)
                #wait for tek amplitude to set
                time.sleep(self.cfg.hardware.awg_update_time)
                
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

        plt.subplot(212, xlabel="sigma [ns]", ylabel="Pulse Amp [mVpp]")
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


class AmplitudeLengthRabiSegmentExperiment(mmPulseExperiment):
    """
    Amplitude-Length Rabi Experiment with individual amplitude sweeps
    amplitude is set by awg when possible
    Requires Experimental Config
    expt = dict(
        start_gain: start gain (mVpp), 
        step_gain: stop gain (mVpp),
        start_sigma: start sigma [ns], 
        step_sigma: stop sigma [ns],
        expts_gain: number of gain points,
        expts_sigma: number of sigma points,
        nPtavg: point averages (fastest),
        nAvgs: points in sweep averages (fast),
        pulse_type: 'gauss' or 'square'
        )
    """

    def __init__(self, InstrumentDict, path='', prefix='LengthAmplitudeRabiSegment', config_file=None, progress=None,**kwargs):
        super().__init__(InstrumentDict, path=path, prefix=prefix, config_file=config_file, progress=progress,**kwargs)
        self.pulses_plotted=False
        self.maxTargetTemp=0.90

    #override
    def acquire(self, progress=False,sub_progress=False,plot_pulse=False,start_on=False,leave_on=False,sub_plot=True):
        if 'stop_sigma' in self.cfg.expt:
            if 'step_gain' in self.cfg.expt:
                self.cfg.expt["expts_sigma"] = 1+int(np.ceil(abs(self.cfg.expt["stop_sigma"]-self.cfg.expt["start_sigma"])/self.cfg.expt["step_sigma"]))
            else:
                self.cfg.expt["step_sigma"] = abs(self.cfg.expt["stop_sigma"]-self.cfg.expt["start_sigma"])/self.cfg.expt["expts_sigma"]
        xpts=self.cfg.expt["start_sigma"] + self.cfg.expt["step_sigma"]*np.arange(self.cfg.expt["expts_sigma"])
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

        #default values
        if 'delay' not in self.cfg.expt: self.cfg.expt.delay = 0.0
        if 'phase' not in self.cfg.expt: self.cfg.expt.phase = 0.0
        if 'sigma_cutoff' not in self.cfg.expt: self.cfg.expt.sigma_cutoff=3
        if 'ramp' not in self.cfg.expt: self.cfg.expt.ramp = 0.1

        #turn on and stabilize
        if not start_on:
            self.on(quiet = not progress,tek=True)

        self.data={"xpts":xpts,"gpts":gpts,"avgi":[], "avgq":[], "amps":[], "phases":[]}

        for sigma in tqdm(xpts, disable=not progress):
            #update sigma
            self.cfg.expt.sigma = sigma

            while True:
                #avoid collecting data if already warm
                self.waitForCycle(**self.fridge_config)
                #collect data
                result = self.acquire_pt(gpts,plot_pulse=plot_pulse,progress=sub_progress)
                if self.waitForCycle(**self.fridge_config):
                    break   #the temperature is good. Proceed
            self.data["avgi"].append(result["avgi"])
            self.data["avgq"].append(result["avgq"])
            self.data["amps"].append(result["amps"])
            self.data["phases"].append(result["phases"])

            if sub_plot:
                plt.figure(figsize=(12,5))
                plt.subplot(111, title=f"Length Rabi {sigma}ns",xlabel="Pulse Sigma (ns)", ylabel="Amp (Receiver B)")
                plt.plot(gpts,result["amps"],'o-')
                plt.show()

        
        #self.data["xpts"] = xpts #1D array
        #self.data["gpts"] = gpts #1D array

        for k, a in self.data.items():
            self.data[k]=np.array(a)
        
        #self.data=data
        #turn off if finished
        if not leave_on:
            self.off(quiet = not progress)

        return self.data

    def acquire_pt(self,gpts,plot_pulse=False,progress=True):
        #figure out first domain
        divN0 = 1
        awg_gain = gpts[0]
        while awg_gain < 0.25:
            divN0 = divN0*2
            awg_gain = gpts[0]*divN0
        print(f'first Gain: {gpts[0]} Amplitude domain: 1/{divN0} AWG amp: {awg_gain} Output: {awg_gain/divN0}')
        #store in config
        self.cfg.expt.divN0 = divN0
        self.cfg.expt.awg_gain = awg_gain

        data={"gpts":np.array(gpts), "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for i in tqdm(range(self.cfg.expt["reps"]),disable=not progress):
            divN = divN0
            #load the first pulse
            self.tek.set_amplitude(1,self.cfg.expt.awg_gain)
            self.load_pulse_and_run(type=self.cfg.expt.pulse_type,delay=self.cfg.expt.delay,sigma=self.cfg.expt.sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
                amp=1/divN,ramp=self.cfg.expt.ramp,phase=self.cfg.expt.phase,quiet=True)

            if plot_pulse and not self.pulses_plotted: 
                self.plot_pulses()
                self.pulses_plotted=True
            
            data_shot={"avgi":[], "avgq":[], "amps":[], "phases":[]}

            for a in tqdm(gpts, disable=not progress,desc='%d/%d'%(i+1,self.cfg.expt['reps']),leave=False):
                #update amplitude
                awg_gain=np.round(a*divN,3) # min step is 1mV
                if awg_gain>0.5:
                    divN=divN//2
                    awg_gain=np.round(a*divN,3)
                    self.tek.stop()
                    self.tek.set_amplitude(1,awg_gain)
                    self.load_pulse_and_run(type=self.cfg.expt.pulse_type,delay=self.cfg.expt.delay,sigma=self.cfg.expt.sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
                        amp=1/divN,ramp=self.cfg.expt.ramp,phase=self.cfg.expt.phase,quiet=True)
                else:
                    self.tek.set_amplitude(1,awg_gain)
                #wait for tek amplitude to set
                time.sleep(self.cfg.hardware.awg_update_time)

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
        y_sweep = data['xpts']
        x_sweep = data['gpts']
        if ampPhase:
            avgi = data['amps']
            avgq = data['phases']
        else:
            avgi = data['avgi']
            avgq = data['avgq']

        plt.figure(figsize=(10,8))
        plt.subplot(211, title="Amplitude Rabi", ylabel="Sigma [ns]")
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

        plt.subplot(212, xlabel="Pulse Amp [mVpp]", ylabel="Sigma [ns]")
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