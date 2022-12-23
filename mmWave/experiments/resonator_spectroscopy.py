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

class ResonatorSpectroscopyExperiment(mmPulseExperiment):
    """
    Resonator Spectroscopy Experiment
    Requires Experimental Config
    expt = dict(
        start: start frequency (MHz), 
        step: frequency step (MHz),
        expts: number of experiments,
        nPtavg: point averages (fastest),
        nAvgs: points in sweep averages (fast),
        #not used- 
           reps: start to finish averages (very slow)
        )
    """

    def __init__(self, InstrumentDict, path='', prefix='ResonatorSpectroscopy', config_file=None, progress=None,**kwargs):
        super().__init__(InstrumentDict, path=path, prefix=prefix, config_file=config_file, progress=progress,**kwargs)

    #override since we don't need tek or mixer
    def on(self,quiet=False):
        if not quiet: print('Turning PNAX ON')
        #self.amcMixer.on(stabilize=stabilize_time)
        self.PNAX.set_output(True)
        self.PNAX.set_sweep_mode('CONT')
        #self.tek.run()

    def off(self,quiet=False):
        if not quiet: print('Turning PNAX OFF')
        #self.tek.stop()
        self.PNAX.set_sweep_mode('HOLD')
        self.PNAX.set_output(False)
        #self.amcMixer.off()

    #override
    def acquire(self, progress=False,start_on=False,leave_on=False):
        xpts=self.cfg.expt["start"] + self.cfg.expt["step"]*np.arange(self.cfg.expt["expts"])
        self.prep()                           
        if not start_on:
            self.on(quiet = not progress)

        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}

        for f in tqdm(xpts, disable=not progress):
            self.cfg.device.readout.freq = f
            self.PNAX.set_center_frequency(f*1e9)

            self.PNAX.set_sweep_mode('SING')
            response = self.PNAX.read_data()
            avgi = np.mean(response[1])
            avgq = np.mean(response[2])
            amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phase = np.angle(avgi+1j*avgq) # Calculating the phase

            data["xpts"].append(f)
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

    def analyze(self, data=None, fit=False, findpeaks=False, verbose=True, **kwargs):
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

    def display(self, data=None, fit=True, findpeaks=False, **kwargs):
        if data is None:
            data=self.data 
        plt.figure(figsize=(18,6))
        plt.subplot(111, title=f"Resonator Spectroscopy", xlabel="Resonator Frequency (GHz)", ylabel="Amps (Reciever B)")
        
        plt.plot(data["xpts"][1:-1], data["amps"][1:-1],'o-')
        
        if fit:
            pass
            
        if findpeaks:
            # for peak in np.concatenate((data['maxpeaks'], data['minpeaks'])):
            for peak in data['minpeaks']:
                plt.axvline(peak, linestyle='--', color='0.2')
                print(f'Min Peak [GHz]: {peak}')
            
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)