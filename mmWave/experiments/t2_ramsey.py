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

#for tek70001a
from experiments.PulseExperiments.sequencer import Sequencer
from experiments.PulseExperiments.pulse_classes import Gauss, Square, Idle, Zeroes
from instruments.awg.Tek70001 import write_Tek70001_sequence, write_Tek70001_sequence_batch

from experiments.mmWave.mmExperiments import mmPulseExperiment, floor10

class RamseyT2Experiment(mmPulseExperiment):
    """
    T2 Ramsey Experiment where a new sequence is written for every point
    Requires Experimental Config
    expt = dict(
        start: start delay [ns] (positive), 
        step: step delay [ns],
        expts: number of experiments,
        nPtavg: point averages (fastest),
        nAvgs: points in sweep averages (fast),
        reps: start-finish experiment repeats (extremely slow),
        pulse_type: 'gauss' or 'square',
        sigma: pulse length [ns] (default: from pi_ge in config)
        gain: pulse amplitude [mVpp]
        ramp: pulse sigma for square pulses
        )
    """

    def __init__(self, InstrumentDict, path='', prefix='LengthRabi', config_file=None, progress=None,**kwargs):
        super().__init__(InstrumentDict, path=path, prefix=prefix, config_file=config_file, progress=progress,**kwargs)
        self.pulses_plotted=False
        self.pulses_loaded=False

    #override parent class!
    #for loading and running one pulse of many
    def write_pulse_batch(self,number=0,ramsey_time=0.0,ramsey_freq=0.0,ramsey_phase=None,pulses=None,delay=0.0,type=None,sigma=None,sigma_cutoff=2,amp=1.0,ramp=0.1,phase=0,pulse_name=None,pulse_count=None,quiet=False):
        #override for more complicated pulses
        self.awg_dt = self.cfg.hardware.awg_info.tek70001a.dt
        self.sequencer = Sequencer(list(self.cfg.hardware.awg_channels.keys()),self.cfg.hardware.awg_channels,self.cfg.hardware.awg_info,{})
        self.multiple_sequences = []
        if pulse_count is None:
            pulse_count = self.cfg['hardware']['pulse_count']
        if pulses is not None:
            #feed the pulses into the sequencer
            for pulse in pulses:
                self.sequencer.append('Ch1',pulse)
        else:
            pulse = None
            #define pulse 1
            if type == 'square':
                #Square(max_amp, flat_len, ramp_sigma_len, cutoff_sigma, freq, phase, phase_t0 = 0, dt=None)
                pulse = Square(amp,sigma,ramp,sigma_cutoff,self.cfg.device.qubit.if_freq,phase,dt=self.awg_dt)
            elif type == 'gauss':
                pulse = Gauss(amp,sigma,sigma_cutoff,self.cfg.device.qubit.if_freq,phase)
            else:
                print('Error: could not interpret pulse type!')
            if pulse is not None: self.pulse_length = pulse.get_length() # in ns?

            #pulse is too long! may need to auto-adjust the trig delay (needs to be a multiple of 10)
            if floor10(self.cfg.hardware.awg_offset) - (self.cfg.hardware.awg_trigger_time % 10) < 2*self.pulse_length+delay+ramsey_time:
                print('Warning: pulse is longer than awg_offset!')

            pulse.generate_pulse_array(dt=self.cfg['hardware']['awg_info']['tek70001a']['dt'])
            #exact length in ns
            pulse_len = len(pulse.t_array)*self.cfg['hardware']['awg_info']['tek70001a']['dt']
            ramsey_wait = Idle(ramsey_time)
            ramsey_wait.generate_pulse_array(dt=self.cfg['hardware']['awg_info']['tek70001a']['dt'])

            pulse2 = None
            if ramsey_phase is None:
                phase2 = phase+2*np.pi*ramsey_freq*ramsey_time
            else:
                phase2 = phase + ramsey_phase
            #define pulse 2
            if type == 'square':
                #Square(max_amp, flat_len, ramp_sigma_len, cutoff_sigma, freq, phase, phase_t0 = 0, dt=None)
                #pulse2 = Square(amp,sigma,ramp,sigma_cutoff,self.cfg.device.qubit.if_freq,phase2,phase_t0=-pulse_len-ramsey_time,dt=self.awg_dt)
                pulse2 = Square(amp,sigma,ramp,sigma_cutoff,self.cfg.device.qubit.if_freq,phase2,dt=self.awg_dt)
            elif type == 'gauss':
                print('Error: Gauss pulse not working yet')
                pulse2 = Gauss(amp,sigma,sigma_cutoff,self.cfg.device.qubit.if_freq,phase)
            else:
                print('Error: could not interpret pulse type!')

            

            #initial_delay = floor10(self.cfg.hardware.awg_offset) - (self.cfg.hardware.awg_trigger_time % 10) -pulse.get_length()-delay
            initial_points = int(self.cfg['hardware']['awg_info']['tek70001a']['inverse_dt'])*int(np.round(floor10(self.cfg.hardware.awg_offset) - (self.cfg.hardware.awg_trigger_time % 10)-delay))-2*len(pulse.t_array)-len(ramsey_wait.t_array)

            self.sequencer.new_sequence(points=initial_points)
            self.sequencer.append('Ch1', pulse)
            self.sequencer.append('Ch1',ramsey_wait)
            self.sequencer.append('Ch1', pulse2)

            total_points = len(pulse.t_array)+len(ramsey_wait.t_array)+len(pulse2.t_array)
            inter_pulse_points = int(floor10(self.cfg['hardware']['period']))*int(self.cfg['hardware']['awg_info']['tek70001a']['inverse_dt'])-total_points
            for i in range(pulse_count-1):
                #self.sequencer.append('Ch1',Idle(inter_pulse_delay))
                self.sequencer.append('Ch1',Zeroes(inter_pulse_points))
                self.sequencer.append('Ch1', pulse)
                self.sequencer.append('Ch1',ramsey_wait)
                self.sequencer.append('Ch1', pulse2)
        self.sequencer.end_sequence(self.cfg['hardware']['awg_padding'])# padding
        self.multiple_sequences=self.sequencer.complete()

        if pulse_name is None:
            pulse_name='%s_%.2fGHz'%(type,self.cfg.device.qubit.if_freq)
        write_Tek70001_sequence_batch(number,[seq['Ch1'] for seq in self.multiple_sequences],os.path.join(self.path, self.seqFolder), pulse_name,awg=self.tek,quiet=quiet)
        #note: need to do tek.prep_experiment
        #note need to do tek.run after this

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
        if 'delay' not in self.cfg.expt: self.cfg.expt.delay = 0.0
        if 'sigma_cutoff' not in self.cfg.expt: self.cfg.expt.sigma_cutoff=2
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
        
        if not self.pulses_loaded:
            #load waveforms
            self.tek.pre_load()
            for exp_n,t in tqdm(enumerate(xpts),disable=not progress,desc='Loading Waveforms'):
                self.write_pulse_batch(exp_n,ramsey_time=t,ramsey_freq=self.cfg.expt.ramsey_freq,ramsey_phase=None,
                                    type=self.cfg.expt.pulse_type,delay=self.cfg.expt.delay,sigma=self.cfg.expt.sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
                                    amp=1/divN,ramp=self.cfg.expt.ramp,phase=self.cfg.expt.phase,quiet=True)
            self.pulses_loaded=True
        

        data={"xpts":np.array(xpts), "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for i in tqdm(range(self.cfg.expt["reps"]),disable=not progress):

            while True:
                #avoid collecting data if already warm
                self.waitForCycle(**self.fridge_config)
                #collect data
                data_shot={"avgi":[], "avgq":[], "amps":[], "phases":[]}
                for exp_n in tqdm(range(len(xpts)), disable=not progress,desc='%d/%d'%(i+1,self.cfg.expt['reps']),leave=False):
                    #update delay
                    #self.tek.stop()
                    self.tek.set_amplitude(1,awg_gain)
                    #self.load_pulse_and_run(type=self.cfg.expt.pulse_type,delay=self.cfg.expt.delay,sigma=sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
                    #    amp=1/divN,phase=self.cfg.expt.phase,quiet=True)
                    #load experiment waveform
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
                    
                if self.waitForCycle(**self.fridge_config):
                    break   #the temperature is good. Proceed
            
            
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

    def display(self, data=None, ampPhase=False,fit=True, **kwargs):
        if data is None:
            data=self.data 
        plt.figure(figsize=(10,8))
        if ampPhase:
            plt.subplot(211, title="T2 Ramsey", ylabel="Amp (Receiver B)")
            plt.plot(data["xpts"], data["amps"],'o-')
        else:
            plt.subplot(211, title="T2 Ramsey", ylabel="I (Receiver B)")
            plt.plot(data["xpts"], data["avgi"],'o-')
        if fit:
            pass
        if ampPhase:
            plt.subplot(212, xlabel="Delay (ns)", ylabel="Phases (Receiver B)")
            plt.plot(data["xpts"], data["phases"],'o-')
        else:
            plt.subplot(212, xlabel="Delay (ns)", ylabel="Q (Receiver B)")
            plt.plot(data["xpts"], data["avgq"],'o-')
        if fit:
            pass
        plt.tight_layout()
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)

class DoubleLengthRabiExperiment(RamseyT2Experiment):
    """
    Length rabi Experiment with 2 pi/2 pulses instead of 1
    a new sequence is written for every point
    Requires Experimental Config
    expt = dict(
        start: start delay [ns] (positive), 
        step: step delay [ns],
        expts: number of experiments,
        nPtavg: point averages (fastest),
        nAvgs: points in sweep averages (fast),
        reps: start-finish experiment repeats (extremely slow),
        pulse_type: 'gauss' or 'square',
        sigma: pulse length [ns] (default: from pi_ge in config)
        gain: pulse amplitude [mVpp]
        ramp: pulse sigma for square pulses
        )
    """

    def __init__(self, InstrumentDict, path='', prefix='LengthRabi', config_file=None, progress=None,**kwargs):
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

        #if 'sigma' not in self.cfg.expt: self.cfg.expt.sigma = self.cfg.device.qubit.pulses.pi_ge.sigma
        if 'gain' not in self.cfg.expt: self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_ge.gain
        #default values
        if 'phase' not in self.cfg.expt: self.cfg.expt.phase = 0.0
        if 'delay' not in self.cfg.expt: self.cfg.expt.delay = 0.0
        if 'sigma_cutoff' not in self.cfg.expt: self.cfg.expt.sigma_cutoff=2
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
        for exp_n,sigma in tqdm(enumerate(xpts),disable=not progress,desc='Loading Waveforms'):
            self.write_pulse_batch(exp_n,ramsey_time=0.0,ramsey_freq=0.0,ramsey_phase=None,
                                   type=self.cfg.expt.pulse_type,delay=self.cfg.expt.delay,sigma=sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
                                   amp=1/divN,ramp=self.cfg.expt.ramp,phase=self.cfg.expt.phase,quiet=True)

        

        data={"xpts":np.array(xpts), "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for i in tqdm(range(self.cfg.expt["reps"]),disable=not progress):
            while True:
                #avoid collecting data if already warm
                self.waitForCycle(**self.fridge_config)
                #collect data
                data_shot={"avgi":[], "avgq":[], "amps":[], "phases":[]}
                for exp_n in tqdm(range(len(xpts)), disable=not progress,desc='%d/%d'%(i+1,self.cfg.expt['reps']),leave=False):
                    #update delay
                    #self.tek.stop()
                    self.tek.set_amplitude(1,awg_gain)
                    #self.load_pulse_and_run(type=self.cfg.expt.pulse_type,delay=self.cfg.expt.delay,sigma=sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
                    #    amp=1/divN,phase=self.cfg.expt.phase,quiet=True)
                    #load experiment waveform
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
                    
                if self.waitForCycle(**self.fridge_config):
                    break   #the temperature is good. Proceed
            
            
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

    def display(self, data=None, ampPhase=False,fit=True, **kwargs):
        if data is None:
            data=self.data 
        plt.figure(figsize=(10,8))
        if ampPhase:
            plt.subplot(211, title="Double Length Rabi", ylabel="Amp (Receiver B)")
            plt.plot(data["xpts"], data["amps"],'o-')
        else:
            plt.subplot(211, title="Double Length Rabi", ylabel="I (Receiver B)")
            plt.plot(data["xpts"], data["avgi"],'o-')
        if fit:
            pass
        if ampPhase:
            plt.subplot(212, xlabel="Sigma (ns)", ylabel="Phases (Receiver B)")
            plt.plot(data["xpts"], data["phases"],'o-')
        else:
            plt.subplot(212, xlabel="Sigma (ns)", ylabel="Q (Receiver B)")
            plt.plot(data["xpts"], data["avgq"],'o-')
        if fit:
            pass
        plt.tight_layout()
        plt.show()

class DoubleAmpRabiExperiment(RamseyT2Experiment):
    """
    Amp rabi Experiment with 2 pi/2 pulses instead of 1
    a new sequence is written for every point
    Requires Experimental Config
    expt = dict(
        start: start gain (mVpp), 
        step: stop gain (mVpp),
        expts: number of experiments,
        nPtavg: point averages (fastest),
        nAvgs: points in sweep averages (fast),
        reps: start-finish experiment repeats (extremely slow),
        pulse_type: 'gauss' or 'square',
        sigma: pulse length [ns] (default: from pi_ge in config)
        gain: pulse amplitude [mVpp]
        ramp: pulse sigma for square pulses
        )
    """

    def __init__(self, InstrumentDict, path='', prefix='LengthRabi', config_file=None, progress=None,**kwargs):
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

        #if 'sigma' not in self.cfg.expt: self.cfg.expt.sigma = self.cfg.device.qubit.pulses.pi_ge.sigma
        if 'gain' not in self.cfg.expt: self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_ge.gain
        #default values
        if 'phase' not in self.cfg.expt: self.cfg.expt.phase = 0.0
        if 'delay' not in self.cfg.expt: self.cfg.expt.delay = 0.0
        if 'sigma_cutoff' not in self.cfg.expt: self.cfg.expt.sigma_cutoff=2
        if 'ramp' not in self.cfg.expt: self.cfg.expt.ramp = 0.1
        

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
            self.on(quiet = not progress,stabilize_time=2,tek=False)
        
        #load waveforms
        self.tek.pre_load()
        self.write_pulse_batch(divN0,ramsey_time=0.0,ramsey_freq=0.0,ramsey_phase=None,
                                type=self.cfg.expt.pulse_type,delay=self.cfg.expt.delay,sigma=self.cfg.expt.sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
                                amp=1/divN0,ramp=self.cfg.expt.ramp,phase=self.cfg.expt.phase,quiet=True)
        divN=divN0
        for a in tqdm(xpts,disable=not progress,desc='Loading Waveforms'):
            awg_gain=np.round(a*divN,3) # min step is 1mV
            if awg_gain>0.5:
                divN=divN//2
                self.write_pulse_batch(divN,ramsey_time=0.0,ramsey_freq=0.0,ramsey_phase=None,
                                        type=self.cfg.expt.pulse_type,delay=self.cfg.expt.delay,sigma=self.cfg.expt.sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
                                        amp=1/divN,ramp=self.cfg.expt.ramp,phase=self.cfg.expt.phase,quiet=True)
        #self.load_experiment_and_run(divN)
        

        data={"xpts":np.array(xpts), "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for i in tqdm(range(self.cfg.expt["reps"]),disable=not progress):
            while True:
                #avoid collecting data if already warm
                self.waitForCycle(**self.fridge_config)
                #collect data
                divN = self.cfg.expt.divN0
                #load the first pulse
                self.tek.set_amplitude(1,self.cfg.expt.awg_gain)
                #self.load_pulse_and_run(type=self.cfg.expt.pulse_type,delay=self.cfg.expt.delay,sigma=self.cfg.expt.sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
                #    amp=1/divN,ramp=self.cfg.expt.ramp,phase=self.cfg.expt.phase)
                self.load_experiment_and_run(divN)
                #plot pulses if first time
                if plot_pulse and not self.pulses_plotted: 
                    self.plot_pulses()
                    self.pulses_plotted=True

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
                        #self.load_pulse_and_run(type=self.cfg.expt.pulse_type,delay=self.cfg.expt.delay,sigma=self.cfg.expt.sigma,sigma_cutoff=self.cfg.expt.sigma_cutoff,
                        #    amp=1/divN,ramp=self.cfg.expt.ramp,phase=self.cfg.expt.phase,quiet=True)
                        self.load_experiment_and_run(divN)
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
                    
                if self.waitForCycle(**self.fridge_config):
                    break   #the temperature is good. Proceed
            
            
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

    def display(self, data=None, ampPhase=False,fit=True, **kwargs):
        if data is None:
            data=self.data 
        plt.figure(figsize=(10,8))
        if ampPhase:
            plt.subplot(211, title="Double Amplitude Rabi", ylabel="Amp (Receiver B)")
            plt.plot(data["xpts"], data["amps"],'o-')
        else:
            plt.subplot(211, title="Double Amplitude Rabi", ylabel="I (Receiver B)")
            plt.plot(data["xpts"], data["avgi"],'o-')
        if fit:
            pass
        if ampPhase:
            plt.subplot(212, xlabel="Gain (mVpp)", ylabel="Phases (Receiver B)")
            plt.plot(data["xpts"], data["phases"],'o-')
        else:
            plt.subplot(212, xlabel="Gain (mVpp)", ylabel="Q (Receiver B)")
            plt.plot(data["xpts"], data["avgq"],'o-')
        if fit:
            pass
        plt.tight_layout()
        plt.show()