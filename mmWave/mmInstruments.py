# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 09:02:48 2021
@author: aanferov

A library for shared code used in millimeter wave experiments

"""
import time
import numpy as np
from instruments.PNAX import N5242A

# --------------------------------------------------------------------------------------------
# N5242A wrapper class

class PNAX(N5242A):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def setDir(self,dataPath='S:\\Sasha\\Data\\',dataFolder=None,dataSubFolder=None,dataSubSubFolder=None):
        self.write('MMEM:CDIR \"'+dataPath+(dataFolder is not None and (str(dataFolder) + '\\') or '')+(dataSubFolder is not None and (str(dataSubFolder) + '\\') or '')+
            (dataSubSubFolder is not None and (str(dataSubSubFolder) + '\\') or '')+'\\\"')

    def getDir(self):
        return self.query('MMEM:CDIR?')
        
    def saveCSV(self,filename,dataPath='S:\\Sasha\\Data\\',dataFolder=None,dataSubFolder=None,dataSubSubFolder=None,log=True):
        print('Saving as: '+dataPath+(dataFolder is not None and (str(dataFolder) + '\\') or '')+(dataSubFolder is not None and (str(dataSubFolder) + '\\') or '')+
            (dataSubSubFolder is not None and (str(dataSubSubFolder) + '\\') or '')+filename)
        self.write('MMEM:STORE:DATA \"'+dataPath+(dataFolder is not None and (str(dataFolder) + '\\') or '')+(dataSubFolder is not None and (str(dataSubFolder) + '\\') or '')+
            (dataSubSubFolder is not None and (str(dataSubSubFolder) + '\\') or '')+filename+'\",\"CSV Formatted Data\",\"displayed\",\"RI\",-1')

    def saveCSVhere(self,filename):
        self.write('MMEM:STORE:DATA \"'+filename+'\",\"CSV Formatted Data\",\"displayed\",\"RI\",-1')
    

# --------------------------------------------------------------------------------------------
#meta instruments (as a class)

class multiplier():
    def __init__(self,sc,bk,channel=2,voltage=8.0,chAtt=1,rangeAtt=10.0,power=-18.0,freq=75.0,max_power=-10.0):
        #variable attenuator setup
        self.chAtt=chAtt
        self.rangeAtt=rangeAtt
        #multiplier setup
        self.channel=channel
        self.voltage=voltage
        self.power=min(max_power,power)
        self.max_power=max_power
        #convert freq to ghz
        if freq>1e9:
            freq = freq/1e9
        self.freq=freq
        self.updateInstruments(sc,bk)
        #setup normal powers and frequency
        self.sc.set_power(self.power)
        self.sc.set_frequency(self.freq*1e9/6.0)
    
    def updateInstruments(self,sc,bk):
        #cache references to instruments
        self.sc = sc
        self.bk = bk

    def on(self,delay=2.0,stabilize=20.0,initialize=False):
        #turn on power
        self.bk.set_voltage(self.channel,self.voltage)
        self.bk.set_output(True, channel=self.channel)
        #turn on attenuator
        self.bk.set_output(True, channel=self.chAtt)
        time.sleep(delay)
        print(self.bk.query('MEAS:ALL?'))
        if initialize:
            #setup normal powers and frequency
            self.sc.set_power(self.power)
            self.sc.set_frequency(self.freq*1e9/6.0)
        #turn on RF power
        self.sc.set_output_state(True)
        self.sc.set_standby(False)
        print('Stabilizing Multiplier, %ds'%stabilize)
        for t in range(20):
            time.sleep(stabilize/20)

    def off(self,delay=2.0):
        self.sc.set_output_state(False)
        self.sc.set_standby(True)
        time.sleep(delay)
        self.bk.set_voltage(self.channel,0.0)
        time.sleep(0.5)
        print(self.bk.query('MEAS:ALL?'))

    def set_frequency(self, freq, acknowledge = True):
        #convert freq to ghz
        if freq>1e9:
            freq = freq/1e9
        if freq> 60 and freq<130:
            self.freq = freq
            self.sc.set_frequency(freq=self.freq*1e9/6.0,acknowledge=acknowledge)
        else:
            print('Warning, could not set frequency out of range.')

    def set_power(self,power):
        #power in dBm
        if power <= self.max_power:
            self.sc.set_power(power)
        else:
            print('Error: could not set power (too high)!')
    
    def set_attenuation(self,voltage):
        #attenuator voltage in volts
        if voltage <= self.rangeAtt and voltage >= 0.0:
            self.bk.set_voltage(self.chAtt,voltage)
        else:
            print('Error: could not set attenuator voltage (out of bounds)!')


class mixer():
    def __init__(self,sc,bk,power=-3.0,freq=75.0,chAtt=1,rangeAtt=10.0,chAmp=2,vAmp=8.0,max_power=-3.0):
        #variable attenuator setup
        self.chAtt=chAtt
        self.rangeAtt=rangeAtt
        #amplifier setup
        self.chAmp=chAmp
        self.vAmp=vAmp
        #signalcore setup
        self.power=min(max_power,power)
        self.max_power=max_power
        #convert freq to ghz
        if freq>1e9:
            freq = freq/1e9
        self.freq=freq
        self.updateInstruments(sc,bk)
    
    def updateInstruments(self,sc,bk):
        #cache references to instruments
        self.sc = sc
        self.bk = bk

    def on(self,stabilize=20.0,initialize=False):
        #turn on amp
        self.bk.set_voltage(self.chAmp,self.vAmp)
        self.bk.set_output(True, channel=self.chAmp)
        
        if initialize:
            #setup multiplier output
            self.bk.set_voltage(self.chAtt,0.0)
            self.bk.set_output(True, channel=self.chAtt)    #this might actually turn on all channels in bk...
            #setup normal powers and frequency
            self.sc.set_power(self.power)
            self.sc.set_frequency(self.freq*1e9/6.0)
        #turn on RF power
        self.sc.set_output_state(True)
        self.sc.set_standby(False)
        print('Stabilizing Multiplier, %ds'%stabilize)
        for t in range(20):
            time.sleep(stabilize/20)

    def off(self):
        self.sc.set_output_state(False)
        self.sc.set_standby(True)
        self.bk.set_voltage(self.chAmp,0.0)

    def set_frequency(self, freq, acknowledge = True):
        #convert freq to ghz
        if freq>1e9:
            freq = freq/1e9
        if freq> 60 and freq<120:
            self.freq = freq
            self.sc.set_frequency(freq=self.freq*1e9/6.0,acknowledge=acknowledge)
        else:
            print('Warning, could not set frequency out of range.')

    def set_power(self,power):
        #power in dBm
        if power <= self.max_power:
            self.sc.set_power(power)

    def set_attenuation(self,voltage):
        #attenuator voltage in volts
        if voltage <= self.rangeAtt and voltage >= 0.0:
            self.bk.set_voltage(self.chAtt,voltage)