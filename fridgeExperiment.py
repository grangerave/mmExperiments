# coding: utf-8 
"""
Created on Fri Mar 12 09:02:48 2021
@author: aanferov

A fridge experiment class which checks the temperature before saving the data

"""
import time
import pycurl
from io import BytesIO
import json
from urllib.parse import urlencode

from experiments.experiment import Experiment

# --------------------------------------------------------------------------------------------
# fridge functions (eventually these should go into an instrument file)

def getCurlTemp():
    try:
        buffer = BytesIO()
        crl = pycurl.Curl()
        crl.setopt(crl.URL,'http://raspberrium/flask/get/temps')
        crl.setopt(crl.WRITEFUNCTION,buffer.write)
        crl.perform()
        crl.close()
        temps = json.loads(buffer.getvalue().decode('utf-8'))
        if temps['1k_plate']>0.1:
            return temps['1k_plate']
        else:
            return temps['4k_plate']
    except:
        #try again a bit later
        time.sleep(1.0)
        try:
            buffer = BytesIO()
            crl = pycurl.Curl()
            crl.setopt(crl.URL,'http://raspberrium/flask/get/temps')
            crl.setopt(crl.WRITEFUNCTION,buffer.write)
            crl.perform()
            crl.close()
            temps = json.loads(buffer.getvalue().decode('utf-8'))
            if temps['1k_plate']>0.1:
                return temps['1k_plate']
            else:
                return temps['4k_plate']
        except:
            print('error getting temp!')
            return 4


def setFridgeState(state='NONE'):
    try:
        crl = pycurl.Curl()
        crl.setopt(crl.URL,'http://raspberrium/flask/set/automation/state')
        crl.setopt(crl.POSTFIELDS,urlencode({'state':state}))
        crl.perform()
        crl.close()
    except:
        print('failed to set state')
        
def setFridgeSubroutine(subroutine='NONE'):
    try:
        crl = pycurl.Curl()
        crl.setopt(crl.URL,'http://raspberrium/flask/set/automation/subroutine')
        crl.setopt(crl.POSTFIELDS,urlencode({'subroutine':subroutine}))
        crl.perform()
        crl.close()
    except:
        print('failed to set subroutine')

# --------------------------------------------------------------------------------------------
# temp and cycle functions
def waitForTempBelow(maxTargetTemp,nTimeSteps=30,tempCheckDelay=500,quiet=False):
    #wait until cycle is finished by checking temp
    while getCurlTemp()>maxTargetTemp:
        if not quiet:print('.',end='')
        for i in range(nTimeSteps):time.sleep(tempCheckDelay/nTimeSteps) #wait check time

def waitForTempAbove(minTargetTemp,nTimeSteps=30,tempCheckDelay=500,quiet=False):
    #wait until cycle is finished by checking temp
    while getCurlTemp()<minTargetTemp:
        if not quiet:print('.',end='')
        for i in range(nTimeSteps):time.sleep(tempCheckDelay/nTimeSteps) #wait check time


def waitForCycle(offFunc=None,onFunc=None,maxTargetTemp=1.2,maxDesorbTemp=4.7,nTimeSteps=30,desorbTime=1800,tempCheckDelay=500,finalDelay=120,quiet=False):
    # check that temperature didn't start warming up
    # normally returns True (for use in while loops)

    # intended use:
    # while True:
    #   takeData()
    #   if waitForCycle():
    #       break
    # saveData()

    if getCurlTemp()<maxTargetTemp:
        return True
        
    if not quiet:print('Temperature too high! Waiting for cycle ...')
    if offFunc is not None:offFunc()
    
    #wait (total of 30 min) without checking temp to make sure cycle is in desorb phase
    for i in range(nTimeSteps):
        if not quiet:print('.',end='')
        time.sleep(desorbTime/nTimeSteps)
    
    #verify cycle is in desorb phase by checking temp
    waitForTempBelow(maxDesorbTemp,nTimeSteps=nTimeSteps,tempCheckDelay=tempCheckDelay)
    if not quiet:print('Cycle Desorbing.')
    
    #calibrations go here
    #TODO run calibration

    #wait until cycle is finished by checking temp
    waitForTempBelow(maxTargetTemp,nTimeSteps=nTimeSteps,tempCheckDelay=tempCheckDelay)

    for i in range(nTimeSteps):time.sleep(finalDelay/nTimeSteps) #wait final delay
    print('Cycle finished. Resuming...')
    if onFunc is not None:onFunc()
    #flag the parent loop that we had to wait for a cycle (should discard the last datapoint)
    return False


def acquireDuringCycle(acquire,onFunc=None,offFunc=None,maxTargetTemp=1.2,maxDesorbTemp=4.7,nTimeSteps=30,desorbTime=1800,tempCheckDelay=500,finalDelay=120,quiet=False,**kwargs):
    #avoid collecting data if already warm
    waitForCycle(onFunc=onFunc,offFunc=offFunc,maxTargetTemp=maxTargetTemp,maxDesorbTemp=maxDesorbTemp,nTimeSteps=nTimeSteps,desorbTime=desorbTime/2,tempCheckDelay=tempCheckDelay,finalDelay=finalDelay,quiet=quiet)
    #call the acquire function with the given arguments
    while True:
        result = acquire(**kwargs)
        if waitForCycle(onFunc=onFunc,offFunc=offFunc,maxTargetTemp=maxTargetTemp,maxDesorbTemp=maxDesorbTemp,nTimeSteps=nTimeSteps,desorbTime=desorbTime,tempCheckDelay=tempCheckDelay,finalDelay=finalDelay,quiet=quiet):
            break   #the temperature is good
    return result
            
# --------------------------------------------------------------------------------------------
#   support for turning on/off power when waiting
#   wrapper for experiment class
#   call acquireInCycle() instead of acquire()

class FridgeExperiment(Experiment):
    def __init__(self,fridge_config=None,**kwargs):
        super().__init__(**kwargs)
        if fridge_config is None: self.fridge_config = dict(powerOffonWait=True,maxTargetTemp=1.2,maxDesorbTemp=4.7,nTimeSteps=30,desorbTime=1800,tempCheckDelay=500,finalDelay=120)

    def on(self,quiet=False):
        if not quiet: print('Turning Instruments ON')

    def off(self,quiet=False):
        if not quiet: print('Turning Instruments OFF')

    def waitForCycle(self,powerOffonWait=True,maxTargetTemp=1.2,maxDesorbTemp=4.7,nTimeSteps=30,desorbTime=600,tempCheckDelay=500,finalDelay=120,quiet=False):
        # check that temperature didn't start warming up. normally returns True (for use in while loops)
        #print(locals())
        if getCurlTemp()<maxTargetTemp:
            return True #temperature is correct
            
        if not quiet:print('Temperature too high! Waiting for cycle ...')
        if powerOffonWait:self.off(quiet)
        
        #wait (total of 30 min) without checking temp to make sure cycle is in desorb phase
        for i in range(nTimeSteps):
            if not quiet:print('.',end='')
            time.sleep(desorbTime/nTimeSteps)
        
        #verify cycle is in desorb phase by checking temp
        waitForTempBelow(maxDesorbTemp,nTimeSteps=nTimeSteps,tempCheckDelay=tempCheckDelay)
        if not quiet:print('Cycle Desorbing.')
        
        #calibrations go here
        #TODO run calibration

        #wait until cycle is finished by checking temp
        waitForTempBelow(maxTargetTemp,nTimeSteps=nTimeSteps,tempCheckDelay=tempCheckDelay)

        for i in range(nTimeSteps):time.sleep(finalDelay/nTimeSteps) #wait final delay
        if not quiet:print('Cycle finished. Resuming...')
        if powerOffonWait:self.on(quiet)
        #flag the parent loop that we had to wait for a cycle (should discard the last datapoint)
        return False

    def acquireDuringCycle(self,powerOffonWait=None,maxTargetTemp=None,maxDesorbTemp=None,nTimeSteps=None,desorbTime=None,tempCheckDelay=None,finalDelay=None,quiet=None,**kwargs):
        #use passed arguments to override fridgeConfig
        argsPassed = locals()
        del argsPassed['kwargs']
        del argsPassed['self']
        overrides=self.fridge_config
        for k,v in argsPassed.items():
            if v is not None: overrides[k] = v
        #call our own acquire function with the given arguments
        while True:
            #avoid collecting data if already warm
            self.waitForCycle(**overrides)
            #collect data
            result = self.acquire(**kwargs)
            if self.waitForCycle(**overrides):
                break   #the temperature is good. Proceed
        return result

    def go(self, save=False, analyze=False, display=False, progress=False,**kwargs):
        # get data during cycle using the most basic settings. Pass kwargs to acquire function
        data=self.acquireDuringCycle(progress,**kwargs)
        if analyze:
            data=self.analyze(data)
        if save:
            self.save_data(data)
        if display:
            self.display(data)
                