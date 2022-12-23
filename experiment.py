__author__ = 'David Schuster'

# from liveplot import LivePlotClient
# from dataserver import dataserver_client
import os.path
import json
import yaml
import numpy as np
import traceback
import time

from experiments.datamanagement import SlabFile, AttrDict, get_next_filename

class NpEncoder(json.JSONEncoder):
    """ Ensure json dump can handle np arrays """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class Experiment:
    """Base class for all experiments"""

    def __init__(self, path='', dataFolder='data',prefix='test', config_file=None, liveplot_enabled=False, safeFileSaving=True, **kwargs):
        """ Initializes experiment class
            @param path - directory where data will be stored
            @param prefix - prefix to use when creating data files
            @param config_file - parameters for config file specified are loaded into the class dict
                                 (name relative to expt_directory if no leading /)
                                 Default = None looks for path/prefix.json

            @param **kwargs - by default kwargs are updated to class dict
        """

        self.__dict__.update(kwargs)
        self.path = path
        self.dataFolder=dataFolder
        self.prefix = prefix
        self.cfg = None
        if config_file is not None:
            self.config_file = os.path.join(path, config_file)
        else:
            self.config_file = None
        if safeFileSaving:
            filename = get_next_filename(os.path.join(path, dataFolder), prefix, suffix='.h5')
        else:
            filename = time.strftime('%y%m%d') + '_' + prefix + '.h5'
        self.fname = os.path.join(path, dataFolder, filename)

        self.load_config()

    def load_config(self):
        if self.config_file is None:
            self.config_file = os.path.join(self.path, self.prefix + ".json")
        try:
            if self.config_file[-3:] == '.h5':
                with SlabFile(self.config_file) as f:
                    self.cfg = AttrDict(f.load_config())
                    self.fname = self.config_file
            elif self.config_file[-4:].lower() =='.yml':
                with open(self.config_file,'r') as fid:
                    self.cfg = AttrDict(yaml.safe_load(fid))
            else:
                with open(self.config_file, 'r') as fid:
                    cfg_str = fid.read()
                    self.cfg = AttrDict(json.loads(cfg_str))

        except Exception as e:
            print("Could not load config.")
            traceback.print_exc()

    def save_config(self):
        if self.config_file[:-3] != '.h5':
            with open(self.config_file, 'w') as fid:
                json.dump(self.cfg, fid, cls=NpEncoder), 
            self.datafile().attrs['config'] = json.dumps(self.cfg, cls=NpEncoder)

    def datafile(self, group=None, remote=False, data_file = None, swmr=False):
        """returns a SlabFile instance
           proxy functionality not implemented yet"""
        if data_file ==None:
            data_file = self.fname
        if swmr==True:
            f = SlabFile(data_file, 'w', libver='latest')
        elif swmr==False:
            f = SlabFile(data_file, 'a')
        else:
            raise Exception('ERROR: swmr must be type boolean')

        if group is not None:
            f = f.require_group(group)
        if 'config' not in f.attrs:
            try:
                f.attrs['config'] = json.dumps(self.cfg, cls=NpEncoder)
            except TypeError as err:
                print(('Error in saving cfg into datafile (experiment.py):', err))

        return f

    def go(self, save=False, analyze=False, display=False, progress=False):
        # get data
        data=self.acquire(progress)
        if analyze:
            data=self.analyze(data)
        if save:
            self.save_data(data)
        if display:
            self.display(data)

    def acquire(self, progress=False, debug=False):
        pass

    def analyze(self, data=None, **kwargs):
        pass

    def display(self, data=None, **kwargs):
        pass

    def save_data(self, data=None):  #do I want to try to make this a very general function to save a dictionary containing arrays and variables?
        if data is None:
            data=self.data

        with self.datafile() as f:
            for k, d in data.items():
                f.add(k, np.array(d))
    
    def get_data(self):
        data=self.data
        return data
        
    def load_data(self, f):
        data={}
        for k in f.keys():
            data[k]=np.array(f[k])
        data['attrs']=f.get_dict()
        return data