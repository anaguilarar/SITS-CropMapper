from .decorators import check_path, check_output_fn
from typing import List, Union, Dict

import numpy as np
import json

import os

class EvaluateSuffix(object):
    @staticmethod
    def _check_json_suffix(fn):
        #assert fn[0](fn)
        return fn.endswith('.json')
            
    def __init__(self, arg) -> None:
        self._arg = arg
        
    def __call__(self, *args):
        
        if len(args) == 1:
            fn = args[0]
        else:
            fn = args[1]

        out =  self._arg(fn) if self._check_json_suffix(fn) else None
        return out
    
@EvaluateSuffix
def loadjson(fn):
    """
    Load JSON data from a file.

    Parameters
    ----------
    fn : str
        Filename of the JSON file to load.

    Returns
    -------
    dict or None
        Dictionary containing the loaded JSON data.
        Returns None if the file does not exist.
    """
    
    if os.path.exists(fn):
        with open(fn, "rb") as fn:
            reporter = json.load(fn)
    else:
        reporter = None
    return reporter



def concatenate_lists(list_of_lists: List[List]):
    """
    Concatenate sublists into strings with elements separated by dashes.

    Parameters
    ----------
    list_of_lists : List[List[Union[str, int, List[int]]]]
        A list of lists, where each sublist contains elements (strings, integers, or lists of integers)
        that need to be concatenated into a single string.

    Returns
    -------
    List[str]
        A list of strings, each representing the concatenated result of the sublists with elements
        separated by dashes.

    Examples
    --------
    >>> concatenate_lists([["a", 1, [2, 3]], ["b", 2]])
    ['a-1-2-3', 'b-2']
    """
    idsfromgroup = [] 
    for j in range(len(list_of_lists)):
        listunique = []
        for item in list_of_lists[j]:
            listunique.append('-'.join(map(str, item)) if isinstance(item,list) else str(item))
        idsfromgroup.append('-'.join(listunique))
    return(idsfromgroup)


class ReporterBase():
    """
    A class for handling reporting of key metrics during training or evaluation processes.

    Attributes
    ----------
    _previous_groups : dict
        Stores previous group computations to avoid redundant calculations.
    _report_keys : list
        A list of keys that are being reported.
    report : dict
        A dictionary to collect the reporting data.
    """
    
    def __init__(self) -> None:
        self._previous_groups= {}
        self._report_keys = []
        pass
        
    def set_reporter(self, checkpoint_keys):
        """
        Initialize the reporter with specified keys.

        Parameters
        ----------
        checkpoint_keys : List[str]
            A list of keys to report.

        Returns
        -------
        None
        """
        reporter = {}
        for keyname in checkpoint_keys:
            reporter.update({keyname: []})
        self.report = reporter
        self._update_keys(reporter)
        return reporter
    
    def _update_keys(self, reporter_dict):
        self._report_keys = [keyarg for keyarg in reporter_dict.keys()]
    
    def _unique_attributes(self, group_by):
        
        assert len(group_by) == sum([i in self._report_keys for i in group_by])
        
        datafeatunique =[]
        for i in range(len(self.report[group_by[0]])):
            attr = [str(self.report[attr][i]) for attr in group_by]
            if attr not in datafeatunique:
                datafeatunique.append(attr)
    
        return datafeatunique
    
    def _group_data_by_keys(self,group_by: Union[str, List[str]]) -> List[Dict]:
        
        if isinstance(group_by, str):
            group_by = [group_by]
        
        uniqueresults = self._unique_attributes(group_by)
        uniqueresults = concatenate_lists(uniqueresults)
        
        #uniqueresults = [concatenate_lists([getattr(self.report, featname)[j] if isinstance(getattr(self.report, featname)[j], list) 
        #                                 else [getattr(self.report, featname)[j]] for featname in group_by])
        #             for j in range(len(self.report[group_by[0]]))]
        data_groups = {}
        for k in uniqueresults:
            # get attributes
            indvals = []
            for j in range(len(self.report[group_by[0]])):
                compared = concatenate_lists(
                    [self.report[featname][j] if type(self.report[featname][j]) is list 
                     else [self.report[featname][j]] for featname in group_by])
                if '-'.join(compared) == k:
                    indvals.append({featname:self.report[featname][j] 
                                    for featname in list(self.report.keys())})
            
            data_groups[k] = indvals
            
        return data_groups
    
    def _check_previous_groups(self, key_name):
        if key_name in self._previous_groups.keys():
            return self._previous_groups[key_name]
        else:
            return None
            
    
    def summarise_by_groups(self,group_by: Union[str, List[str]], fnc = None):
        
        fnc = np.nanmean if fnc is None else fnc
        data_groups = self._group_data_by_keys(group_by)
        data_summarised = {}
        for k in data_groups.keys():
            value = self._check_previous_groups(k) 
            if value is None:
                # get only numerical data
                flatdict = [z[j] for z in data_groups[k] for j in z.keys() if isinstance(z[j],(float, int))]
                # reshape the list
                lenregisters = len(data_groups[k][0])
                reordered = np.array(flatdict).reshape(len(flatdict)//lenregisters,lenregisters)
                # summarise the data
                summarized = fnc(reordered, axis = 0)
                # from list to dict and save
                value = {j: summarized[i] 
                                        for i, j in enumerate(data_groups[k][0].keys()
                                                            ) if isinstance(data_groups[k][0][j],(float, int))}
            else:
                value
            data_summarised[k] = value
            
        self._previous_groups = data_summarised
        return data_summarised
    
    def update_report(self, new_entry):    
        
        """
        Update the reporter with a new entry.

        Parameters
        ----------
        new_entry : dict
            A dictionary containing the new entry to add. Keys in this dictionary should match 
            those in the _reporter_keys attribute.

        Raises
        ------
        ValueError
            If the keys in the new_entry do not match the _reporter_keys.

        Returns
        -------
        None
        """
        
        if not all(key in self._report_keys for key in new_entry):
            raise ValueError("Keys in the new entry do not match the reporter keys.")
        
        for k in list(self._report_keys):
            self.report[k].append(new_entry[k])   
        
    #@check_output_fn
    def save_reporter(self, path: str, fn:str, suffix = '.json'):
        json_object = json.dumps(self.report, indent=4)
        with open(os.path.join(path, fn), "w") as outfile:
            outfile.write(json_object)
            
    @check_path
    def load_reporter(self, path, verbose = True):
        reporter = loadjson(path)
        if reporter is None:
            self.set_reporter([''])
            if verbose: print('No data was found')
        else:
            if verbose: print('load')
        self._update_keys(reporter)
        self.report = reporter