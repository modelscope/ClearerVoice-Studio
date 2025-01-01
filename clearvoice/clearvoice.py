from network_wrapper import network_wrapper
import os
import warnings
warnings.filterwarnings("ignore")

class ClearVoice:
    """ The main class inferface to the end users for performing speech processing
        this class provides the desired model to perform the given task
    """
    def __init__(self, task, model_names):
        """ Load the desired models for the specified task. Perform all the given models and return all results.
   
        Parameters:
        ----------
        task: str
            the task matching any of the provided tasks: 
            'speech_enhancement'
            'speech_separation'
            'target_speaker_extraction'
        model_names: str or list of str
            the model names matching any of the provided models: 
            'FRCRN_SE_16K'
            'MossFormer2_SE_48K'
            'MossFormerGAN_SE_16K'
            'MossFormer2_SS_16K'
            'AV_MossFormer2_TSE_16K'

        Returns:
        --------
        A ModelsList object, that can be run to get the desired results
        """        
        self.network_wrapper = network_wrapper()
        self.models = []
        for model_name in model_names:
            model = self.network_wrapper(task, model_name)
            self.models += [model]  
            
    def __call__(self, input_path, online_write=False, output_path=None, extract_noise=False):
        results = {}
        for model in self.models:
            result = model.process(input_path, online_write, output_path, extract_noise)
            if not online_write:
                if extract_noise:
                    results[model.name] = result  # result 现在是 (enhanced, noise) 元组
                else:
                    results[model.name] = result

        if not online_write:
            if len(results) == 1:
                return next(iter(results.values()))
            else:
                return results

    def write(self, results, output_path):
        add_subdir = False
        use_key = False
        if len(self.models) > 1: add_subdir = True #multi_model is True        
        for model in self.models:
            if isinstance(results, dict):
                if model.name in results: 
                   if len(results[model.name]) > 1: use_key = True
                       
                else:
                   if len(results) > 1: use_key = True #multi_input is True
            break

        for model in self.models:
            model.write(output_path, add_subdir, use_key)
