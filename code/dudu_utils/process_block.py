# Built-in
import os

# Libraries

# Custom
from dudu_utils import data_loader

# Settings



def make_dir_if_not_exist(path_to_dir):
    """
    Create a directory if it does not exist
    :param path_to_dir: path to the directory
    :return:
    """
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)


# TODO Thou shalt comment
class ProcessBlock(object):
    def __init__(self, process_func, process_dir, process_name=None):
        self.process_func = process_func
        self.process_dir = process_dir
        make_dir_if_not_exist(self.process_dir)
        if process_name is None:
            self.process_name = process_func.__name__
        else:
            self.process_name = process_name

    @staticmethod
    def check_complete(state_file):
        """
        check complete status of running process_func
        :param state_file: file where stores the complete status as True (Complete) or False (Incomplete)
        :return: Complete as True, Incomplete as False
        """
        state = data_loader.load_file(state_file)[0]
        if state == 'Complete':
            return True
        else:
            return False

    def run(self, force_run=False, verb=False, **kwargs):
        """
        run the process func, update state_file and save the result value
        :param force_run: force the process_func to run again no matter whether the complete status is incomplete or not
        :param kwargs: other parameters used by process_func
        :return: result value of process_func
        """
        state_file = os.path.join(self.process_dir, '{}_state.txt'.format(self.process_name))
        value_file = os.path.join(self.process_dir, '{}.pkl'.format(self.process_name))
        # write state file as incomplete if the state file does not exist
        if not os.path.exists(state_file):
            data_loader.save_file(state_file, 'Incomplete')

        if not self.check_complete(state_file) or not os.path.exists(value_file) or force_run:
            # run the process
            if verb:
                print('Start running process {}'.format(self.process_name))
            data_loader.save_file(state_file, 'Incomplete')

            val = self.process_func(**kwargs)
            data_loader.save_file(value_file, val)

            data_loader.save_file(state_file, 'Complete')
            if verb:
                print('Complete!')
        else:
            # load the data
            if verb:
                print('File already exits, load value')
            val = data_loader.load_file(value_file)

        return val


if __name__ == '__main__':
    pass
