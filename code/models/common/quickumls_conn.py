import os
import time
import subprocess


class QuickUMLS(object):
    """initialize QuickUMLS server"""

    def __init__(self, window=1, threshold=0.7, semtypes=None,
                 cwd=os.path.dirname(os.path.realpath('QuickUMLS')) + '/QuickUMLS', host='localhost', port='4645'):
        self.window = window
        self.threshold = threshold
        self.semtypes = semtypes
        self.cwd = cwd
        self.host = host
        self.port = port

    def launch_quickumls(self):
        """launch quickumls server process"""
        cmd = "python3 -m quickumls.server data_files/ --threshold " + str(self.threshold) + " --window " + str(self.window)
        if self.semtypes is not None:
            cmd = cmd + " --accepted_semtypes " + self.semtypes
        params = cmd.split() + ["--host", self.host, "--port", self.port]
        self.process = subprocess.Popen(params, cwd=self.cwd)
        time.sleep(8)  # required minimum time to allow server initialization (in seconds)
        return self

    def close_quickumls(self):
        """close quickumls server process"""
        self.process.terminate()
        return self
