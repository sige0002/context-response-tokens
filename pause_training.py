"""Temporarily isolate one verified GPU trainer, restoring its prior state."""
from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import signal
import time

from benchmark_streaming import idle_compute_processes
from run_video_comparisons import bind_predecessor


@contextmanager
def isolated_gpu(trainer_pid=None, trainer_output=None):
    fd, paused = None, False
    handlers = {sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    libc = ctypes.CDLL(None,use_errno=True)
    libc.pidfd_send_signal.argtypes,libc.pidfd_send_signal.restype = [ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint],ctypes.c_int
    def send(sig):
        if libc.pidfd_send_signal(fd,sig,None,0)<0:
            raise OSError(ctypes.get_errno(),'Could not signal the bound trainer')
    def stop(signum,frame):
        raise InterruptedError(f'GPU diagnostic interrupted: signal {signum}')
    for sig in handlers:
        signal.signal(sig,stop)
    try:
        if trainer_pid is not None:
            if trainer_output is None:
                raise ValueError('A verified trainer output is required when pausing')
            fd=bind_predecessor(trainer_pid,Path(trainer_output).resolve())
            state_file=Path(f'/proc/{trainer_pid}/stat')
            if state_file.read_text().rsplit(') ',1)[1].split()[0] not in ('T','t'):
                send(signal.SIGSTOP);paused=True
                for _ in range(100):
                    if state_file.read_text().rsplit(') ',1)[1].split()[0] in ('T','t'):
                        break
                    time.sleep(.05)
                else:
                    raise RuntimeError('Verified trainer did not pause')
        yield idle_compute_processes()
    finally:
        try:
            if fd is not None:
                if paused:
                    send(signal.SIGCONT)
                    print(f'Resumed verified trainer {trainer_pid}',flush=True)
        finally:
            if fd is not None:
                os.close(fd)
            for sig,handler in handlers.items():
                signal.signal(sig,handler)
