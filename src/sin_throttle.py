"""Desactiva el 'power throttling' (EcoQoS) de Windows 11 para el proceso actual: sin esto, un proceso lanzado
en segundo plano puede ser relegado a nucleos eficientes y sus tiempos se degradan ~2x entre repeticiones."""
import ctypes, sys

class _PPTS(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

def sin_throttle():
    if sys.platform != "win32": return
    st = _PPTS(1, 1, 0)  # version 1, controlar EXECUTION_SPEED, estado 0 = no limitar
    ctypes.windll.kernel32.SetProcessInformation(ctypes.windll.kernel32.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st))
