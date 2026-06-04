"""Read compile-time info (entry / target / defines / flags / args) from an EXTERNAL DXC
shader PDB — the MSF/DIA .pdb files Unity writes to UnityProject\\ShaderPDB, named by shader
hash. Unity compiles without -Qembed_debug, so the capture's DXIL container has no embedded
compile args; they live only in the side-car PDB, reachable via the IDxcPdbUtils2 COM
interface exposed by dxcompiler.dll.

Windows-only. No COM registration and no pythoncom/comtypes needed: DxcCreateInstance is
dxc's own in-proc class factory, and we drive the vtables directly with ctypes. The PDB is
handed to IDxcPdbUtils2::Load through a tiny Python-implemented IDxcBlob (this build's
dxcompiler.dll does not expose IDxcUtils/IDxcLibrary to create one).
"""

import ctypes
from ctypes import (POINTER, c_void_p, c_long, c_ulong, c_size_t, c_uint32, c_uint16,
                    c_ubyte, c_wchar_p, byref, cast, addressof, Structure, WINFUNCTYPE)

# --------------------------------------------------------------------------------------
# GUID helpers
# --------------------------------------------------------------------------------------

class GUID(Structure):
    _fields_ = [("Data1", c_uint32), ("Data2", c_uint16),
                ("Data3", c_uint16), ("Data4", c_ubyte * 8)]

_ole = ctypes.WinDLL("ole32")

def _guid(s):
    g = GUID()
    if _ole.CLSIDFromString(c_wchar_p("{" + s + "}"), byref(g)) != 0:
        raise ValueError(f"bad GUID {s}")
    return g

def _guid_eq(riid_ptr, g):
    return ctypes.string_at(riid_ptr, 16) == ctypes.string_at(byref(g), 16)

CLSID_DxcPdbUtils = "54621dfb-f2ce-457e-ae8c-ec355faeec7c"
IID_IDxcPdbUtils2 = "4315D938-F369-4F93-95A2-252017CC3807"
IID_IUnknown      = "00000000-0000-0000-C000-000000000046"
IID_IDxcBlob      = "8BA5FB08-5195-40e2-AC58-0D989C3A0102"

# IDxcPdbUtils2 vtable slots (after IUnknown 0..2).
_LOAD = 3
_FLAG_COUNT, _FLAG = 9, 10
_ARG_COUNT,  _ARG  = 11, 12
_DEF_COUNT,  _DEF  = 15, 16
_TARGET, _ENTRY, _MAINFILE = 17, 18, 19
# IDxcBlob slots.
_BLOB_RELEASE, _BLOB_GETPTR = 2, 3

# --------------------------------------------------------------------------------------
# Minimal Python IDxcBlob over a bytes buffer (so IDxcPdbUtils2::Load has something to read)
# --------------------------------------------------------------------------------------

class _PyBlob:
    """A live COM IDxcBlob backed by `data`. Keep the instance alive for as long as the
    consumer (IDxcPdbUtils2) might hold it — i.e. until after Load + all reads."""
    def __init__(self, data):
        self._buf = (c_ubyte * len(data)).from_buffer_copy(data)
        self._size = len(data)
        self._iunk = _guid(IID_IUnknown)
        self._iblob = _guid(IID_IDxcBlob)
        QI = WINFUNCTYPE(c_long, c_void_p, c_void_p, POINTER(c_void_p))
        UL = WINFUNCTYPE(c_ulong, c_void_p)
        GP = WINFUNCTYPE(c_void_p, c_void_p)
        GS = WINFUNCTYPE(c_size_t, c_void_p)
        # keep the trampolines referenced so they aren't collected while COM holds them
        self._fns = [QI(self._qi), UL(self._addref), UL(self._release),
                     GP(self._getptr), GS(self._getsize)]
        self._vtbl = (c_void_p * 5)(*[cast(f, c_void_p) for f in self._fns])
        self._obj = c_void_p(addressof(self._vtbl))

    @property
    def ptr(self):
        return cast(byref(self._obj), c_void_p)

    def _qi(self, this, riid, ppv):
        if _guid_eq(riid, self._iunk) or _guid_eq(riid, self._iblob):
            ppv[0] = this
            return 0
        ppv[0] = None
        return 0x80004002  # E_NOINTERFACE

    def _addref(self, this):  return 1
    def _release(self, this): return 1
    def _getptr(self, this):  return cast(self._buf, c_void_p).value
    def _getsize(self, this): return self._size

# --------------------------------------------------------------------------------------
# IDxcPdbUtils2 access
# --------------------------------------------------------------------------------------

def _vmethod(p, idx, *argtypes, restype=c_long):
    """Bind vtable slot `idx` of COM interface `p`. restype defaults to HRESULT (c_long);
    pass restype=c_void_p for pointer-returning slots (e.g. GetBufferPointer) so the 64-bit
    return value isn't truncated."""
    vtbl = ctypes.cast(p, POINTER(POINTER(c_void_p)))
    return WINFUNCTYPE(restype, c_void_p, *argtypes)(vtbl.contents[idx])

def _blobwide_str(blob_ptr):
    """Read an IDxcBlobWide (UTF-16 string blob) via GetBufferPointer, then release it."""
    if not blob_ptr:
        return ""
    sp = _vmethod(blob_ptr, _BLOB_GETPTR, restype=c_void_p)(blob_ptr)
    s = ctypes.wstring_at(sp) if sp else ""
    _vmethod(blob_ptr, _BLOB_RELEASE, restype=c_ulong)(blob_ptr)
    return s

def _get_str(pPdb, idx):
    out = c_void_p()
    _vmethod(pPdb, idx, POINTER(c_void_p))(pPdb, byref(out))
    return _blobwide_str(out.value)

def _get_str_at(pPdb, idx, i):
    out = c_void_p()
    _vmethod(pPdb, idx, c_uint32, POINTER(c_void_p))(pPdb, i, byref(out))
    return _blobwide_str(out.value)

def _get_count(pPdb, idx):
    n = c_uint32(0)
    _vmethod(pPdb, idx, POINTER(c_uint32))(pPdb, byref(n))
    return n.value

def read_pdb_compile_info(pdb_path, dxcompiler_dll):
    """Return {entry, target, main_file, defines, flags, args} from a DXC PDB, or raise.
    `dxcompiler_dll` is the path to the dxcompiler.dll to drive (use the one beside the
    dxc.exe that built the shaders)."""
    with open(pdb_path, "rb") as fh:
        data = fh.read()
    dll = ctypes.WinDLL(dxcompiler_dll)
    create = dll.DxcCreateInstance
    create.restype = c_long
    create.argtypes = [POINTER(GUID), POINTER(GUID), POINTER(c_void_p)]

    pPdb = c_void_p()
    hr = create(byref(_guid(CLSID_DxcPdbUtils)), byref(_guid(IID_IDxcPdbUtils2)), byref(pPdb))
    if hr != 0 or not pPdb.value:
        raise OSError(f"DxcCreateInstance(IDxcPdbUtils2) failed: 0x{hr & 0xffffffff:08x}")
    blob = _PyBlob(data)  # must outlive Load + all reads
    try:
        hr = _vmethod(pPdb, _LOAD, c_void_p)(pPdb, blob.ptr)
        if hr != 0:
            raise OSError(f"IDxcPdbUtils2::Load failed: 0x{hr & 0xffffffff:08x}")
        info = {
            "entry":     _get_str(pPdb, _ENTRY),
            "target":    _get_str(pPdb, _TARGET),
            "main_file": _get_str(pPdb, _MAINFILE),
            "defines":   [_get_str_at(pPdb, _DEF,  i) for i in range(_get_count(pPdb, _DEF_COUNT))],
            "flags":     [_get_str_at(pPdb, _FLAG, i) for i in range(_get_count(pPdb, _FLAG_COUNT))],
            "args":      [_get_str_at(pPdb, _ARG,  i) for i in range(_get_count(pPdb, _ARG_COUNT))],
        }
    finally:
        _vmethod(pPdb, _BLOB_RELEASE)(pPdb)  # Release the pdb-utils instance
    return info
