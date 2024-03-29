

# Boolean options do not have an argument; they are either specified on a command line or not
# Format: (option) : (new_option)
BOOLEAN_OPTIONS = {
  '--verbose': '',
  '-v': '',
  '--device-c': '-fcuda-rdc -c', 
  '-dc': '-fcuda-rdc -c',
  '--cuda': '-c',
  '-cuda': '-c',
  '--cubin': '-c',
  '-cubin': '-c',
  '--fatbin': '-c',
  '-fatbin': '-c',
  '--ptx': '-c',
  '-ptx': '-c',
  '--preprocess': '',
  '-E': '',
  '--generate-dependencies': '',
  '-M': '',
  '--compile': '-c',
  '-c': '-c',
  '--device-w': '',
  '-dw': '',
  '--device-link': '',
  '-dlink': '',
  '--link': '',
  '-link': '',
  '--lib': '',
  '-lib': '',
  '--run': '',
  '-run': '',
  '--profile': '',
  '-pg': '',
  '--debug': '-g',
  '-g': '-g',
  '--device-debug': '',
  '-G': '',
  '--generate-line-info': '',
  '-lineinfo': '',
  '--shared': '',
  '-shared': '',
  '--no-host-device-initializer-list': '',
  '-nohdinitlist': '',
  '--no-host-device-move-forward': '',
  '-nohdmoveforward': '',
  '--expt-relaxed-constexpr': '',
  '--expt-extended-lambda': '',
  '--extended-lambda': '',
  '-extended-lambda': '',
  '--dont-use-profile': '',
  '-noprof': '',
  '--dryrun': '',
  '-dryrun': '',
  '--keep': '',
  '-keep': '',
  '--save-temps': '',
  '-save-temps': '',
  '--clean-targets': '',
  '-clean': '',
  '--no-align-double': '',
  '--no-device-link': '',
  '--use_fast_math': '',
  '-use_fast_math': '',
  '--disable-warnings': '',
  '-w': '',
  '--keep-device-functions': '',
  '-keep-device-functions': '',
  '--source-in-ptx': '',
  '-source-in-ptx': '',
  '--restrict': '',
  '-restrict': '',
  '--Wreorder': '',
  '-Wreorder': '',
  '--Wno-deprecated-declarations': '',
  '-Wno-deprecated-declarations': '',
  '--Wno-deprecated-gpu-targets': '',
  '-Wno-deprecated-gpu-targets': '',
  '--resource-usage': '',
  '--res-usage': '',
  '--help': '',
  '-h': '',
  '--version': '--version',
  '-V': '--version'
}

# Single value options must be specified at most once
# Format:  (option) : (new_option, new_value, 'assigment_uses_equal_clang?')
# If the value of an option is SAME, we pass the same value from nvcc to clang
# If the value is CHECK, we use the SINGLE_VALUE_OPTIONS_TRANS table to translate it
SINGLE_VALUE_OPTIONS = {
  '--output-file': ('-o', 'SAME', 0),
  '-o': ('-o', 'SAME', 0),
  '--compiler-bindir': ('', '', 0),
  '-ccbin': ('', '', 0),
  '--output-directory': ('', '', 0),
  '-odir': ('', '', 0),
  '--cudart': ('', '', 0),
  '-cudart': ('', '', 0),
  '--libdevice-directory': ('', '', 0),
  '-ldir': ('', '', 0),
  '--use-local-env': ('', '', 0),
  '--optimize': ('-O', '', 0),
  '-O': ('-O', '', 0),
  '--ftemplate-backtrace-limit': ('', '', 0),
  '-ftemplate-backtrace-limit': ('', '', 0),
  '--ftemplate-depth': ('', '', 0),
  '--x': ('CHECK', 'CHECK', 0), ######################! cu should be cuda
  '-x': ('CHECK', 'CHECK', 0),
  '--std': ('-std', 'SAME', 1), ######################## should be -std=XXX
  '-std': ('-std', 'SAME', 1),
  '--machine': ('', '', 0),
  '-m': ('', '', 0),
  '--keep-dir': ('', '', 0),
  '-keep-dir': ('', '', 0),
  '--input-drive-prefix': ('', '', 0),
  '-idp': ('', '', 0),
  '--dependency-drive-prefix': ('', '', 0),
  '-ddp': ('', '', 0),
  '--drive-prefix': ('', '', 0),
  '-dp': ('', '', 0),
  '--dependency-target-name': ('', '', 0),
  '-MT': ('', '', 0),
  '--gpu-architecture': ('--cuda-gpu-arch', 'SAME', 1),
  '-arch': ('--cuda-gpu-arch', 'SAME', 1),
  '--generate-code': ('', '', 0),
  '-gencode': ('', '', 0),
  '--relocatable-device-code': ('CHECK', 'CHECK', 0), ################ if this false it should be -fno-cuda-rdc
  '-rdc': ('CHECK', 'CHECK', 0), ################ if this false it should be -fno-cuda-rdc
  '--maxrregcount': ('', '', 0),
  '-maxrregcount': ('', '', 0),
  '--ftz': ('', '', 0),
  '-ftz': ('', '', 0),
  '--prec-div': ('', '', 0),
  '-prec-div': ('', '', 0),
  '--prec-sqrt': ('', '', 0),
  '-prec-sqrt': ('', '', 0),
  '--fmad': ('', '', 0),
  '-fmad': ('', '', 0),
  '--default-stream': ('', '', 0),
  '-default-stream': ('', '', 0)
}

# Tables for the CHECK case in the SINGLE_VALUE_OPTIONS table
SINGLE_VALUE_OPTIONS_TRANS = {
  '--x cu': '-x cuda',
  '--x c': '-x c',
  '--x c++': '-x c++',
  '-x cu': '-x cuda',
  '-x c': '-x c',
  '-x c++': '-x c++',
  '--relocatable-device-code true': '-fcuda-rdc',
  '--relocatable-device-code false': '-fno-cuda-rdc',
  '-rdc true': '-fcuda-rdc',
  '-rdc false': '-fno-cuda-rdc',
}

# List options may be repeated.
# Format: (option) : (new_option, new_value)
LIST_OPTIONS = {
  '--include-path': ('-I', 'SAME', 0),
  '-I': ('-I', 'SAME', 0), 
  '--ptxas-options': ('','', 0), 
  '-Xptxas':('','', 0),
  '--pre-include': ('-include', 'SAME', 0),
  '-include': ('-include', 'SAME', 0),
  '--library': ('', '', 0),
  '-l': ('', '', 0),
  '--define-macro': ('-D', 'SAME', 0),
  '-D': ('-D', 'SAME', 0),
  '--undefine-macro': ('-U', 'SAME', 0),
  '-U': ('-U', 'SAME', 0),
  '--system-include': ('-isystem', 'SAME', 0),
  '-isystem': ('-isystem', 'SAME', 0),
  '--library-path': ('-L', 'SAME', 0),
  '-L': ('-L', 'SAME', 0),
  '--compiler-options': ('', '', 0),
  '-Xcompiler': ('', '', 0),
  '--linker-options': ('', '', 0),
  '-Xlinker': ('', '', 0),
  '--archive-options': ('', '', 0),
  '-Xarchive': ('', '', 0),
  '--nvlink-options': ('', '', 0),
  '-Xnvlink': ('', '', 0),
  '--run-args': ('', '', 0),
  '-run-args': ('', '', 0),
  '--gpu-code': ('--cuda-gpu-arch', 'SAME', 1),
  '-code': ('--cuda-gpu-arch', 'SAME', 1),
  '--entries': ('', '', 0),
  '-e': ('', '', 0),
  '--Werror': ('-Werror', '', 0),
  '-Werror': ('-Werror', '', 0),
  '--options-file': ('', '', 0),
  '-optf': ('', '', 0),
  '-Xcudafe': ('', '', 0)
}


