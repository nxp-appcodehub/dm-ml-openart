'''
/*
 * Copyright (c) 2006-2018, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */
'''
import os

# toolchains options
ARCH='arm'
CPU='cortex-m7'
CROSS_TOOL='gcc'

# bsp lib config
BSP_LIBRARY_TYPE = None

if os.getenv('RTT_CC'):
    CROSS_TOOL = os.getenv('RTT_CC')
if os.getenv('RTT_ROOT'):
    RTT_ROOT = os.getenv('RTT_ROOT')

# cross_tool provides the cross compiler
# EXEC_PATH is the compiler execute path, for example, CodeSourcery, Keil MDK, IAR
if  CROSS_TOOL == 'gcc':
    PLATFORM    = 'gcc'
    EXEC_PATH   = r'C:\Users\XXYYZZ'
elif CROSS_TOOL == 'keil':
    PLATFORM    = 'armcc'
    EXEC_PATH   = r'C:/Keil_v5'
    #EXEC_PATH   = r'D:/Keil_v5'
elif CROSS_TOOL == 'iar':
    PLATFORM    = 'iccarm'
    EXEC_PATH   = r'C:/Program Files (x86)/IAR Systems/Embedded Workbench 8.3'

if os.getenv('RTT_EXEC_PATH'):
    EXEC_PATH = os.getenv('RTT_EXEC_PATH')

BUILD = 'debug' 
#BUILD = 'release'
CPP_COMMAND = ''

if PLATFORM == 'gcc':
    PREFIX = 'arm-none-eabi-'
    CC = PREFIX + 'gcc'
    CXX = PREFIX + 'g++'
    AS = PREFIX + 'gcc'
    AR = PREFIX + 'ar'
    LINK = PREFIX + 'gcc'
    TARGET_EXT = 'elf'
    SIZE = PREFIX + 'size'
    OBJDUMP = PREFIX + 'objdump'
    OBJCPY = PREFIX + 'objcopy'
    STRIP = PREFIX + 'strip'

    DEVICE = ' -mcpu=' + CPU + ' -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard -ffunction-sections -fdata-sections'
    CFLAGS = DEVICE + ' -Wall -D__FPU_PRESENT -eentry'
    AFLAGS = ' -c' + DEVICE + ' -x assembler-with-cpp -Wa,-mimplicit-it=thumb -D__START=entry'
    LFLAGS = DEVICE + ' -lm -lgcc -lc' + ' -nostartfiles -Wl,--gc-sections,-Map=rtthread.map,-cref,-u,Reset_Handler -T board/linker_scripts/link_final.ld -Xlinker -print-memory-usage -lstdc++'
    CPP_COMMAND = "${CMAKE_CP_COMPILER} -include ${CMAKE_SOURCE_DIR}/rtconfig.h ${CMAKE_SOURCE_DIR}/board/linker_scripts/link.ld ${CMAKE_SOURCE_DIR}/board/linker_scripts/link_final.ld"
    CPATH = ''
    LPATH = ''

    AFLAGS += ' -D__STARTUP_INITIALIZE_NONCACHEDATA'
    AFLAGS += ' -D__STARTUP_CLEAR_BSS'

    if BUILD == 'debug':
        CFLAGS += ' -g'
        AFLAGS += ' -g'
        CFLAGS += ' -O0'
    else:
        CFLAGS += '  -fno-common -O3'

    POST_ACTION = OBJCPY + ' -O binary $TARGET rtthread.bin\n' + SIZE + ' $TARGET \n'
    # POST_ACTION = OBJCPY + ' -O ihex $TARGET rtthread-gcc.hex\n' + SIZE + ' $TARGET \n'

    # module setting 
    CXXFLAGS = ' -Woverloaded-virtual -fno-exceptions -fno-rtti '
    M_CFLAGS = CFLAGS + ' -mlong-calls -fPIC '
    M_CXXFLAGS = CXXFLAGS + ' -mlong-calls -fPIC'
    M_LFLAGS = DEVICE + CXXFLAGS + ' -Wl,--gc-sections,-z,max-page-size=0x4' +\
                                    ' -shared -fPIC -nostartfiles -static-libgcc'
    M_POST_ACTION = STRIP + ' -R .hash $TARGET\n' + SIZE + ' $TARGET \n'

elif PLATFORM == 'armcc':
    CC = 'armcc'
    CXX = 'armcc'
    AS = 'armasm'
    AR = 'armar'
    LINK = 'armlink'
    TARGET_EXT = 'axf'

    DEVICE = ' --cpu ' + CPU + '.fp.sp'
    CFLAGS = DEVICE + ' --apcs=interwork'
    AFLAGS = DEVICE
    LFLAGS = DEVICE + ' --libpath "' + EXEC_PATH + '/ARM/ARMCC/lib" --info sizes --info totals --info unused --info veneers --list rtthread.map --scatter "board\linker_scripts\link.sct"'

    CFLAGS += ' --diag_suppress=66,1296,186'
    CFLAGS += ' -I' + EXEC_PATH + '/ARM/RV31/INC'
    LFLAGS += ' --libpath ' + EXEC_PATH + '/ARM/RV31/LIB'

    EXEC_PATH += '/arm/bin40/'

    if BUILD == 'debug':
        CFLAGS += ' -g -O0'
        AFLAGS += ' -g'
    else:
        CFLAGS += ' -O3'

    CXXFLAGS = CFLAGS
    CFLAGS += ' --c99'

    # POST_ACTION = 'fromelf -z $TARGET'
    POST_ACTION = 'fromelf --bin $TARGET --output rtthread.bin \nfromelf -z $TARGET'    
    # POST_ACTION = 'fromelf --i32combined $TARGET --output="rtthread-mdk.hex" \nfromelf -z $TARGET'

elif PLATFORM == 'armclang':
    CC = 'armclang'
    CXX = 'armclang'
    AS = 'armclang'
    AR = 'armar'
    LINK = 'armlink'
    TARGET_EXT = 'axf'
    print("armclang +++++")
    DEVICE = ' --cpu=Cortex-M7.fp.dp '
    CFLAGS = ' -xc -std=c99 '
    CFLAGS += ' --target=arm-arm-none-eabi -mcpu=cortex-m7 -mfpu=fpv5-d16 '
    CFLAGS += ' -mfloat-abi=hard -c -gdwarf-4 '
    CFLAGS += ' -fno-rtti -funsigned-char -fshort-enums -fshort-wchar -ffunction-sections '
    CFLAGS += ' -Wa,armasm,--pd,"CPU_MIMXRT1062DVL6A SETA 1"'
    CFLAGS += ' -Wa,armasm,--pd,"__STARTUP_INITIALIZE_NONCACHEDATA SETA 1"'
    AFLAGS = ' -masm=auto '
    AFLAGS += ' --target=arm-arm-none-eabi -mcpu=cortex-m7 -mfpu=fpv5-d16 '
    AFLAGS += ' -mfloat-abi=hard -c -gdwarf-4 '
    LFLAGS = DEVICE + ' --info sizes --info totals --info unused --info veneers --list rt-thread.map '
    LFLAGS += r' --strict --scatter "board\linker_scripts\link.sct" '
    CFLAGS += ' -I' + EXEC_PATH + '/ARM/ARMCLANG/include'
    LFLAGS += ' --libpath=' + EXEC_PATH + '/ARM/ARMCLANG/lib '

    EXEC_PATH += '/ARM/ARMCLANG/bin/'

    if BUILD == 'debug':
        CFLAGS += ' -g -O1' # armclang recommend
        AFLAGS += ' -g'
    else:
        CFLAGS += ' -O3'

    CXXFLAGS = CFLAGS

    POST_ACTION = 'fromelf --bin $TARGET --output rtthread.bin \nfromelf -z $TARGET'

elif PLATFORM == 'iccarm':
    CC = 'iccarm'
    CXX = 'iccarm'
    AS = 'iasmarm'
    AR = 'iarchive'
    LINK = 'ilinkarm'
    TARGET_EXT = 'out'

    DEVICE = ' -D__FPU_PRESENT'

    CFLAGS = DEVICE
    CFLAGS += ' --diag_suppress Pa050'
    CFLAGS += ' --no_cse'
    CFLAGS += ' --no_unroll'
    CFLAGS += ' --no_inline'
    CFLAGS += ' --no_code_motion'
    CFLAGS += ' --no_tbaa'
    CFLAGS += ' --no_clustering'
    CFLAGS += ' --no_scheduling'
    CFLAGS += ' --debug'
    CFLAGS += ' --endian=little'
    CFLAGS += ' --cpu=' + CPU
    CFLAGS += ' -e'
    CFLAGS += ' --fpu=None'
    CFLAGS += ' --dlib_config "' + EXEC_PATH + '/arm/INC/c/DLib_Config_Normal.h"'
    CFLAGS += ' -Ol'
    CFLAGS += ' --use_c++_inline'

    AFLAGS = ''
    AFLAGS += ' -s+'
    AFLAGS += ' -w+'
    AFLAGS += ' -r'
    AFLAGS += ' --cpu ' + CPU
    AFLAGS += ' --fpu None'

    if BUILD == 'debug':
        CFLAGS += ' --debug'
        CFLAGS += ' -On'
    else:
        CFLAGS += ' -Oh'

    LFLAGS = ' --config "board/linker_scripts/link.icf"'
    LFLAGS += ' --redirect _Printf=_PrintfTiny'
    LFLAGS += ' --redirect _Scanf=_ScanfSmall'
    LFLAGS += ' --entry __iar_program_start'

    CXXFLAGS = CFLAGS

    EXEC_PATH = EXEC_PATH + '/arm/bin/'
    POST_ACTION = 'ielftool --bin $TARGET rtthread.bin'

def dist_handle(BSP_ROOT, dist_dir):
    import sys
    cwd_path = os.getcwd()
    sys.path.append(os.path.join(os.path.dirname(BSP_ROOT), 'tools'))
    from sdk_dist import dist_do_building
    dist_do_building(BSP_ROOT, dist_dir)
