# TOOLCHAIN EXTENSION
IF(WIN32)
    SET(TOOLCHAIN_EXT ".exe")
ELSE()
    SET(TOOLCHAIN_EXT "")
ENDIF()

# EXECUTABLE EXTENSION
SET (CMAKE_EXECUTABLE_SUFFIX ".elf")


# TOOLCHAIN_DIR AND NANO LIBRARY
SET(TOOLCHAIN_DIR "C:/nxp/MCUXpressoIDE_11.5.1_7266/ide/plugins/com.nxp.mcuxpresso.tools.win32_11.5.1.202201181444/tools")

MESSAGE(STATUS "TOOLCHAIN_DIR: " ${TOOLCHAIN_DIR})
# TARGET_TRIPLET

SET(TOOLCHAIN_BIN_DIR ${TOOLCHAIN_DIR}/bin)
SET(CMAKE_AR ${TOOLCHAIN_BIN_DIR}/arm-none-eabi-ar${TOOLCHAIN_EXT})

SET(CMAKE_SYSTEM_NAME Generic)
SET(CMAKE_SYSTEM_PROCESSOR arm)
UNSET(TARGET_TRIPLET)
SET(TARGET_TRIPLET "arm-none-eabi-gcc")
SET(CMAKE_C_COMPILER ${TOOLCHAIN_BIN_DIR}/${TARGET_TRIPLET}${TOOLCHAIN_EXT})
SET(CMAKE_CXX_COMPILER ${TOOLCHAIN_BIN_DIR}/arm-none-eabi-g++${TOOLCHAIN_EXT})
SET(CMAKE_ASM_COMPILER ${TOOLCHAIN_BIN_DIR}/${TARGET_TRIPLET}${TOOLCHAIN_EXT})
SET(CMAKE_CP_COMPILER ${TOOLCHAIN_BIN_DIR}/arm-none-eabi-cpp${TOOLCHAIN_EXT})
SET(CMAKE_OBJCOPY ${TOOLCHAIN_BIN_DIR}/arm-none-eabi-objcopy${TOOLCHAIN_EXT} CACHE STRING "objcopy")
SET(CMAKE_SIZE ${TOOLCHAIN_BIN_DIR}/arm-none-eabi-size${TOOLCHAIN_EXT} CACHE STRING "size")

set(CMAKE_AR_FORCED TRUE)
set(CMAKE_C_COMPILER_FORCED TRUE)
set(CMAKE_CXX_COMPILER_FORCED TRUE)
SET(CMAKE_C_FLAGS " -O2 -Os --specs=nosys.specs -g -mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -DARM_MATH_CM7 -MP -MMD -DC_HAS_MFPU_NEON=0 -D__FPU_PRESENT -DCMAKE_SYSTEM_PROCESSOR=" "arm" CACHE STRING "c compiler flags")
SET(CMAKE_CXX_FLAGS " -O2 -Os --specs=nosys.specs -fshort-enums -Wno-narrowing -fno-pie -fno-pic -fpermissive -mabi=aapcs -mcpu=cortex-m7 -mfp16-format=ieee -mfpu=fpv5-d16 -mthumb -mfloat-abi=hard -g --std=c++11 -DCMAKE_CROSSCOMPILING -MP -MMD -DC_HAS_MFPU_NEON=0 -DCMAKE_SYSTEM_PROCESSOR=" "arm" CACHE STRING "cxx compiler flags")
SET(CMAKE_C_FLAGS_DEBUG "-O2 -Os -g" CACHE INTERNAL "c compiler flags debug")
SET(CMAKE_CXX_FLAGS_DEBUG "-O2 -Os -g" CACHE INTERNAL "cxx compiler flags debug")

SET(CMAKE_C_FLAGS_RELEASE "-O3 " CACHE INTERNAL "c compiler flags release")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 " CACHE INTERNAL "cxx compiler flags release")
UNSET(TARGET_TRIPLET)
SET(CMAKE_FIND_ROOT_PATH ${TOOLCHAIN_DIR} ${EXTRA_FIND_PATH})
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_SIZEOF_VOID_P 4)