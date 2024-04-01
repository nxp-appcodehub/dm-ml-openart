/*
 * This file is part of the MicroPython project, http://micropython.org/
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2017 Damien P. George
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include "py/mphal.h"
#include "py/mpthread.h"
#include "py/runtime.h"
#include "py/stream.h"
#include "extmod/vfs.h"
#include "extmod/vfs_posix.h"
#include "extmod/vfs_fat.h"
#include "extmod/vfs_lfs.h"
#include "extmod/misc.h"
#if MICROPY_VFS

// These are defined in modos.c
MP_DECLARE_CONST_FUN_OBJ_VAR_BETWEEN(mod_os_errno_obj);
MP_DECLARE_CONST_FUN_OBJ_1(mod_os_getenv_obj);
MP_DECLARE_CONST_FUN_OBJ_1(mod_os_putenv_obj);
MP_DECLARE_CONST_FUN_OBJ_1(mod_os_unsetenv_obj);
MP_DECLARE_CONST_FUN_OBJ_1(mod_os_system_obj);
MP_DECLARE_CONST_FUN_OBJ_1(mod_os_stat_obj);
MP_DECLARE_CONST_FUN_OBJ_1(mod_os_remove_obj);
MP_DECLARE_CONST_FUN_OBJ_1(mod_os_rmdir_obj);
MP_DECLARE_CONST_FUN_OBJ_1(mod_os_mkdir_obj);
MP_DECLARE_CONST_FUN_OBJ_1(mod_os_stat_obj);
MP_DECLARE_CONST_FUN_OBJ_1(mod_os_ilistdir_obj);
MP_DECLARE_CONST_FUN_OBJ_1(mod_os_rename_obj);

STATIC const mp_rom_map_elem_t uos_vfs_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_uos_vfs) },
    { MP_ROM_QSTR(MP_QSTR_sep), MP_ROM_QSTR(MP_QSTR__slash_) },

    { MP_ROM_QSTR(MP_QSTR_errno), MP_ROM_PTR(&mod_os_errno_obj) },
    { MP_ROM_QSTR(MP_QSTR_getenv), MP_ROM_PTR(&mod_os_getenv_obj) },
    { MP_ROM_QSTR(MP_QSTR_putenv), MP_ROM_PTR(&mod_os_putenv_obj) },
    { MP_ROM_QSTR(MP_QSTR_unsetenv), MP_ROM_PTR(&mod_os_unsetenv_obj) },
    { MP_ROM_QSTR(MP_QSTR_system), MP_ROM_PTR(&mod_os_system_obj) },
	{ MP_ROM_QSTR(MP_QSTR_stat), MP_ROM_PTR(&mod_os_stat_obj) },
	{ MP_ROM_QSTR(MP_QSTR_ilistdir), MP_ROM_PTR(&mod_os_ilistdir_obj) },
	{ MP_ROM_QSTR(MP_QSTR_mkdir), MP_ROM_PTR(&mod_os_mkdir_obj) },
	{ MP_ROM_QSTR(MP_QSTR_remove), MP_ROM_PTR(&mod_os_remove_obj) },
	{ MP_ROM_QSTR(MP_QSTR_rename),MP_ROM_PTR(&mod_os_rename_obj) },
    { MP_ROM_QSTR(MP_QSTR_rmdir), MP_ROM_PTR(&mod_os_rmdir_obj) },
	
	
    { MP_ROM_QSTR(MP_QSTR_mount), MP_ROM_PTR(&mp_vfs_mount_obj) },
    { MP_ROM_QSTR(MP_QSTR_umount), MP_ROM_PTR(&mp_vfs_umount_obj) },

    { MP_ROM_QSTR(MP_QSTR_chdir), MP_ROM_PTR(&mp_vfs_chdir_obj) },
    { MP_ROM_QSTR(MP_QSTR_getcwd), MP_ROM_PTR(&mp_vfs_getcwd_obj) },
    
    { MP_ROM_QSTR(MP_QSTR_listdir), MP_ROM_PTR(&mp_vfs_listdir_obj) },
    
    
    
    
    { MP_ROM_QSTR(MP_QSTR_statvfs), MP_ROM_PTR(&mp_vfs_statvfs_obj) },
    { MP_ROM_QSTR(MP_QSTR_unlink), MP_ROM_PTR(&mp_vfs_remove_obj) }, // unlink aliases to remove

    #if MICROPY_PY_OS_DUPTERM
    { MP_ROM_QSTR(MP_QSTR_dupterm), MP_ROM_PTR(&mp_uos_dupterm_obj) },
    #endif

    #if MICROPY_VFS_POSIX
    { MP_ROM_QSTR(MP_QSTR_VfsPosix), MP_ROM_PTR(&mp_type_vfs_posix) },
    #endif
    #if MICROPY_VFS_FAT
    { MP_ROM_QSTR(MP_QSTR_VfsFat), MP_ROM_PTR(&mp_fat_vfs_type) },
    #endif
    #if MICROPY_VFS_LFS1
    { MP_ROM_QSTR(MP_QSTR_VfsLfs1), MP_ROM_PTR(&mp_type_vfs_lfs1) },
    #endif
    #if MICROPY_VFS_LFS2
    { MP_ROM_QSTR(MP_QSTR_VfsLfs2), MP_ROM_PTR(&mp_type_vfs_lfs2) },
    #endif
};

STATIC MP_DEFINE_CONST_DICT(uos_vfs_module_globals, uos_vfs_module_globals_table);

const mp_obj_module_t mp_module_uos_vfs = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&uos_vfs_module_globals,
};


mp_import_stat_t mp_posix_import_stat(const char *path) {

    struct stat state;
    int st;

	st = stat(path, &state);
    if (st == 0) {
        if (S_ISDIR(state.st_mode)) {
            return MP_IMPORT_STAT_DIR;
        } else {
            return MP_IMPORT_STAT_FILE;
        }
    } else {
        return MP_IMPORT_STAT_NO_EXIST;
    }
}

#include "extmod/vfs_posix.h"
mp_obj_t mp_builtin_open(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kwargs) {
    enum { ARG_file, ARG_mode };
    STATIC const mp_arg_t allowed_args[] = {
        { MP_QSTR_file, MP_ARG_OBJ | MP_ARG_REQUIRED, {.u_rom_obj = MP_ROM_NONE} },
        { MP_QSTR_mode, MP_ARG_OBJ, {.u_obj = MP_OBJ_NEW_QSTR(MP_QSTR_r)} },
        { MP_QSTR_buffering, MP_ARG_INT, {.u_int = -1} },
        { MP_QSTR_encoding, MP_ARG_OBJ, {.u_rom_obj = MP_ROM_NONE} },
    };
    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args, pos_args, kwargs, MP_ARRAY_SIZE(allowed_args), allowed_args, args);
    return mp_vfs_posix_file_open(&mp_type_textio, args[ARG_file].u_obj, args[ARG_mode].u_obj);
}
MP_DEFINE_CONST_FUN_OBJ_KW(mp_builtin_open_obj, 1, mp_builtin_open);

void mp_reader_new_file(mp_reader_t *reader, const char *filename) {
    MP_THREAD_GIL_EXIT();
    int fd = open(filename, O_RDONLY, 0644);
    MP_THREAD_GIL_ENTER();
    if (fd < 0) {
        mp_raise_OSError(errno);
    }
    mp_reader_new_file_from_fd(reader, fd, true);
}

#endif // MICROPY_VFS
