/*
 * This file is part of the OpenMV project.
 * Copyright (c) 2013-2016 Kwabena W. Agyeman <kwagyeman@openmv.io>
 * This work is licensed under the MIT license, see the file LICENSE for details.
 *
 * File System Helper Functions
 *
 */
#include "imlib_config.h"
#if defined(IMLIB_ENABLE_IMAGE_FILE_IO)

#include <string.h>
#include <fcntl.h>
#include "ff.h"
#include "py/runtime.h"
#include "extmod/vfs_posix.h"
#include <sys/stat.h>
#include "common.h"
#include "fb_alloc.h"
#include "ff_wrapper.h"
#define FF_MIN(x,y) (((x)<(y))?(x):(y))

#ifndef SEEK_SET
#define	SEEK_SET	0	/* set file offset to offset */
#endif
#ifndef SEEK_CUR
#define	SEEK_CUR	1	/* set file offset to current plus offset */
#endif
#ifndef SEEK_END
#define	SEEK_END	2	/* set file offset to EOF plus offset */
#endif
const char *ffs_strerror(FRESULT res)
{
    static const char *ffs_errors[]={
        "Succeeded",
        "A hard error occurred in the low level disk I/O layer",
        "Assertion failed",
        "The physical drive cannot work",
        "Could not find the file",
        "Could not find the path",
        "The path name format is invalid",
        "Access denied due to prohibited access or directory full",
        "Access denied due to prohibited access",
        "The file/directory object is invalid",
        "The physical drive is write protected",
        "The logical drive number is invalid",
        "The volume has no work area",
        "There is no valid FAT volume",
        "The f_mkfs() aborted due to any parameter error",
        "Could not get a grant to access the volume within defined period",
        "The operation is rejected according to the file sharing policy",
        "LFN working buffer could not be allocated",
        "Number of open files > _FS_SHARE",
        "Given parameter is invalid",
    };

    if (res>sizeof(ffs_errors)/sizeof(ffs_errors[0])) {
        return "unknown error";
    } else {
        return ffs_errors[res];
    }
}

static void ff_fail(int fp, FRESULT res, const char *msg)
{
    if (fp >= 0) close(fp);
    mp_raise_msg_varg(&mp_type_OSError, "%s %s",msg, ffs_strerror(res));
    //mp_raise_msg(&mp_type_OSError, (mp_rom_error_text_t) ffs_strerror(res));
}

static void ff_read_fail(int fp)
{
    if (fp >= 0) close(fp);
    mp_raise_msg(&mp_type_OSError, MP_ERROR_TEXT("Failed to read requested bytes!"));
}

static void ff_write_fail(int fp)
{
    if (fp >= 0) close(fp);
    mp_raise_msg(&mp_type_OSError, MP_ERROR_TEXT("Failed to write requested bytes!"));
}

static void ff_expect_fail(int fp)
{
    if (fp >= 0) close(fp);
    mp_raise_msg(&mp_type_OSError, MP_ERROR_TEXT("Unexpected value read!"));
}

void ff_unsupported_format(FIL *fp)
{
    if (fp->obj.id >= 0) close(fp->obj.id);
    mp_raise_msg(&mp_type_OSError, MP_ERROR_TEXT("Unsupported format!"));
}

void ff_file_corrupted(FIL *fp)
{
    if (fp->obj.id >= 0) close(fp->obj.id);
    mp_raise_msg(&mp_type_OSError, MP_ERROR_TEXT("File corrupted!"));
}

void ff_not_equal(FIL *fp)
{
    if (fp->obj.id >= 0) close(fp->obj.id);
    mp_raise_msg(&mp_type_OSError, MP_ERROR_TEXT("Images not equal!"));
}

void ff_no_intersection(FIL *fp)
{
    if (fp->obj.id >= 0) close(fp->obj.id);
    mp_raise_msg(&mp_type_OSError, MP_ERROR_TEXT("No intersection!"));
}

void file_read_open(FIL *fp, const char *path)
{
    int res = f_open_helper(fp, path, FA_READ|FA_OPEN_EXISTING);
    if (res != FR_OK) ff_fail(fp->obj.id, res,path);
}

void file_write_open(FIL *fp, const char *path)
{
    int res = f_open_helper(fp, path, FA_WRITE|FA_CREATE_ALWAYS);
    if (res != FR_OK) ff_fail(fp->obj.id, res,path);
}

void file_read_write_open_existing(FIL *fp, const char *path)
{
    int res = f_open_helper(fp, path, FA_READ|FA_WRITE|FA_OPEN_EXISTING);
    if (res != FR_OK) ff_fail(fp->obj.id, res,path);
}

void file_read_write_open_always(FIL *fp, const char *path)
{
    int res = f_open_helper(fp, path, FA_READ|FA_WRITE|FA_OPEN_ALWAYS);
    if (res != FR_OK) ff_fail(fp->obj.id, res,path);
}

void file_close(FIL *fp)
{
    int res = close(fp->obj.id);
    if (res < 0) ff_fail(fp->obj.id, res," ");
}

void file_seek(FIL *fp, UINT offset)
{
    int res = lseek(fp->obj.id, offset,SEEK_SET);
    if (res < 0) ff_fail(fp->obj.id, res," ");
	fp->fptr = res;
}

void file_truncate(FIL *fp)
{
    //int res = f_truncate(fp);
    //if (res != FR_OK) ff_fail(fp, res);
}

void file_sync(FIL *fp)
{
    int res = fsync(fp->obj.id);
    if (res < 0) ff_fail(fp->obj.id, res," ");
}

// These wrapper functions are used for backward compatibility with
// OpenMV code using vanilla FatFS. Note: Extracted from cc3200 ftp.c

int lookup_fullpath(const TCHAR *fname, vstr_t *fpath)
{
	size_t path_num;
	mp_obj_t *path_items;
	mp_obj_list_get(mp_sys_path, &path_num, &path_items);
	for (size_t i = 0; i < path_num; i++) {
		vstr_reset(fpath);
		memset(fpath->buf,0x00,fpath->alloc);
		size_t p_len;
		const char *p = mp_obj_str_get_data(path_items[i], &p_len);
		if (p_len > 0) {
				vstr_add_strn(fpath, p, p_len);
		}
        vstr_add_strn(fpath, "/", strlen("/"));
		vstr_add_strn(fpath, fname, strlen(fname));
		fpath->buf[fpath->len+1] = 0x00;
		mp_import_stat_t stat = mp_import_stat(fpath->buf);
		if (stat == MP_IMPORT_STAT_FILE)
		{
			return FR_OK;
		}
	}
	return FR_NO_FILE;
}

FRESULT f_open_helper(FIL *fp, const TCHAR *path, int mode) {
    VSTR_FIXED(full_path, MICROPY_ALLOC_PATH_MAX)
    int res,fd;

    int mode_rw = 0, mode_x = 0;

    if ((mode & FA_READ) == FA_READ)
        mode_rw = O_RDONLY;

    if ((mode & FA_WRITE) == FA_WRITE) 
        mode_rw = O_WRONLY;

    if ((mode & FA_OPEN_ALWAYS) == FA_OPEN_ALWAYS)
        mode_x = O_CREAT | O_TRUNC;
	if ((mode & FA_CREATE_ALWAYS) == FA_CREATE_ALWAYS)
		mode_x = O_CREAT;
//    if ((mode & (FA_OPEN_EXISTING|FA_WRITE)) == (FA_OPEN_EXISTING|FA_WRITE))
//        mode_x = O_CREAT | O_APPEND;

    if (strstr(path,"/sd/") != 0)
    {
        fd = open(path, mode_x | mode_rw);
    }
    else
    {
        res = lookup_fullpath(path,&full_path);
        if (res == FR_NO_PATH) {
            return FR_NO_PATH;
        }
        fd = open(full_path.buf, mode_x | mode_rw);
    }

    if(fd < 0)
        return FR_NO_FILE;
    struct stat buf;
    fstat(fd,&buf);
    fp->obj.id = fd;
    fp->flag = mode;
    fp->obj.objsize = buf.st_size;
	fp->fptr = lseek(fd,0,SEEK_CUR);
    return FR_OK;
}
#if 0
FRESULT f_opendir_helper(FF_DIR *dp, const TCHAR *path) {
   #if 0 
    FATFS *fs = lookup_path(&path);
    if (fs == NULL) {
        return FR_NO_PATH;
    }
    return f_opendir(fs, dp, path);
    #else
    return FR_INVALID_PARAMETER;
    #endif
}
#endif
FRESULT f_stat_helper(const TCHAR *path, FILINFO *fno) {
    VSTR_FIXED(full_path, MICROPY_ALLOC_PATH_MAX)
    int res;
    struct stat buf;

    if (strstr(path,"/sd/") != 0)
    {
        res = stat(path,&buf);
    }
    else
    {
        res = lookup_fullpath(path,&full_path);
        if (res == FR_NO_PATH) {
            return FR_NO_PATH;
        }
        
        res = stat(full_path.buf,&buf);
    }

    fno->fsize = buf.st_size;

    return FR_OK;
}

FRESULT f_mkdir_helper(const TCHAR *path) {
#if 0     
    FATFS *fs = lookup_path(&path);
    if (fs == NULL) {
        return FR_NO_PATH;
    }
    return f_mkdir(fs, path);
#else
    return FR_INVALID_PARAMETER;
#endif    
}

FRESULT f_unlink_helper(const TCHAR *path) {
#if 0    
    FATFS *fs = lookup_path(&path);
    if (fs == NULL) {
        return FR_NO_PATH;
    }
    return f_unlink(fs, path);
#else
    return FR_INVALID_PARAMETER;
#endif 
}

FRESULT f_rename_helper(const TCHAR *path_old, const TCHAR *path_new) {
#if 0    
    FATFS *fs_old = lookup_path(&path_old);
    if (fs_old == NULL) {
        return FR_NO_PATH;
    }
    FATFS *fs_new = lookup_path(&path_new);
    if (fs_new == NULL) {
        return FR_NO_PATH;
    }
    if (fs_old != fs_new) {
        return FR_NO_PATH;
    }
    return f_rename(fs_new, path_old, path_new);
#else
    return FR_INVALID_PARAMETER;
#endif 
}

FRESULT f_touch_helper(const TCHAR *path) {
#if 0    
    FIL fp;
    FATFS *fs = lookup_path(&path);
    if (fs == NULL) {
        return FR_NO_PATH;
    }

    if (f_stat(fs, path, NULL) != FR_OK) {
        f_open(fs, &fp, path, FA_WRITE | FA_CREATE_ALWAYS);
        f_close(&fp);
    }

    return FR_OK;
#else
    return FR_INVALID_PARAMETER;
#endif 
}
// When a sector boundary is encountered while writing a file and there are
// more than 512 bytes left to write FatFs will detect that it can bypass
// its internal write buffer and pass the data buffer passed to it directly
// to the disk write function. However, the disk write function needs the
// buffer to be aligned to a 4-byte boundary. FatFs doesn't know this and
// will pass an unaligned buffer if we don't fix the issue. To fix this problem
// we use a temporary buffer to fix the alignment and to speed everything up.

// We use this temporary buffer for both reads and writes. The buffer allows us
// to do multi-block reads and writes which signifcantly speed things up.

static uint32_t file_buffer_offset = 0;
static uint8_t *file_buffer_pointer = 0;
static uint32_t file_buffer_size = 0;
static uint32_t file_buffer_index = 0;

void file_buffer_init0()
{
    file_buffer_offset = 0;
    file_buffer_pointer = 0;
    file_buffer_size = 0;
    file_buffer_index = 0;
}

OMV_ATTR_ALWAYS_INLINE static void file_fill(FIL *fp)
{
    if (file_buffer_index == file_buffer_size) {
        file_buffer_pointer -= file_buffer_offset;
        file_buffer_size += file_buffer_offset;
        file_buffer_offset = 0;
        file_buffer_index = 0;
		uint32_t current_offset = lseek(fp->obj.id,0,SEEK_CUR);
        uint32_t file_remaining = lseek(fp->obj.id,0,SEEK_END) - current_offset;
        uint32_t can_do = FF_MIN(file_buffer_size, file_remaining);
		lseek(fp->obj.id,current_offset,SEEK_SET);
        UINT bytes = read(fp->obj.id, file_buffer_pointer, can_do);
        if (bytes != can_do) ff_read_fail(fp->obj.id);
        fp->fptr = lseek(fp->obj.id,0,SEEK_CUR);
    }
}

OMV_ATTR_ALWAYS_INLINE static void file_flush(FIL *fp)
{
    if (file_buffer_index == file_buffer_size) {
        UINT bytes = write(fp->obj.id, file_buffer_pointer, file_buffer_index);
        if (bytes != file_buffer_index) ff_write_fail(fp->obj.id);
        fp->fptr = lseek(fp->obj.id,0,SEEK_CUR);
        file_buffer_pointer -= file_buffer_offset;
        file_buffer_size += file_buffer_offset;
        file_buffer_offset = 0;
        file_buffer_index = 0;
    }
}

uint32_t file_tell_w_buf(FIL *fp)
{
    if (fp->flag & FA_READ) {
        return lseek(fp->obj.id,0,SEEK_CUR)- file_buffer_size + file_buffer_index;
    } else {
        return lseek(fp->obj.id,0,SEEK_CUR) + file_buffer_index;
    }
}

uint32_t file_size_w_buf(FIL *fp)
{
    struct stat buf;
    fstat(fp->obj.id,&buf);
    int size = buf.st_size;

    if ((fp->flag & FA_READ)) {
        return size;
    } else {
        return size + file_buffer_index;
    }
}

void file_buffer_on(FIL *fp)
{
    file_buffer_offset = lseek(fp->obj.id,0,SEEK_CUR) % 4;
    file_buffer_pointer = fb_alloc_all(&file_buffer_size, FB_ALLOC_PREFER_SIZE) + file_buffer_offset;
    if (!file_buffer_size) {
        mp_raise_msg(&mp_type_MemoryError, MP_ERROR_TEXT("No memory!"));
    }
    file_buffer_size -= file_buffer_offset;
    file_buffer_index = 0;
    if ((fp->flag & FA_READ)) {
		uint32_t current_offset = lseek(fp->obj.id,0,SEEK_CUR);
        uint32_t file_remaining = lseek(fp->obj.id,0,SEEK_END) - current_offset;
		lseek(fp->obj.id,current_offset,SEEK_SET);
        uint32_t can_do = FF_MIN(file_buffer_size, file_remaining);
        UINT bytes = read(fp->obj.id, file_buffer_pointer, can_do);
        if (bytes != can_do) ff_read_fail(fp->obj.id);
        fp->fptr = lseek(fp->obj.id,0,SEEK_CUR);
    }
}

void file_buffer_off(FIL *fp)
{
    if ((fp->flag & FA_WRITE) && file_buffer_index) {
        UINT bytes = write(fp->obj.id, file_buffer_pointer, file_buffer_index);
        if (bytes != file_buffer_index) ff_write_fail(fp->obj.id);
        fp->fptr = lseek(fp->obj.id,0,SEEK_CUR);
    }
    file_buffer_pointer = 0;
    fb_free();
}

void read_byte(FIL *fp, uint8_t *value)
{
    if (file_buffer_pointer) {
        // We get a massive speed boost by buffering up as much data as possible
        // via massive reads. So much so that the time wasted by
        // all these operations does not cost us.
        for (size_t i = 0; i < sizeof(*value); i++) {
            file_fill(fp);
            ((uint8_t *) value)[i] = file_buffer_pointer[file_buffer_index++];
        }
    } else {
        UINT bytes = read(fp->obj.id, value, sizeof(*value));
        if (bytes != sizeof(*value)) ff_read_fail(fp->obj.id);
        fp->fptr = lseek(fp->obj.id,0,SEEK_CUR);
    }
}

void read_byte_expect(FIL *fp, uint8_t value)
{
    uint8_t compare;
    read_byte(fp, &compare);
    if (value != compare) ff_expect_fail(fp->obj.id);
}

void read_byte_ignore(FIL *fp)
{
    uint8_t trash;
    read_byte(fp, &trash);
}

void read_word(FIL *fp, uint16_t *value)
{
    if (file_buffer_pointer) {
        // We get a massive speed boost by buffering up as much data as possible
        // via massive reads. So much so that the time wasted by
        // all these operations does not cost us.
        for (size_t i = 0; i < sizeof(*value); i++) {
            file_fill(fp);
            ((uint8_t *) value)[i] = file_buffer_pointer[file_buffer_index++];
        }
    } else {
        UINT bytes = read(fp->obj.id, value, sizeof(*value));
        if (bytes != sizeof(*value)) ff_read_fail(fp->obj.id);
        fp->fptr = lseek(fp->obj.id,0,SEEK_CUR);
    }
}

void read_word_expect(FIL *fp, uint16_t value)
{
    uint16_t compare;
    read_word(fp, &compare);
    if (value != compare) ff_expect_fail(fp->obj.id);
}

void read_word_ignore(FIL *fp)
{
    uint16_t trash;
    read_word(fp, &trash);
}

void read_long(FIL *fp, uint32_t *value)
{
    if (file_buffer_pointer) {
        // We get a massive speed boost by buffering up as much data as possible
        // via massive reads. So much so that the time wasted by
        // all these operations does not cost us.
        for (size_t i = 0; i < sizeof(*value); i++) {
            file_fill(fp);
            ((uint8_t *) value)[i] = file_buffer_pointer[file_buffer_index++];
        }
    } else {
        UINT bytes = read(fp->obj.id, value, sizeof(*value));
        if (bytes != sizeof(*value)) ff_read_fail(fp->obj.id);
        fp->fptr = lseek(fp->obj.id,0,SEEK_CUR);
    }
}

void read_long_expect(FIL *fp, uint32_t value)
{
    uint32_t compare;
    read_long(fp, &compare);
    if (value != compare) ff_expect_fail(fp->obj.id);
}

void read_long_ignore(FIL *fp)
{
    uint32_t trash;
    read_long(fp, &trash);
}

void read_data(FIL *fp, void *data, UINT size)
{
    if (file_buffer_pointer) {
        // We get a massive speed boost by buffering up as much data as possible
        // via massive reads. So much so that the time wasted by
        // all these operations does not cost us.
        while (size) {
            file_fill(fp);
            uint32_t file_buffer_space_left = file_buffer_size - file_buffer_index;
            uint32_t can_do = FF_MIN(size, file_buffer_space_left);
            memcpy(data, file_buffer_pointer+file_buffer_index, can_do);
            file_buffer_index += can_do;
            data += can_do;
            size -= can_do;
        }
    } else {
        UINT bytes = read(fp->obj.id, data, size);
        if (bytes != size) ff_read_fail(fp->obj.id);
        fp->fptr = lseek(fp->obj.id,0,SEEK_CUR);
    }
}


void write_byte(FIL *fp, uint8_t value)
{
    if (file_buffer_pointer) {
        // We get a massive speed boost by buffering up as much data as possible
        // before a write to the SD card. So much so that the time wasted by
        // all these operations does not cost us.
        for (size_t i = 0; i < sizeof(value); i++) {
            file_buffer_pointer[file_buffer_index++] = ((uint8_t *) &value)[i];
            file_flush(fp);
        }
    } else {
        UINT bytes = write(fp->obj.id, &value, sizeof(value));
        if (bytes != sizeof(value)) ff_write_fail(fp->obj.id);
        fp->fptr = lseek(fp->obj.id,0,SEEK_CUR);
    }
}

void write_word(FIL *fp, uint16_t value)
{
    if (file_buffer_pointer) {
        // We get a massive speed boost by buffering up as much data as possible
        // before a write to the SD card. So much so that the time wasted by
        // all these operations does not cost us.
        for (size_t i = 0; i < sizeof(value); i++) {
            file_buffer_pointer[file_buffer_index++] = ((uint8_t *) &value)[i];
            file_flush(fp);
        }
    } else {
        UINT bytes = write(fp->obj.id, &value, sizeof(value));
        if (bytes != sizeof(value)) ff_write_fail(fp->obj.id);
        fp->fptr = lseek(fp->obj.id,0,SEEK_CUR);
    }
}

void write_long(FIL *fp, uint32_t value)
{
    if (file_buffer_pointer) {
        // We get a massive speed boost by buffering up as much data as possible
        // before a write to the SD card. So much so that the time wasted by
        // all these operations does not cost us.
        for (size_t i = 0; i < sizeof(value); i++) {
            file_buffer_pointer[file_buffer_index++] = ((uint8_t *) &value)[i];
            file_flush(fp);
        }
    } else {
        UINT bytes = write(fp->obj.id, &value, sizeof(value));
        if (bytes != sizeof(value)) ff_write_fail(fp->obj.id);
        fp->fptr = lseek(fp->obj.id,0,SEEK_CUR);
    }
}

void write_data(FIL *fp, const void *data, UINT size)
{
    if (file_buffer_pointer) {
        // We get a massive speed boost by buffering up as much data as possible
        // before a write to the SD card. So much so that the time wasted by
        // all these operations does not cost us.
        while (size) {
            uint32_t file_buffer_space_left = file_buffer_size - file_buffer_index;
            uint32_t can_do = FF_MIN(size, file_buffer_space_left);
            memcpy(file_buffer_pointer+file_buffer_index, data, can_do);
            file_buffer_index += can_do;
            data += can_do;
            size -= can_do;
            file_flush(fp);
        }
    } else {
        UINT bytes = write(fp->obj.id, data, size);
        if (bytes != size) ff_write_fail(fp->obj.id);
        fp->fptr = lseek(fp->obj.id,0,SEEK_CUR);
    }
}
#endif //IMLIB_ENABLE_IMAGE_FILE_IO
