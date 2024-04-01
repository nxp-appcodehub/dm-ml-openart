/*
 * Copyright (c) 2006-2018, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */
#ifndef __FAKE__SYS_STATVFS_H__
#define __FAKE__SYS_STATVFS_H__
struct statvfs {
               unsigned long  f_bsize;    /* Filesystem block size */
               unsigned long  f_frsize;   /* Fragment size */
               unsigned long     f_blocks;   /* Size of fs in f_frsize units */
               unsigned long     f_bfree;    /* Number of free blocks */
               unsigned long     f_bavail;   /* Number of free blocks for
                                             unprivileged users */
               unsigned long     f_files;    /* Number of inodes */
               unsigned long     f_ffree;    /* Number of free inodes */
               unsigned long     f_favail;   /* Number of free inodes for
                                             unprivileged users */
               unsigned long  f_fsid;     /* Filesystem ID */
               unsigned long  f_flag;     /* Mount flags */
               unsigned long  f_namemax;  /* Maximum filename length */
           };


#endif