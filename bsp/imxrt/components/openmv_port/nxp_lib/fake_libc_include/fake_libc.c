/*
 * Copyright (c) 2006-2018, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */
#include <stdio.h>
#include "sys/statvfs.h"

int statvfs (const char *file, struct statvfs *buf)
{
    if (buf)
        memset(buf,0,sizeof(struct statvfs));
	
	return 0;
}