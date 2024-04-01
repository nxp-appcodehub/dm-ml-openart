# Copyright 2022-2023 NXP
import os
import sys
from auto_generate_qstr import gen_qstr

path = os.path.normpath(os.getcwd()) + '/../../'
print('Generate Qstring in :',path)
headerfile = './mimxrt_port/genhdr/qstrdefs.generated.h'
gen_qstr(path=path,hash_len=2,headerfile=headerfile)