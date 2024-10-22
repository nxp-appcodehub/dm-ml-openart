#Copyright 2023 NXP 
import time
def compute_hash(key, modulo=17):
    hash = 5381
    bKey = key.encode()
    for b in bKey:
        hash = (hash * 33) ^ b
    return hash % modulo

def open_helpfile(key):
    hash = compute_hash(key)
    sPath = '/aia_doc/zzhsh_%02d.md' % hash
    fd = open(sPath, 'ab')

import os
import os.path as path

def modmain():

    def OpenOutFile(sLine):
        key = sLine[5:]
        hash = compute_hash(key)
        sOutPath = 'zzhsh_%02d.md' % hash
        fdOut = open(sOutPath, 'ab')
        return fdOut, key

    os.system('del zzhsh_*.md')
    cnt = 0
    dict = {}
    for i in range(17):
        dict[i] = []
    for root, dirs, files in os.walk('.', topdown=False):
        for file in files:
            if file[-3:] != '.md' or file[:6] == 'zzhsh_':
                continue
            sPath = path.join(root, file)
            fd = open(sPath, encoding='utf-8')
            s = fd.read()
            lst = s.split('\n')
            fd.close()
            state = 0
            for sLine in lst:
                sLine = sLine.rstrip()
                if state == 0:
                    if sLine[:5] == '#### ':
                        state = 1
                        key = sLine[5:]
                        hash = compute_hash(key, 17)
                        #fdOut, key = OpenOutFile(sLine)
                if state == 1:
                    if sLine[:5] == '#### ':                    
                        if sLine[5:] != key:
                            #fdOut.close()
                            key = sLine[5:]
                            hash = compute_hash(key, 17)
                            #fdOut,key = OpenOutFile(sLine)
                    dict[hash].append((sLine))
                    #fdOut.write((sLine+'\r\n').encode())
    for i in range(17):
        lst = dict[i]
        sOut = '\r\n'.join(lst) + '\r\n'
        print('.', end='')
        if len(lst) < 1:
            continue
        fd = open('zzhsh_%02d.md' % i, 'wb')
        fd.write(sOut.encode())
        fd.close()

if __name__ == '__main__':
    #help()
    modmain()
