import numpy as np



def read_calibration_file(file):
    global mtx
    global dist
    # read calibration file
    data_mtx = []
    data_dist = []
    with open(file, "r" ) as file: 
        lines = file.readlines()
        state = None
        for line in lines: 
            header = line.split('=')
            # search for identity
            if(header[0] == 'mtx'): 
                state = 'mtx'
                continue
            elif(header[0] == 'dist'): 
                state = 'dist'
                continue
            # append data
            if (state == 'mtx'): 
                header = line.split('\n')
                data_mtx.append(header[0])  # remove \n
            elif (state == 'dist'): 
                header = line.split('\n')   # remove \n
                data_dist.append(header[0])
    # convert data to array
    str_mtx = ''.join(data_mtx).replace('[','').replace(']','')
    mtxfloat = np.array(list(filter(None,str_mtx.strip(' ').split(' ')))).astype(np.float32).reshape(3,3)
    # print(mtxfloat)
    str_dist = ''.join(data_dist).replace('[','').replace(']','')
    distfloat = np.array(list(filter(None,str_dist.strip(' ').split(' ')))).astype(np.float32)
    # print(distfloat)
    mtx = mtxfloat
    dist = distfloat