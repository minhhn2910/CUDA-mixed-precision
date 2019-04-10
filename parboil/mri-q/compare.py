import sys
import struct

def readfile(filename):
    f = open(filename,'r')
    raw_data = f.read()
    len_data = len(raw_data)
    num_elem = len_data/4 #4 bytes per fp32 val, change if other format
    result = struct.unpack('f'*num_elem, raw_data)
    return result
def compare_float (arr1,arr2):
    err = 0.0
    rmse = 0.0
    for i in range(len(arr1)):
        curr_err =  arr2[i]
        if (arr2[i])!= 0:
            curr_err = abs((arr1[i] - arr2[i])/arr2[i])
        if (curr_err > 1):
            curr_err = 1
        rmse = rmse + (arr1[i] - arr2[i])*(arr1[i] - arr2[i])
        err = err + curr_err
    rmse = rmse/len(arr1)
    return err/len(arr1), rmse**0.5

def main(argv):
    if (len(argv) !=2):
        print "usage python compare.py file1 ref"
    arr1 = readfile(argv[0])
    arr2 = readfile(argv[1])
    if len(arr1) != len(arr2):
        print "len mismatch"
    print len(arr1)
    rel_err, rmse = compare_float(arr1, arr2)
    print "rel err %f  rmse %f"%(rel_err, rmse)
if __name__ == '__main__':
    main(sys.argv[1:])
