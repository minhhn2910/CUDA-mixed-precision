import sys
def parse_file(file_t):
   result = []
   with open(file_t) as res_file:
        line_num = 0
        for line in res_file:
            #print line
            line_num = line_num+1
            array = line.split(' ')
            for target in array:
                if len(target) > 0 and ord(target[0]) != 10:
                    try:
                        result.append(float(target))
                    except:
                        print ("cant convert " + target )
                        print ord(target[0])
   return result
def calculate_err (array_1, array_2):
    #calculate avg rel err
    err = 0.0;
    rmse = 0.0;
    for i in range(len(array_1)):
        #if i == 0:
        #    print "%f %f %f "%(array_1[i],array_2[i], abs((array_1[i]- array_2[i])/ array_2[i]))
        if array_2[i]!=0:
            err += abs((array_1[i]- array_2[i])/ array_2[i])
        else:
            err += abs(array_1[i])
        rmse = rmse + (array_1[i]- array_2[i])*(array_1[i]- array_2[i]);

    #print ("sum err %f "%(err))
    return err/float(len(array_1)), (rmse/float(len(array_1)))**0.5
def main(argv):
    if (len(argv) != 2):
        print ("usage: python calculate_error.py file1.txt file2.txt")
    file1 = argv[0]
    file2 = argv[1]
    array_1 = parse_file(file1)
    array_2 = parse_file(file2)
    if (len(array_1)!= len(array_2)):
        print ("len mismatch by %d "%(len(array_1) - len(array_2)))
    print ("len array %d " %(len(array_1)))
    err,rmse = calculate_err (array_1, array_2)
    print ("err %f  rmse %f"%(err,rmse))
if __name__ == '__main__':
    main(sys.argv[1:])
