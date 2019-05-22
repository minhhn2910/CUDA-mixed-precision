import sys
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("half", help="file containing half precision code")
parser.add_argument("float", help="file containing floating point code")
parser.add_argument("speed", help="fraction of threads to be run in half  precision from the total number of threads", type=int)
parser.add_argument("-o", "--output", default="mixed_precision_code.cu")
args = parser.parse_args()

# check if file paths exist
half_file_exists = os.path.isfile(args.half)
if not half_file_exists:
    print("Cannot open half precision code file at : " + args.half)
    quit()

float_file_exists = os.path.isfile(args.float)
if not float_file_exists:
    print("Cannot open floating point code file at : " + args.float)
    quit()

# read files and check if necessary comments exist in code
half_contains_main = 0
with open(args.half, 'r') as half_file:
    half_content = half_file.read()
if(not "MP_BEGIN" in half_content or not "MP_END" in half_content):
    print("Annotate the beginning and end of the half precision function body with comments \"MP_BEGIN\" and \"MP_END\"")
    quit()
if("MP_MAIN_BEGIN" in half_content and "MP_MAIN_END" in half_content):
    half_contains_main = 1

float_contains_main = 0
with open(args.float, 'r') as float_file:
    float_content = float_file.read()
if(not "MP_BEGIN" in float_content or not "MP_END" in float_content):
    print("Annotate the beginning and end of the floating point function body with comments \"MP_BEGIN\" and \"MP_END\"")
    quit()
if("MP_MAIN_BEGIN" in float_content and "MP_MAIN_END" in float_content):
    float_contains_main = 1

if(not half_contains_main and not float_contains_main):
    print("Annotate the beginning and end of the main function body in either the float or half precision code with comments \"MP_MAIN_BEGIN\" and \"MP_MAIN_END\"")
    quit()

# create mixed precision code
# get the function bodies of methods
half_body = half_content.split("MP_BEGIN")[1].split("MP_END")[0]
half_body = half_body[half_body.find('\n') + 1:half_body.rfind('\n')]
float_body = float_content.split("MP_BEGIN")[1].split("MP_END")[0]
float_body = float_body[float_body.find('\n') + 1:float_body.rfind('\n')]

# get the main function
main_body = ""
if(half_contains_main):
    main_body = half_content.split("MP_MAIN_BEGIN")[1].split("MP_MAIN_END")[0]
else:
    main_body = float_content.split("MP_MAIN_BEGIN")[1].split("MP_MAIN_END")[0]
main_body = main_body[main_body.find('\n') + 1:main_body.rfind('\n')]

# get the function call
method_signature = [line for line in float_content.split('\n') if "__global" in line][0]

#add speed arguments
method_signature = method_signature.split(')')[0] + ", int speed)"
invocations = [line for line in float_content.split('\n') if "<<<" in line]
replacements = []
for invocation in invocations:
    replacements.append((invocation, invocation.split(")")[0] + ", speed);"))
for replacement in replacements:
    main_body = main_body.replace(replacement[0], replacement[1])

# get all the includes, defines and using lines
half_includes = [line for line in half_content.split('\n') if line.startswith("#include")]
float_includes = [line for line in float_content.split('\n') if line.startswith("#include")]
half_defines = [line for line in half_content.split('\n') if line.startswith("#define")]
float_defines = [line for line in float_content.split('\n') if line.startswith("#define")]
half_using = [line for line in half_content.split('\n') if line.startswith("using")]
float_using = [line for line in float_content.split('\n') if line.startswith("using")]
# Remove duplicates
includes = float_includes + list(set(half_includes) - set(float_includes))
defines = float_defines + list(set(half_defines) - set(float_defines))
usings = float_using + list(set(half_using) - set(float_using))

# peice everything together
mp_code = ""
for include in includes:
    if(not "fast_math" in include):
        mp_code += (include + "\n")
mp_code += "\n"

for define in defines:
    mp_code += (define + "\n")
mp_code += "\n"

for using in usings:
    mp_code += (using + "\n")
mp_code += "\n"

mp_code += "#ifdef SLOW_MATH\n#include \"../include/cuda_math.cuh\"\n#else\n#include \"../include/fast_math.cuh\"\n#endif"

mp_code += "\n\n#define SCATTER\n\n"

for using in usings:
    mp_code += (using + "\n")

mp_code += method_signature + " {\n#ifdef SCATTER\n\tif(blockIdx.x %100 < speed) {\n#else\n\tif(blockIdx.x < speed) {\n#endif\n"

mp_code += (half_body + "\n} else {" + float_body + "\n}\n}\n")

mp_code += ("\nint main(int argc, char* argv[])\n{\n\tint speed = " + str(args.speed) + ";\n\tstd::cout << \"# Speed = \" << speed << std::endl;\n\n" + main_body + "\n}")

# output file
output_file = open(args.output, "w")
output_file.write(mp_code)
output_file.close()

