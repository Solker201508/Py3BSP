project(Py3BSP)
cmake_minimum_required(VERSION 2.6)
set( CMAKE_CXX_FLAGS "-O3 -fopenmp" )
set( CMAKE_LD_FLAGS "-O3 -fopenmp" )
set( CMAKE_C_COMPILER "mpicc" )
set( CMAKE_CXX_COMPILER "mpic++" )
set( CMAKE_LINKER "mpic++" )

exec_program( python3-config ARGS --includes | cut -d' '  -f1 | sed 's/-I//g' OUTPUT_VARIABLE PYTHON3_INC )
message ( ${PYTHON3_INC} )
exec_program( python3-config ARGS --ldflags OUTPUT_VARIABLE PYTHON3_LIB )
exec_program( python3 ARGS -c "'import numpy;print(numpy.__path__[0])'" OUTPUT_VARIABLE NUMPY )
set( NUMPY_INC ${NUMPY}/core/include )
message( ${NUMPY_INC} )

aux_source_directory( src/runtime DIR_RUNTIME )
aux_source_directory( src/frontend DIR_FRONTEND )
add_executable( py3bsp ${DIR_FRONTEND} ${DIR_RUNTIME} )

find_path( OS_INC stdio.h PATHS /usr/include /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/usr/include )
include_directories( include/frontend include/runtime ${OS_INC} ${PYTHON3_INC} ${NUMPY_INC} )

target_link_libraries( py3bsp ${PYTHON3_LIB} )
