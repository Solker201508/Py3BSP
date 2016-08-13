project(Py3BSP)
cmake_minimum_required(VERSION 2.6)
set( CMAKE_CXX_FLAGS -O3 )
set( CMAKE_LD_FLAGS -O3 )

exec_program( python3-config ARGS --includes | cut -d' '  -f1 | sed 's/-I//g' OUTPUT_VARIABLE PYTHON3_INC )
message ( ${PYTHON3_INC} )
exec_program( python3-config ARGS --ldflags OUTPUT_VARIABLE PYTHON3_LIB )
exec_program( python3 ARGS -c "'import numpy;print(numpy.__path__[0])'" OUTPUT_VARIABLE NUMPY )
set( NUMPY_INC ${NUMPY}/core/include )
message( ${NUMPY_INC} )

aux_source_directory( src/runtime DIR_RUNTIME )
aux_source_directory( src/frontend DIR_FRONTEND )
add_executable( py3bsp ${DIR_FRONTEND} ${DIR_RUNTIME} )

find_path( MPI_INC mpi.h PATHS /usr/include /usr/include/mpi /usr/local/include /usr/local/include/mpi /usr/local/mpi3-dynamic/include /public/soft/openmpi143-gnu/include)
include_directories( include/frontend include/runtime ${MPI_INC} ${PYTHON3_INC} ${NUMPY_INC} )

find_library( MPI_LIB NAMES mpi PATHS /usr/local/mpi3-dynamic/lib /usr/local/lib /usr/lib /public/soft/openmpi143-gnu/lib )
find_library( MPICXX_LIB NAMES mpi_cxx mpicxx PATHS /usr/local/mpi3-dynamic/lib /usr/local/lib /usr/lib /public/soft/openmpi143-gnu/lib )
target_link_libraries( py3bsp ${MPI_LIB} ${MPICXX_LIB} ${PYTHON3_LIB} )
