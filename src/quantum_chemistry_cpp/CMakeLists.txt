cmake_minimum_required(VERSION 3.15)
project(simcim_quantum_chemistry)

set(CMAKE_CXX_STANDARD 17)

message(STATUS "Checking OpenMP")
find_package(OpenMP)
IF(OPENMP_FOUND)
    message("Found OpenMP!")
    # add flags for OpenMP
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")
ELSE()
    message("Missed OpenMP!")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
ENDIF()

# Boost
set(boost_min_ver 1.50.0)
set(boost_libs system filesystem)
set(Boost_USE_STATIC_LIBS   ON)
find_package(Boost ${boost_min_ver} REQUIRED)

if(Boost_FOUND)
    find_package(Boost ${boost_min_ver} COMPONENTS ${boost_libs} REQUIRED)
endif()

include_directories(${Boost_INCLUDE_DIRS})

# Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(simcim_quantum_chemistry source/main.cpp source/Hamiltonian.cpp headers/Hamiltonian.h source/Term.cpp headers/Term.h source/quantum_chemistry.cpp headers/quantum_chemistry.h source/two_local_simcim.cpp headers/two_local_simcim.h source/utils.cpp headers/utils.h source/ziggurat.cpp headers/ziggurat.h source/simcim.cpp headers/simcim.h source/k_local_simcim.cpp headers/k_local_simcim.h headers/use_open_mp.h headers/use_open_mp.h)

target_link_libraries(simcim_quantum_chemistry ${Boost_LIBRARIES})
target_link_libraries(simcim_quantum_chemistry Eigen3::Eigen)