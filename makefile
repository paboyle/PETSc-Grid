LOCDIR		 = 
MANSEC           = SNES
EXAMPLESMATLAB   = 
DIRS		 = 
CLEANFILES       = 

PETSC_DIR=${HOME}/QCD/WP1/petsc
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

GRID=${HOME}/QCD/Grid-develop/install/
GRID_CXX=`$(GRID)/bin/grid-config --cxx`
GRID_CXXFLAGS=`$(GRID)/bin/grid-config --cxxflags ` -I$(GRID)/include/ -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-function
GRID_LIBS=`$(GRID)/bin/grid-config --libs` -lGrid
GRID_LDFLAGS=`$(GRID)/bin/grid-config --ldflags  ` -L$(GRID)/lib/ 

PETSC_LIB+=$(GRID_LIBS)
CXXLINKER+=$(GRID_LDFLAGS)
CXXCPPFLAGS+=$(GRID_CXXFLAGS)

