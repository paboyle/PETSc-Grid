LOCDIR		 = 
MANSEC           = SNES
EXAMPLESMATLAB   = 
DIRS		 = 
CLEANFILES       = 

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

GRID=/Users/peterboyle/QCD/Grid-petsc/Grid/installs/mac-arm/
GRID_CXX=`$(GRID)/bin/grid-config --cxx`
GRID_CXXFLAGS=`$(GRID)/bin/grid-config --cxxflags ` -I$(GRID)/include/ -Wno-unused-local-typedef -Wno-unused-variable -Wno-reorder-ctor -Wno-unused-but-set-variable -Wno-unused-function
GRID_LIBS=`$(GRID)/bin/grid-config --libs` -lGrid
GRID_LDFLAGS=`$(GRID)/bin/grid-config --ldflags  ` -L$(GRID)/lib/ 

CXXFLAGS+=$(GRID_CXXFLAGS)
CXX_LINKER_FLAGS+=$(GRID_LIBS) $(GRID_LDFLAGS)
