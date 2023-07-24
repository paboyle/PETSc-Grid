#include "petsc_lattice.h"
#include "petsc_fermion.h"
#include "petsc_grid.h"

/*includes the global parameters*/
#include "petsc_fermion_parameters.h"

static char help[] = "Shamir Domain Wall fermions on a hypercubic lattice.\n\n";
NAMESPACE_BEGIN(Grid);
int CheckDwfWithGrid(DM dm,Vec psi,Vec res)
{
  DomainWallParameters p;
  p.M5   = 1.8;  
  p.m    = 0.01;
  p.Ls   = 16;
  
  SetDomainWallParameters(&p);
  
  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd,vComplexD::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();

  GridCartesian         GRID(latt_size,simd_layout,mpi_layout);
  GridRedBlackCartesian RBGRID(&GRID);

  std::vector<int> seeds({1,2,3,4});
  GridParallelRNG          pRNG(&GRID);
  pRNG.SeedFixedIntegers(seeds);
  
  LatticeGaugeFieldD Umu(&GRID);
  LatticeGaugeField     U_GT(&GRID); // Gauge transformed field
  LatticeColourMatrix   g(&GRID);    // local Gauge xform matrix

  if (0) {

    SU3::TepidConfiguration(pRNG,Umu); /*random config*/
    //Umu = ComplexD(1.0,0.0);  /*Unit config*/
    U_GT = Umu;
    // Make a random xform to the gauge field
    SU<Nc>::RandomGaugeTransform(pRNG,U_GT,g); // Unit gauge
  } else {
    std::string name ("ckpoint_lat.4000");
    std::cout<<GridLogMessage <<"Loading configuration from "<< name<<std::endl;
    FieldMetaData header;
    NerscIO::readConfiguration(Umu, header, name);
    U_GT = Umu;
  }
  
  ////////////////////////////////////////////////////
  // DWF test
  ////////////////////////////////////////////////////
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(p.Ls,&GRID);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(p.Ls,&GRID);
  
  LatticeFermionD    g_src(FGrid);
  LatticeFermionD    g_res(FGrid); // Grid result
  LatticeFermionD    p_res(FGrid); // Petsc result
  LatticeFermionD    diff(FGrid); // Petsc result

  std::cout << "Setting gauge to Grid for Ls="<<p.Ls<<std::endl;
  SetGauge_Grid5D(dm,U_GT,p.Ls,p.m);

  PetscToGrid(dm,psi,g_src);
  
  DomainWallFermionD DWF(U_GT,*FGrid,*FrbGrid,GRID,RBGRID,p.m,p.M5);
    
  //  DWF.Dhop(g_src,g_res,0); Passes with no 5d term
  RealD t0=usecond();
  DWF.M(g_src,g_res);
  RealD t1=usecond();
  PetscCall(Ddwf(dm, psi, res)); // Applies DW
  RealD t2=usecond();
  PetscToGrid(dm,res,p_res); 
  
  diff = p_res - g_res;

  std::cout << "******************************"<<std::endl;
  std::cout << "CheckDwWithGrid Grid  " << norm2(g_res)<<std::endl;
  std::cout << "CheckDwWithGrid Petsc " << norm2(p_res)<<std::endl;
  std::cout << "CheckDwWithGrid diff  " << norm2(diff)<<std::endl;
  std::cout << "Grid  " << t1-t0 <<" us"<<std::endl;
  std::cout << "Petsc " << t2-t1 <<" us"<<std::endl;
  std::cout << "******************************"<<std::endl;

  DWF.Mdag(g_src,g_res);

  PetscCall(DdwfDag(dm, psi, res)); // Applies DW
  PetscToGrid(dm,res,p_res); 
  
  diff = p_res - g_res;

  std::cout << "******************************"<<std::endl;
  std::cout << "CheckDwDagWithGrid Grid  " << norm2(g_res)<<std::endl;
  std::cout << "CheckDwDagWithGrid Petsc " << norm2(p_res)<<std::endl;
  std::cout << "CheckDwDagWithGrid diff  " << norm2(diff)<<std::endl;
  std::cout << "******************************"<<std::endl;

  assert(norm2(diff) < 1.0e-7);
  return 0;
}
NAMESPACE_END(Grid);

int main(int argc, char **argv)
{
  Grid::Grid_init(&argc,&argv);

  DM     dm;
  Vec    u, f;
  Vec    p;
  Mat    J;
  AppCtx user;

  PetscRandom        r;
  
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SetupDiscretization(dm, &user));
  PetscCall(SetupAuxDiscretization(dm, &user));

  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(DMCreateGlobalVector(dm, &f));

  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &r));
  PetscCall(PetscRandomSetType(r, PETSCRAND48));
  PetscCall(VecSetRandom(u, r));
  PetscCall(PetscRandomDestroy(&r));

  Grid::CheckDwfWithGrid(dm,u,f);

  PetscCall(VecDestroy(&f));
  PetscCall(VecDestroy(&u));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());

  Grid::Grid_finalize();
  return 0;
}

/*TEST

  build:
    requires: complex

  test:
    requires: fftw
    suffix: dirac_free_field
        args: 
            -dm_plex_dim 5 -dm_plex_shape hypercubic -dm_plex_box_faces 16,16,16,16,32 \
            -dm_plex_check_symmetry -dm_plex_check_skeleton -dm_plex_check_faces -dm_plex_check_pointsf

TEST*/
