#include "petsc_lattice.h"
#include "petsc_fermion.h"
#include "petsc_grid.h"

/*includes the global parameters*/
#include "petsc_fermion_parameters.h"

static char help[] = "Fermions on a hypercubic lattice.\n\n";

NAMESPACE_BEGIN(Grid);
int CheckDwWithGrid(DM dm,Vec psi,Vec res)
{
  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd,vComplexD::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();

  GridCartesian         GRID(latt_size,simd_layout,mpi_layout);
  GridRedBlackCartesian RBGRID(&GRID);

  std::vector<int> seeds({1,2,3,4});
  GridParallelRNG          pRNG(&GRID);
  pRNG.SeedFixedIntegers(seeds);
  
  LatticeGaugeFieldD Umu(&GRID);
  if (0) {
    SU3::TepidConfiguration(pRNG,Umu); /*random config*/
    //    Umu = ComplexD(1.0,0.0);    /*Unit/free config*/
  } else {
    std::string name ("ckpoint_lat.4000");
    std::cout<<GridLogMessage <<"Loading configuration from "<< name<<std::endl;
    FieldMetaData header;
    NerscIO::readConfiguration(Umu, header, name);
  }

  std::cout<<"****************************************"<<std::endl;
  std::cout << "Gauge invariance test \n";
  std::cout<<"****************************************"<<std::endl;
  LatticeGaugeField     U_GT(&GRID); // Gauge transformed field
  LatticeColourMatrix   g(&GRID);    // local Gauge xform matrix
  U_GT = Umu;
  // Make a random xform to the gauge field
  //  SU<Nc>::RandomGaugeTransform(pRNG,U_GT,g); // Unit gauge
  ////////////////////////////////////////////////////
  // Wilson test
  ////////////////////////////////////////////////////
  LatticeFermionD    g_src(&GRID);
  LatticeFermionD    g_res(&GRID); // Grid result
  LatticeFermionD    p_res(&GRID); // Petsc result
  LatticeFermionD    diff(&GRID); // Petsc result
  LatticeFermionD    tmp(&GRID);

  PetscToGrid(dm,psi,g_src);
  
  WilsonParameters *p = GetWilsonParameters();
  RealD mass=p->M;
  WilsonFermionD Dw(U_GT,GRID,RBGRID,mass);
    

  std::cout << "Setting gauge to Grid "<<std::endl;
  SetGauge_Grid(dm,U_GT);

  std::cout << "Testing Dw "<<std::endl;
  const int ncall = 10;
  const int ncallg = 10;
  RealD t0=usecond();
  for(int i=0;i<ncallg;i++){
    Dw.M(g_src,g_res);
  }
  RealD t1=usecond();
  for(int i=0;i<ncall;i++){
      PetscCall(Dwilson(dm, psi, res)); // Applies DW
  }
  RealD t2=usecond();
  PetscToGrid(dm,res,p_res); 
  
  diff = p_res - g_res;
  std::cout << "******************************"<<std::endl;
  std::cout << "CheckDwWithGrid Grid  " << norm2(g_res)<<std::endl;
  std::cout << "CheckDwWithGrid Petsc " << norm2(p_res)<<std::endl;
  std::cout << "CheckDwWithGrid diff  " << norm2(diff)<<std::endl;
  std::cout << "Grid  " << (t1-t0)/1000/ncallg <<" ms"<<std::endl;
  std::cout << "Petsc " << (t2-t1)/1000/ncall <<" ms"<<std::endl;
  std::cout << "******************************"<<std::endl;

  Dw.Mdag(g_src,g_res);

  PetscCall(DwilsonDag(dm, psi, res)); // Applies DW
  PetscToGrid(dm,res,p_res); 
  
  diff = p_res - g_res;
  
  std::cout << "******************************"<<std::endl;
  std::cout << "CheckDwDagWithGrid Grid  " << norm2(g_res)<<std::endl;
  std::cout << "CheckDwDagWithGrid Petsc " << norm2(p_res)<<std::endl;
  std::cout << "CheckDwDagWithGrid diff  " << norm2(diff)<<std::endl;
  std::cout << "******************************"<<std::endl;

  
  assert(norm2(diff) < 1.0e-7);

  std::cout << "******************************"<<std::endl;
  std::cout << " Wilson CGNR solve with Grid " <<std::endl;
  std::cout << "******************************"<<std::endl;
  MdagMLinearOperator<WilsonFermionD,LatticeFermionD> HermOp(Dw);
  ConjugateGradient<LatticeFermionD> CG(1.0e-8,10000);

  gaussian(pRNG,g_src);
  Dw.Mdag(g_src,tmp);
  CG(HermOp,tmp,g_res);

  return 0;
}
NAMESPACE_END(Grid);

int main(int argc, char **argv)
{
  Grid::Grid_init(&argc,&argv);

  WilsonParameters parm;
  parm.M = 0.1;  // 78 iters
  parm.M = -0.5; // 174
  parm.M = -0.7; // 318
  parm.M = -0.75;// 403
  parm.M = -0.80;// 551
  parm.M = -0.85;// 874
  parm.M = -0.87;// 1137
  //  parm.M = -0.88;// 1348
  //  parm.M = -0.89;// 1630
  parm.M = -0.90;//2088
  parm.M = -0.91;//2882
  parm.M = -0.92;//4057
  SetWilsonParameters(&parm);
  
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

  PetscCall(SetGauge_Identity(dm));

  Grid::CheckDwWithGrid(dm,u,f);

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
    args: -dm_plex_dim 4 -dm_plex_shape hypercubic -dm_plex_box_faces 16,16,16,32 -dm_view \
          -dm_plex_check_symmetry -dm_plex_check_skeleton -dm_plex_check_faces -dm_plex_check_pointsf --grid 16.16.16.32

TEST*/
