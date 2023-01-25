static char help[] = "Fermions on a hypercubic lattice.\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>

/* Common operations:

 - View the input \psi as ASCII in lexicographic order: -psi_view
*/

const int Ls=16;
const PetscReal M = 1.0; // M5
const PetscReal m = 0.01; // mf

typedef struct {
  PetscBool usePV; /* Use Pauli-Villars preconditioning */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBegin;
  options->usePV = PETSC_TRUE;

  PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-use_pv", "Use Pauli-Villars preconditioning", "ex1.c", options->usePV, &options->usePV, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *ctx)
{
  PetscSection s;
  PetscInt     vStart, vEnd, v;

  PetscFunctionBegin;
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(PetscSectionSetChart(s, vStart, vEnd));
  for (v = vStart; v < vEnd; ++v) {
    PetscCall(PetscSectionSetDof(s, v, 12));
    /* TODO Divide the values into fields/components */
  }
  PetscCall(PetscSectionSetUp(s));
  PetscCall(DMSetLocalSection(dm, s));
  PetscCall(PetscSectionDestroy(&s));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupAuxDiscretization(DM dm, AppCtx *user)
{
  DM           dmAux, coordDM;
  PetscSection s;
  Vec          gauge;
  PetscInt     eStart, eEnd, e;

  PetscFunctionBegin;
  /* MUST call DMGetCoordinateDM() in order to get p4est setup if present */
  PetscCall(DMGetCoordinateDM(dm, &coordDM));
  PetscCall(DMClone(dm, &dmAux));
  PetscCall(DMSetCoordinateDM(dmAux, coordDM));
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
  PetscCall(PetscSectionSetChart(s, eStart, eEnd));
  for (e = eStart; e < eEnd; ++e) {
    /* TODO Should we store the whole SU(3) matrix, or the symmetric part? */
    PetscCall(PetscSectionSetDof(s, e, 9));
  }
  PetscCall(PetscSectionSetUp(s));
  PetscCall(DMSetLocalSection(dmAux, s));
  PetscCall(PetscSectionDestroy(&s));
  PetscCall(DMCreateLocalVector(dmAux, &gauge));
  PetscCall(DMDestroy(&dmAux));
  PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 0, gauge));
  PetscCall(VecDestroy(&gauge));
  PetscFunctionReturn(0);
}

static PetscErrorCode PrintVertex(DM dm, PetscInt v)
{
  MPI_Comm       comm;
  PetscContainer c;
  PetscInt      *extent;
  PetscInt       dim, cStart, cEnd, sum;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(PetscObjectQuery((PetscObject)dm, "_extent", (PetscObject *)&c));
  PetscCall(PetscContainerGetPointer(c, (void **)&extent));
  sum = 1;
  PetscCall(PetscPrintf(comm, "Vertex %" PetscInt_FMT ":", v));
  for (PetscInt d = 0; d < dim; ++d) {
    PetscCall(PetscPrintf(comm, " %" PetscInt_FMT, (v / sum) % extent[d]));
    if (d < dim) sum *= extent[d];
  }
  PetscFunctionReturn(0);
}

// Apply \gamma_\mu
static PetscErrorCode ComputeGamma(PetscInt d, PetscInt ldx, PetscScalar f[])
{
  const PetscScalar fin[4] = {f[0 * ldx], f[1 * ldx], f[2 * ldx], f[3 * ldx]};

  PetscFunctionBeginHot;
  switch (d) {
  case 0:
    f[0 * ldx] = PETSC_i * fin[3];
    f[1 * ldx] = PETSC_i * fin[2];
    f[2 * ldx] = -PETSC_i * fin[1];
    f[3 * ldx] = -PETSC_i * fin[0];
    break;
  case 1:
    f[0 * ldx] = -fin[3];
    f[1 * ldx] = fin[2];
    f[2 * ldx] = fin[1];
    f[3 * ldx] = -fin[0];
    break;
  case 2:
    f[0 * ldx] = PETSC_i * fin[2];
    f[1 * ldx] = -PETSC_i * fin[3];
    f[2 * ldx] = -PETSC_i * fin[0];
    f[3 * ldx] = PETSC_i * fin[1];
    break;
  case 3:
    f[0 * ldx] = fin[2];
    f[1 * ldx] = fin[3];
    f[2 * ldx] = fin[0];
    f[3 * ldx] = fin[1];
    break;
  case 4:
    f[0 * ldx] = fin[0];
    f[1 * ldx] = fin[1];
    f[2 * ldx] =-fin[2];
    f[3 * ldx] =-fin[3];
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Direction for gamma %" PetscInt_FMT " not in [0, 4)", d);
  }
  PetscFunctionReturn(0);
}

// Apply (1 \pm \gamma_\mu)/2
static PetscErrorCode ComputeGammaFactor(PetscInt d, PetscBool forward, PetscInt ldx, PetscScalar f[])
{
  const PetscReal   sign   = forward ? -1. : 1.;
  const PetscScalar fin[4] = {f[0 * ldx], f[1 * ldx], f[2 * ldx], f[3 * ldx]};

  PetscFunctionBeginHot;
  switch (d) {
  case 0:
    f[0 * ldx] += sign * PETSC_i * fin[3];
    f[1 * ldx] += sign * PETSC_i * fin[2];
    f[2 * ldx] += sign * -PETSC_i * fin[1];
    f[3 * ldx] += sign * -PETSC_i * fin[0];
    break;
  case 1:
    f[0 * ldx] += -sign * fin[3];
    f[1 * ldx] += sign * fin[2];
    f[2 * ldx] += sign * fin[1];
    f[3 * ldx] += -sign * fin[0];
    break;
  case 2:
    f[0 * ldx] += sign * PETSC_i * fin[2];
    f[1 * ldx] += sign * -PETSC_i * fin[3];
    f[2 * ldx] += sign * -PETSC_i * fin[0];
    f[3 * ldx] += sign * PETSC_i * fin[1];
    break;
  case 3:
    f[0 * ldx] += sign * fin[2];
    f[1 * ldx] += sign * fin[3];
    f[2 * ldx] += sign * fin[0];
    f[3 * ldx] += sign * fin[1];
    break;
  case 4:
    f[0 * ldx] += sign * fin[0];
    f[1 * ldx] += sign * fin[1];
    f[2 * ldx] -= sign * fin[2];
    f[3 * ldx] -= sign * fin[3];
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Direction for gamma %" PetscInt_FMT " not in [0, 4)", d);
  }
  f[0 * ldx] *= 0.5;
  f[1 * ldx] *= 0.5;
  f[2 * ldx] *= 0.5;
  f[3 * ldx] *= 0.5;
  PetscFunctionReturn(0);
}

#include <petsc/private/dmpleximpl.h>

// ComputeAction() sums the action of 1/2 (1 \pm \gamma_\mu) U \psi into f
static PetscErrorCode ComputeAction(PetscInt d, PetscBool forward, const PetscScalar U[], const PetscScalar psi[], PetscScalar f[])
{
  PetscScalar tmp[12];

  PetscFunctionBeginHot;
  // Apply U
  for (PetscInt beta = 0; beta < 4; ++beta) {
    if (forward) DMPlex_Mult3D_Internal(U, 1, &psi[beta * 3], &tmp[beta * 3]);
    else DMPlex_MultTranspose3D_Internal(U, 1, &psi[beta * 3], &tmp[beta * 3]);
  }
  int gamma[] = {4,0,1,2,3};
  // Apply (1 \pm \gamma_\mu)/2 to each color
  for (PetscInt c = 0; c < 3; ++c) PetscCall(ComputeGammaFactor(gamma[d], forward, 3, &tmp[c]));
  // Note that we are subtracting this contribution
  for (PetscInt i = 0; i < 12; ++i) f[i] -= tmp[i];
  PetscFunctionReturn(0);
}

/*
  The assembly loop runs over vertices. Each vertex has 2d edges in its support. The edges are ordered first by the dimension along which they run, and second from smaller to larger index, expect for edges which loop periodically. The vertices on edges are also ordered from smaller to larger index except for periodic edges.
*/
static PetscErrorCode ComputeResidual(DM dm, Vec u, Vec f)
{
  DM                 dmAux;
  Vec                gauge;
  PetscSection       s, sGauge;
  const PetscScalar *ua;
  PetscScalar       *fa, *link;
  PetscInt           dim, vStart, vEnd;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(VecGetArrayRead(u, &ua));
  PetscCall(VecGetArray(f, &fa));

  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &gauge));
  PetscCall(VecGetDM(gauge, &dmAux));
  PetscCall(DMGetLocalSection(dmAux, &sGauge));
  PetscCall(VecGetArray(gauge, &link));
  // Loop over y
  for (PetscInt v = vStart; v < vEnd; ++v) {
    const PetscInt *supp;
    PetscInt        xdof, xoff;

    PetscCall(DMPlexGetSupport(dm, v, &supp));
    PetscCall(PetscSectionGetDof(s, v, &xdof));
    PetscCall(PetscSectionGetOffset(s, v, &xoff));
    // Diagonal
    for (PetscInt i = 0; i < xdof; ++i) fa[xoff + i] += (5-M ) * ua[xoff + i];
    // Loop over mu
    for (PetscInt d = 0; d < dim; ++d) {
      const PetscInt *cone;
      PetscInt        yoff, goff;

      // Left action -(1 + \gamma_\mu)/2 \otimes U^\dagger_\mu(y) \delta_{x - \mu,y} \psi(y)
      PetscCall(DMPlexGetCone(dm, supp[2 * d + 0], &cone));
      PetscCall(PetscSectionGetOffset(s, cone[0], &yoff));
      PetscCall(PetscSectionGetOffset(sGauge, supp[2 * d + 0], &goff));
      PetscCall(ComputeAction(d, PETSC_FALSE, &link[goff], &ua[yoff], &fa[xoff]));
      // Right edge -(1 - \gamma_\mu)/2 \otimes U_\mu(x) \delta_{x + \mu,y} \psi(y)
      PetscCall(DMPlexGetCone(dm, supp[2 * d + 1], &cone));
      PetscCall(PetscSectionGetOffset(s, cone[1], &yoff));
      PetscCall(PetscSectionGetOffset(sGauge, supp[2 * d + 1], &goff));
      PetscCall(ComputeAction(d, PETSC_TRUE, &link[goff], &ua[yoff], &fa[xoff]));
    }
  }
  PetscCall(VecRestoreArray(f, &fa));
  PetscCall(VecRestoreArray(gauge, &link));
  PetscCall(VecRestoreArrayRead(u, &ua));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateJacobian(DM dm, Mat *J)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeJacobian(DM dm, Vec u, Mat J)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PrintTraversal(DM dm)
{
  MPI_Comm comm;
  PetscInt vStart, vEnd;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    const PetscInt *supp;
    PetscInt        Ns;

    PetscCall(DMPlexGetSupportSize(dm, v, &Ns));
    PetscCall(DMPlexGetSupport(dm, v, &supp));
    PetscCall(PrintVertex(dm, v));
    PetscCall(PetscPrintf(comm, "\n"));
    for (PetscInt s = 0; s < Ns; ++s) {
      const PetscInt *cone;

      PetscCall(DMPlexGetCone(dm, supp[s], &cone));
      PetscCall(PetscPrintf(comm, "  Edge %" PetscInt_FMT ": ", supp[s]));
      PetscCall(PrintVertex(dm, cone[0]));
      PetscCall(PetscPrintf(comm, " -- "));
      PetscCall(PrintVertex(dm, cone[1]));
      PetscCall(PetscPrintf(comm, "\n"));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeFFT(Mat FT, PetscInt Nc, Vec x, Vec p)
{
  Vec     *xComp, *pComp;
  PetscInt n, N;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc2(Nc, &xComp, Nc, &pComp));
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecGetSize(x, &N));
  for (PetscInt i = 0; i < Nc; ++i) {
    const char *vtype;

    // HACK: Make these from another DM up front
    PetscCall(VecCreate(PetscObjectComm((PetscObject)x), &xComp[i]));
    PetscCall(VecGetType(x, &vtype));
    PetscCall(VecSetType(xComp[i], vtype));
    PetscCall(VecSetSizes(xComp[i], n / Nc, N / Nc));
    PetscCall(VecDuplicate(xComp[i], &pComp[i]));
  }
  PetscCall(VecStrideGatherAll(x, xComp, INSERT_VALUES));
  for (PetscInt i = 0; i < Nc; ++i) PetscCall(MatMult(FT, xComp[i], pComp[i]));
  PetscCall(VecStrideScatterAll(pComp, p, INSERT_VALUES));
  for (PetscInt i = 0; i < Nc; ++i) {
    PetscCall(VecDestroy(&xComp[i]));
    PetscCall(VecDestroy(&pComp[i]));
  }
  PetscCall(PetscFree2(xComp, pComp));
  PetscFunctionReturn(0);
}


#include <Grid/Grid.h>

NAMESPACE_BEGIN(Grid);
template<class vobj>
int PetscToGrid(DM dm,Vec psi,Lattice<vobj> &g_psi)
{
  typedef typename vobj::scalar_object sobj;

  GridBase *grid = g_psi.Grid();
  uint64_t lsites = grid->lSites();
  std::vector<sobj> scalardata(lsites);
  
  const PetscScalar *psih;
  PetscInt           dim, vStart, vEnd;
  PetscSection       s;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(VecGetArrayRead(psi, &psih));

  Integer idx=0  ;
  for (PetscInt v = vStart ; v < vEnd; ++v,++idx) {
    PetscInt    dof, off;
    PetscCall(PetscSectionGetDof(s, v, &dof));
    PetscCall(PetscSectionGetOffset(s, v, &off));
    ComplexD *g_p = (ComplexD *)& scalardata[idx];
    for(int d=0;d<12;d++){
      double re =PetscRealPart(psih[off + d]);
      double im =PetscImaginaryPart(psih[off + d]);
      g_p[d]=ComplexD(re,im);
    }
  }
  assert(idx==lsites);
  vectorizeFromLexOrdArray(scalardata,g_psi);    
  PetscFunctionReturn(0);
}
template<class vobj>
int GridToPetsc(DM dm,Vec psi,Lattice<vobj> &g_psi)
{
  typedef typename vobj::scalar_object sobj;

  GridBase *grid = g_psi.Grid();
  uint64_t lsites = grid->lSites();
  std::vector<sobj> scalardata(lsites);
  unvectorizeToLexOrdArray(scalardata,g_psi);    

  PetscScalar *psih;
  PetscInt           dim, vStart, vEnd;
  PetscSection       s;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(VecGetArray(psi, &psih));
  Integer idx=0  ;
  for (PetscInt v = vStart ; v < vEnd; ++v,++idx) {
    PetscInt    dof, off;
    PetscCall(PetscSectionGetDof(s, v, &dof));
    PetscCall(PetscSectionGetOffset(s, v, &off));
    ComplexD *g_p = (ComplexD *)& scalardata[idx];
    for(int d=0;d<12;d++){
      double re =real(g_p[d]);
      double im =imag(g_p[d]);
      psih[off + d] = re + im * PETSC_i;
    }
  }
  assert(idx==lsites);
  PetscFunctionReturn(0);
}
static PetscErrorCode SetGauge_Grid(DM dm, LatticeGaugeField & Umu)
{
  typedef typename LatticeGaugeField::scalar_object sobj;
  GridBase *grid = Umu.Grid();
  uint64_t lsites = grid->lSites();
  std::vector<sobj> scalardata(lsites);
  unvectorizeToLexOrdArray(scalardata,Umu);

  PetscScalar  idm[9] = {-m, 0., 0., 0., -m, 0., 0., 0., -m};
  PetscScalar  id[9]  = { 1., 0., 0., 0., 1., 0., 0., 0., 1.};
  
  DM           auxDM;
  Vec          auxVec;
  PetscSection s;
  PetscScalar  *up;
  PetscInt     eStart, eEnd;

  PetscFunctionBegin;
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &auxVec));
  PetscCall(VecGetDM(auxVec, &auxDM));
  PetscCall(DMGetLocalSection(auxDM, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
  ColourMatrixD *Grid_p = (ColourMatrixD *)&scalardata[0];

  for (PetscInt i = eStart; i < eEnd; ++i) {
    int d = i % 5;
    int s5=i/5;
    int ss=s5%Ls;
    int s4=s5/Ls;
    if ( d==0 ) { // Unit link in s-dim
      if ( ss == Ls-1 ) {
	PetscCall(VecSetValuesSection(auxVec, s, i, idm, INSERT_VALUES));
      } else { 
	PetscCall(VecSetValuesSection(auxVec, s, i, id, INSERT_VALUES));
      }
    } else {
      int jj=d-1 + 4*s4;
      ColourMatrixD U = Grid_p[jj];
      up = (PetscScalar *) &U;
      PetscCall(VecSetValuesSection(auxVec, s, i, up, INSERT_VALUES));
    }
  }
  PetscCall(VecViewFromOptions(auxVec, NULL, "-gauge_view"));
  PetscFunctionReturn(0);
}
int CheckDwfWithGrid(DM dm,Vec psi,Vec res)
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
  LatticeGaugeField     U_GT(&GRID); // Gauge transformed field
  LatticeColourMatrix   g(&GRID);    // local Gauge xform matrix

  if (0) {
    Umu = ComplexD(1.0,0.0);
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
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,&GRID);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,&GRID);
  
  LatticeFermionD    g_src(FGrid);
  LatticeFermionD    g_res(FGrid); // Grid result
  LatticeFermionD    p_res(FGrid); // Petsc result
  LatticeFermionD    diff(FGrid); // Petsc result

  PetscToGrid(dm,psi,g_src);
  
  RealD mass=m;
  RealD M5=M;
  DomainWallFermionD Ddwf(U_GT,*FGrid,*FrbGrid,GRID,RBGRID,mass,M5);
    
  //  Ddwf.Dhop(g_src,g_res,0); Passes with no 5d term
  Ddwf.M(g_src,g_res);

  std::cout << "Setting gauge to Grid "<<std::endl;
  SetGauge_Grid(dm,U_GT);

  PetscCall(ComputeResidual(dm, psi, res)); // Applies DW
  PetscToGrid(dm,res,p_res); 
  
  diff = p_res - g_res;

  std::cout << "******************************"<<std::endl;
  std::cout << "CheckDwWithGrid Grid  " << norm2(g_res)<<std::endl;
  std::cout << "CheckDwWithGrid Petsc " << norm2(p_res)<<std::endl;
  std::cout << "CheckDwWithGrid diff  " << norm2(diff)<<std::endl;
  std::cout << "******************************"<<std::endl;

  //  std::cout << " g_src "<< g_src <<std::endl;
  //  std::cout << " g_res "<< g_res <<std::endl;
  //  std::cout << " p_res "<< p_res <<std::endl;
  //  std::cout << " diff "<< diff <<std::endl;
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
            -dm_plex_dim 5 -dm_plex_shape hypercubic -dm_plex_box_faces 8,4,4,4,4 \
            -dm_plex_check_symmetry -dm_plex_check_skeleton -dm_plex_check_faces -dm_plex_check_pointsf

TEST*/
