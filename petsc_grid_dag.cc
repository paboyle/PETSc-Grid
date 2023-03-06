static char help[] = "Fermions on a hypercubic lattice.\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>

/* Common operations:

 - View the input \psi as ASCII in lexicographic order: -psi_view
*/

const PetscReal M = 1.0;

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
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
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
static PetscErrorCode ComputeAction(PetscInt d,
				    PetscBool forward,
				    PetscBool dag,
				    const PetscScalar U[], const PetscScalar psi[], PetscScalar f[])
{
  PetscScalar tmp[12];

  PetscFunctionBeginHot;
  // Apply U
  for (PetscInt beta = 0; beta < 4; ++beta) {
    if (forward) DMPlex_Mult3D_Internal(U, 1, &psi[beta * 3], &tmp[beta * 3]);
    else DMPlex_MultTranspose3D_Internal(U, 1, &psi[beta * 3], &tmp[beta * 3]);
  }
  // Apply (1 \pm \gamma_\mu)/2 to each color
  PetscBool gforward;
  if ( dag ) {
    if ( forward ) gforward = PETSC_FALSE;
    else           gforward = PETSC_TRUE;
  } else {
    gforward = forward;
  }
  for (PetscInt c = 0; c < 3; ++c) PetscCall(ComputeGammaFactor(d, gforward, 3, &tmp[c]));
  // Note that we are subtracting this contribution
  for (PetscInt i = 0; i < 12; ++i) f[i] -= tmp[i];
  PetscFunctionReturn(0);
}

/*
  The assembly loop runs over vertices. Each vertex has 2d edges in its support. The edges are ordered first by the dimension along which they run, and second from smaller to larger index, expect for edges which loop periodically. The vertices on edges are also ordered from smaller to larger index except for periodic edges.
*/
static PetscErrorCode ComputeResidualGeneral(DM dm, Vec u, Vec f, PetscBool dag)
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
    for (PetscInt i = 0; i < xdof; ++i) fa[xoff + i] = (M + 4) * ua[xoff + i];
    // Loop over mu
    for (PetscInt d = 0; d < dim; ++d) {
      const PetscInt *cone;
      PetscInt        yoff, goff;
      // Left action -(1 + \gamma_\mu)/2 \otimes U^\dagger_\mu(y) \delta_{x - \mu,y} \psi(y)
      PetscCall(DMPlexGetCone(dm, supp[2 * d + 0], &cone));
      PetscCall(PetscSectionGetOffset(s, cone[0], &yoff));
      PetscCall(PetscSectionGetOffset(sGauge, supp[2 * d + 0], &goff));
      PetscCall(ComputeAction(d, PETSC_FALSE, dag, &link[goff], &ua[yoff], &fa[xoff]));
      // Right edge -(1 - \gamma_\mu)/2 \otimes U_\mu(x) \delta_{x + \mu,y} \psi(y)
      PetscCall(DMPlexGetCone(dm, supp[2 * d + 1], &cone));
      PetscCall(PetscSectionGetOffset(s, cone[1], &yoff));
      PetscCall(PetscSectionGetOffset(sGauge, supp[2 * d + 1], &goff));
      PetscCall(ComputeAction(d, PETSC_TRUE, dag, &link[goff], &ua[yoff], &fa[xoff]));
    }
  }
  PetscCall(VecRestoreArray(f, &fa));
  PetscCall(VecRestoreArray(gauge, &link));
  PetscCall(VecRestoreArrayRead(u, &ua));
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeResidual(DM dm, Vec u, Vec f)
{
  return ComputeResidualGeneral(dm,u,f,PETSC_FALSE);
}
static PetscErrorCode ComputeResidualDag(DM dm, Vec u, Vec f)
{
  return ComputeResidualGeneral(dm,u,f,PETSC_TRUE);
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

// Sets each link to be the identity for the free field test
static PetscErrorCode SetGauge_Identity(DM dm)
{
  DM           auxDM;
  Vec          auxVec;
  PetscSection s;
  PetscScalar  id[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
  PetscInt     eStart, eEnd;

  PetscFunctionBegin;
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &auxVec));
  PetscCall(VecGetDM(auxVec, &auxDM));
  PetscCall(DMGetLocalSection(auxDM, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
  for (PetscInt i = eStart; i < eEnd; ++i) { PetscCall(VecSetValuesSection(auxVec, s, i, id, INSERT_VALUES)); }
  PetscCall(VecViewFromOptions(auxVec, NULL, "-gauge_view"));
  PetscFunctionReturn(0);
}

/*
  Test the action of the Wilson operator in the free field case U = I,

    \eta(x) = D_W(x - y) \psi(y)

  The Wilson operator is a convolution for the free field, so we can check that by the convolution theorem

    \hat\eta(x) = \mathcal{F}(D_W(x - y) \psi(y))
                = \hat D_W(p) \mathcal{F}\psi(p)

  The Fourier series for the Wilson operator is

    M + \sum_\mu 2 \sin^2(p_\mu / 2) + i \gamma_\mu \sin(p_\mu)
*/
static PetscErrorCode TestFreeField(DM dm)
{
  PetscSection       s;
  Mat                FT;
  Vec                psi, psiHat;
  Vec                eta, etaHat;
  Vec                DHat; // The product \hat D_w \hat psi
  PetscRandom        r;
  const PetscScalar *psih;
  PetscScalar       *dh;
  PetscReal         *coef, nrm;
  const PetscInt    *extent, Nc = 12;
  PetscInt           dim, V     = 1, vStart, vEnd;
  PetscContainer     c;
  PetscBool          constRhs = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-const_rhs", &constRhs, NULL));

  PetscCall(SetGauge_Identity(dm));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(DMGetGlobalVector(dm, &psi));
  PetscCall(PetscObjectSetName((PetscObject)psi, "psi"));
  PetscCall(DMGetGlobalVector(dm, &psiHat));
  PetscCall(PetscObjectSetName((PetscObject)psiHat, "psihat"));
  PetscCall(DMGetGlobalVector(dm, &eta));
  PetscCall(PetscObjectSetName((PetscObject)eta, "eta"));
  PetscCall(DMGetGlobalVector(dm, &etaHat));
  PetscCall(PetscObjectSetName((PetscObject)etaHat, "etahat"));
  PetscCall(DMGetGlobalVector(dm, &DHat));
  PetscCall(PetscObjectSetName((PetscObject)DHat, "Dhat"));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &r));
  PetscCall(PetscRandomSetType(r, PETSCRAND48));
  if (constRhs) PetscCall(VecSet(psi, 1.));
  else PetscCall(VecSetRandom(psi, r));
  PetscCall(PetscRandomDestroy(&r));

  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(PetscObjectQuery((PetscObject)dm, "_extent", (PetscObject *)&c));
  PetscCall(PetscContainerGetPointer(c, (void **)&extent));
  PetscCall(MatCreateFFT(PetscObjectComm((PetscObject)dm), dim, extent, MATFFTW, &FT));

  PetscCall(PetscMalloc1(dim, &coef));
  for (PetscInt d = 0; d < dim; ++d) {
    coef[d] = 2. * PETSC_PI / extent[d];
    V *= extent[d];
  }
  PetscCall(ComputeResidual(dm, psi, eta));
  PetscCall(VecViewFromOptions(eta, NULL, "-psi_view"));
  PetscCall(VecViewFromOptions(eta, NULL, "-eta_view"));
  PetscCall(ComputeFFT(FT, Nc, psi, psiHat));
  PetscCall(VecScale(psiHat, 1. / V));
  PetscCall(ComputeFFT(FT, Nc, eta, etaHat));
  PetscCall(VecScale(etaHat, 1. / V));
  PetscCall(VecGetArrayRead(psiHat, &psih));
  PetscCall(VecGetArray(DHat, &dh));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    PetscScalar tmp[12], tmp1 = 0.;
    PetscInt    dof, off;

    PetscCall(PetscSectionGetDof(s, v, &dof));
    PetscCall(PetscSectionGetOffset(s, v, &off));
    for (PetscInt d = 0, prod = 1; d < dim; prod *= extent[d], ++d) {
      const PetscInt idx = (v / prod) % extent[d];

      tmp1 += 2. * PetscSqr(PetscSinReal(0.5 * coef[d] * idx));
      for (PetscInt i = 0; i < dof; ++i) tmp[i] = PETSC_i * PetscSinReal(coef[d] * idx) * psih[off + i];
      for (PetscInt c = 0; c < 3; ++c) ComputeGamma(d, 3, &tmp[c]);
      for (PetscInt i = 0; i < dof; ++i) dh[off + i] += tmp[i];
    }
    for (PetscInt i = 0; i < dof; ++i) dh[off + i] += (M + tmp1) * psih[off + i];
  }
  PetscCall(VecRestoreArrayRead(psiHat, &psih));
  PetscCall(VecRestoreArray(DHat, &dh));

  {
    Vec     *etaComp, *DComp;
    PetscInt n, N;

    PetscCall(PetscMalloc2(Nc, &etaComp, Nc, &DComp));
    PetscCall(VecGetLocalSize(etaHat, &n));
    PetscCall(VecGetSize(etaHat, &N));
    for (PetscInt i = 0; i < Nc; ++i) {
      const char *vtype;

      // HACK: Make these from another DM up front
      PetscCall(VecCreate(PetscObjectComm((PetscObject)etaHat), &etaComp[i]));
      PetscCall(VecGetType(etaHat, &vtype));
      PetscCall(VecSetType(etaComp[i], vtype));
      PetscCall(VecSetSizes(etaComp[i], n / Nc, N / Nc));
      PetscCall(VecDuplicate(etaComp[i], &DComp[i]));
    }
    PetscCall(VecStrideGatherAll(etaHat, etaComp, INSERT_VALUES));
    PetscCall(VecStrideGatherAll(DHat, DComp, INSERT_VALUES));
    for (PetscInt i = 0; i < Nc; ++i) {
      if (!i) {
        PetscCall(VecViewFromOptions(etaComp[i], NULL, "-etahat_view"));
        PetscCall(VecViewFromOptions(DComp[i], NULL, "-dhat_view"));
      }
      PetscCall(VecAXPY(etaComp[i], -1., DComp[i]));
      PetscCall(VecNorm(etaComp[i], NORM_INFINITY, &nrm));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Slice %" PetscInt_FMT ": %g\n", i, (double)nrm));
    }
    PetscCall(VecStrideScatterAll(etaComp, etaHat, INSERT_VALUES));
    for (PetscInt i = 0; i < Nc; ++i) {
      PetscCall(VecDestroy(&etaComp[i]));
      PetscCall(VecDestroy(&DComp[i]));
    }
    PetscCall(PetscFree2(etaComp, DComp));
    PetscCall(VecNorm(etaHat, NORM_INFINITY, &nrm));
    PetscCheck(nrm < PETSC_SMALL, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Free field test failed: %g", (double)nrm);
  }

  PetscCall(PetscFree(coef));
  PetscCall(MatDestroy(&FT));
  PetscCall(DMRestoreGlobalVector(dm, &psi));
  PetscCall(DMRestoreGlobalVector(dm, &psiHat));
  PetscCall(DMRestoreGlobalVector(dm, &eta));
  PetscCall(DMRestoreGlobalVector(dm, &etaHat));
  PetscCall(DMRestoreGlobalVector(dm, &DHat));
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
  
  DM           auxDM;
  Vec          auxVec;
  PetscSection s;
  PetscScalar  *id;
  PetscInt     eStart, eEnd;

  PetscFunctionBegin;
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &auxVec));
  PetscCall(VecGetDM(auxVec, &auxDM));
  PetscCall(DMGetLocalSection(auxDM, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
  int j=0;
  ColourMatrixD *Grid_p = (ColourMatrixD *)&scalardata[0];
  // Ordering of edges ???
  for (PetscInt i = eStart; i < eEnd; ++i) {
    ColourMatrixD U = Grid_p[j];
    id = (PetscScalar *) &U;
    PetscCall(VecSetValuesSection(auxVec, s, i, id, INSERT_VALUES));
    j++;
  }
  std::cout << " Set "<<j<<" gauge links "<<std::endl;
  std::cout << " Grid lsites "<<lsites<<std::endl;
  PetscCall(VecViewFromOptions(auxVec, NULL, "-gauge_view"));
  PetscFunctionReturn(0);
}
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
    Umu = ComplexD(1.0,0.0);
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

  PetscToGrid(dm,psi,g_src);
  
  RealD mass=M;
  WilsonFermionD Dw(U_GT,GRID,RBGRID,mass);
    

  std::cout << "Setting gauge to Grid "<<std::endl;
  SetGauge_Grid(dm,U_GT);

  std::cout << "Testing Dw "<<std::endl;
  Dw.M(g_src,g_res);
  PetscCall(ComputeResidual(dm, psi, res)); // Applies DW
  PetscToGrid(dm,res,p_res); 
  
  diff = p_res - g_res;
  std::cout << "******************************"<<std::endl;
  std::cout << "CheckDwWithGrid Grid  " << norm2(g_res)<<std::endl;
  std::cout << "CheckDwWithGrid Petsc " << norm2(p_res)<<std::endl;
  std::cout << "CheckDwWithGrid diff  " << norm2(diff)<<std::endl;
  std::cout << "******************************"<<std::endl;

  Dw.Mdag(g_src,g_res);

  PetscCall(ComputeResidualDag(dm, psi, res)); // Applies DW
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
    args: -dm_plex_dim 4 -dm_plex_shape hypercubic -dm_plex_box_faces 4,4,4,4 -dm_view \
          -dm_plex_check_symmetry -dm_plex_check_skeleton -dm_plex_check_faces -dm_plex_check_pointsf --grid 4.4.4.4

TEST*/
