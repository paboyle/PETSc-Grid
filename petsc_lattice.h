#include <petscdmplex.h>
#include <petscsnes.h>
#include <petsc/private/dmpleximpl.h>

#ifndef _PETSC_LATTICE_H
#define _PETSC_LATTICE_H

/* Common operations:

 - View the input \psi as ASCII in lexicographic order: -psi_view
*/

typedef struct {
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBegin;
  //  options->usePV = PETSC_TRUE;

  PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");
  /*  PetscCall(PetscOptionsBool("-use_pv", "Use Pauli-Villars preconditioning", "ex1.c", options->usePV, &options->usePV, NULL));*/
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
#endif
