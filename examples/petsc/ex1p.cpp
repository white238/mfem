#include "mfem.hpp"
#include "petsc.h"
#include <fstream>
#include <iostream>

#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif

using namespace std;
using namespace mfem;

class LevelContext
{
public:
   LevelContext(ParMesh *mesh, FiniteElementCollection *fec)
      : pmesh(*mesh), pfes(new ParFiniteElementSpace(mesh, fec)), one(1.0)
   {
      Xgf = new ParGridFunction(pfes);

      Xwork = new Vector(pfes->TrueVSize());
      *Xwork = 0.0;

      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;

      bform = new ParLinearForm(pfes);
      bform->AddDomainIntegrator(new DomainLFIntegrator(one));
      bform->Assemble();

      aform = new ParBilinearForm(pfes);
      aform->AddDomainIntegrator(new DiffusionIntegrator(one));
      aform->Assemble();
      aform->EliminateEssentialBC(ess_bdr, *Xgf, *bform);
      aform->Finalize();

      H.SetType(Operator::PETSC_MATAIJ);
      aform->ParallelAssemble(H);

      Xwork = Xgf->ParallelAssemble();
      Rwork = bform->ParallelAssemble();

      Xwork_petsc = new PetscParVector(MPI_COMM_WORLD, *Xwork, true);
      Rwork_petsc = new PetscParVector(MPI_COMM_WORLD, *Rwork, true);

      T.SetType(Operator::Hypre_ParCSR);
   };

   ParMesh pmesh;
   ParFiniteElementSpace *pfes = nullptr;
   Vector *Xwork = nullptr;
   PetscParVector *Xwork_petsc = nullptr;
   OperatorHandle T;
   Mat PRshell;
   Vector *Rwork = nullptr;
   PetscParVector *Rwork_petsc = nullptr;
   OperatorHandle H;
   ParBilinearForm *aform = nullptr;
   ParLinearForm *bform = nullptr;
   ParGridFunction *Xgf = nullptr;
   ConstantCoefficient one;
};

static PetscErrorCode MatMult_Prolong(Mat A, Vec X, Vec Y)
{
   PetscErrorCode ierr;
   LevelContext *ctx;
   PetscScalar *c, *f;

   PetscFunctionBeginUser;

   ierr = MatShellGetContext(A, &ctx);
   CHKERRQ(ierr);

   PetscParVector X_wrap(X, true);
   PetscParVector Y_wrap(Y, true);

   PetscPrintf(MPI_COMM_WORLD, "Prolong MATSHELL: %p\n", (void *) A);
   ctx->T->Mult(X_wrap, Y_wrap);

   PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_Restrict(Mat A, Vec X, Vec Y)
{
   PetscErrorCode ierr;
   LevelContext *ctx;
   PetscScalar *c, *f;

   PetscFunctionBeginUser;

   ierr = MatShellGetContext(A, &ctx);
   CHKERRQ(ierr);

   PetscParVector X_wrap(X, true);
   PetscParVector Y_wrap(Y, true);

   PetscPrintf(MPI_COMM_WORLD, "Restrict MATSHELL: %p\n", (void *) A);
   ctx->T->MultTranspose(X_wrap, Y_wrap);

   PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   const char *mesh_file = "../data/inline-quad.mesh";
   int order = 2;
   bool static_cond = false;
   bool visualization = false;
   bool use_petsc = true;
   const char *petscrc_file = "";
   bool petscmonitor = false;

   MFEMInitializePetsc(&argc, &argv, petscrc_file, NULL);

   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   for (int i = 0; i < 4; i++)
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   FiniteElementCollection *fec = new H1_FECollection(order, dim);

   // Number of MG levels including finest, level 0 is coarsest
   int numlevels = 3;
   int ierr;

   Array<LevelContext *> levelctx(numlevels);

   for (int i = 0; i < numlevels; i++)
   {
      levelctx[i] = new LevelContext(pmesh, fec);

      PetscPrintf(MPI_COMM_WORLD,
                  "Adding FE space of size %d\n",
                  levelctx[i]->pfes->GlobalTrueVSize());

      PetscPrintf(MPI_COMM_WORLD,
                  "%d MATAIJ %p\n",
                  i,
                  (void *) levelctx[i]->H.As<PetscParMatrix>());

      if (i > 0)
      {
         levelctx[i]->pfes->GetTrueTransferOperator(*levelctx[i - 1]->pfes,
                                                    levelctx[i]->T);

         levelctx[i]->PRshell = *(new Mat);

         ierr = MatCreateShell(MPI_COMM_WORLD,
                               levelctx[i]->pfes->GetTrueVSize(),
                               levelctx[i - 1]->pfes->GetTrueVSize(),
                               levelctx[i]->pfes->GlobalTrueVSize(),
                               levelctx[i - 1]->pfes->GlobalTrueVSize(),
                               &*levelctx[i],
                               &levelctx[i]->PRshell);
         CHKERRQ(ierr);

         LevelContext *ctx;
         ierr = MatShellGetContext(levelctx[i]->PRshell, &ctx);
         CHKERRQ(ierr);

         PetscPrintf(MPI_COMM_WORLD,
                     "%d MATSHELL %p\n",
                     i,
                     (void *) levelctx[i]->PRshell);

         ierr = MatShellSetOperation(levelctx[i]->PRshell,
                                     MATOP_MULT,
                                     (void (*)(void)) MatMult_Prolong);
         CHKERRQ(ierr);

         ierr = MatShellSetOperation(levelctx[i]->PRshell,
                                     MATOP_MULT_TRANSPOSE,
                                     (void (*)(void)) MatMult_Restrict);
         CHKERRQ(ierr);
      }

      if (i < numlevels - 1)
      {
         pmesh->UniformRefinement();
      }
   }

   KSP ksp;
   ierr = KSPCreate(MPI_COMM_WORLD, &ksp);
   CHKERRQ(ierr);
   ierr = KSPSetType(ksp, KSPCG);
   CHKERRQ(ierr);
   ierr = KSPSetNormType(ksp, KSP_NORM_NATURAL);
   CHKERRQ(ierr);
   ierr = KSPSetFromOptions(ksp);
   CHKERRQ(ierr);

   ierr = KSPSetOperators(ksp,
                          *levelctx.Last()->H.As<PetscParMatrix>(),
                          *levelctx.Last()->H.As<PetscParMatrix>());
   CHKERRQ(ierr);

   PC pc;
   ierr = KSPGetPC(ksp, &pc);
   CHKERRQ(ierr);

   ierr = PCSetType(pc, PCMG);
   CHKERRQ(ierr);

   ierr = PCMGSetLevels(pc, numlevels, nullptr);
   CHKERRQ(ierr);

   for (int i = 0; i < numlevels; i++)
   {
      KSP smoother;
      PC smoother_pc;
      ierr = PCMGGetSmoother(pc, i, &smoother);
      CHKERRQ(ierr);
      ierr = KSPSetOperators(smoother,
                             *levelctx[i]->H.As<PetscParMatrix>(),
                             *levelctx[i]->H.As<PetscParMatrix>());
      CHKERRQ(ierr);

      if (i < numlevels - 1)
      {
         ierr = PCMGSetX(pc, i, *levelctx[i]->Xwork_petsc);
         CHKERRQ(ierr);
      }

      if (i > 0)
      {
         ierr = PCMGSetInterpolation(pc, i, levelctx[i]->PRshell);
         CHKERRQ(ierr);
      }
   }

   ierr = PCMGSetType(pc, PC_MG_MULTIPLICATIVE);
   CHKERRQ(ierr);

   ierr = KSPSolve(ksp,
                   *levelctx.Last()->Rwork_petsc,
                   *levelctx.Last()->Xwork_petsc);
   CHKERRQ(ierr);

   levelctx.Last()->Xgf->SetFromTrueDofs(*levelctx.Last()->Xwork_petsc);

   if (true)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << levelctx.Last()->pmesh << *levelctx.Last()->Xgf << flush;
   }

   // delete a;
   // delete b;
   // delete fec;
   // delete pmesh;

   MFEMFinalizePetsc();

   MPI_Finalize();

   return 0;
}
