// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "rts.hpp"
#include "forall.hpp"

#include <functional>
#include <list>
#include <string>
#include <unordered_set>
#include <fstream>
#include <iomanip>

namespace mfem
{

// Initialize the unique global Runtime variable.
Runtime Runtime::runtime_singleton;

Runtime::~Runtime()
{
   if (i_am_this) { /* only *this will do something here */}
}

void Runtime::Setup()
{
   InOutClear();
   rank = 0;
   kernels.clear();
   input_address.clear();
   output_address.clear();
   // Copy all data members from the global 'runtime_singleton' into '*this'.
   std::memcpy(this, &Get(), sizeof(Runtime));
   i_am_this = true;
}

void Runtime::Start_()
{
   if (ready) {return;}
   Runtime::InOutClear();
   MFEM_VERIFY(!ready && !record,"");
   dbg("\033[7;1;37mRuntime::Start");
   record = true;
   kernels.push_back({-1,"",0,"","","<<<Start>>>"});
}

// *****************************************************************************
void Runtime::DumpGraph_()
{
   dbg("\033[31;7;1mDumpGraph_");
   std::ofstream graph_of;
   graph_of.open ("kernels.dot");
   graph_of << "digraph {\nordering=out;" << std::endl;
   graph_of << "rankdir=\"TB\"" << std::endl;
   graph_of << "edge[arrowhead=open];" << std::endl;
   graph_of << "graph [fontname = \"helvetica\"];" << std::endl;
   graph_of << "node [fontname = \"helvetica\"];" << std::endl;
   graph_of << "edge [fontname = \"helvetica\"];" << std::endl;
   graph_of << "node [style = filled, shape = circle];\n" << std::endl;
   kernel_l::const_iterator k;
   // Kernels
   /*dbg("all_kernels:");
   for (auto k : loop_kernels)
   { if (k.n < 0) { continue; } std::cout << k.n<<":"<<k.hash << "\n"; }*/
   int nb_kernels = 0;
   for (k=kernels.begin(); k != kernels.end(); ++k)
   {
      if (k->rank < 0) { continue; }
      graph_of << "kernel_" << k->rank <<" [label=\"";
      graph_of << k->hash;
      graph_of << "\" color=\"#CCDDCC\"]" <<  std::endl;
      nb_kernels += 1;
   }
   /*
   for (int k = 1; k < nb_kernels; k++)
   { graph_of << "kernel_" << (2*k-1) << " -> " << "kernel_" << (2*k+1) << std::endl;
   }*/

   // In/Out
   graph_of <<  std::endl;
   inputs_l::const_iterator a;
   inputs_l in = input_address;
   inputs_l out = output_address;
   inputs_l all = in;
   all.merge(out);
   all.sort();
   all.unique();
   const char *ADDRS = "ptr_";
   // All inputs and outputs
   for (a=all.begin(); a != all.end(); ++a)
   {
      graph_of << ADDRS << a->address << " [label=<<B>";
      graph_of << a->address;
      graph_of << "</B>> color=\"#DD8888\"]" <<  std::endl;
   }
   // Inputs
   graph_of <<  std::endl;
   /*dbg("input_address:");
   for (auto v : input_address) { std::cout << v.address << "\n"; }*/
   inputs_l in2 = input_address;
   //in2.sort();
   //in2.unique();
   for (a=in2.begin(); a != in2.end(); ++a)
   {
      const int n = a->kernel.rank;
      graph_of << ADDRS << a->address << " -> " << "kernel_" << n <<
               std::endl;
   }
   // Outputs
   graph_of <<  std::endl;
   /*dbg("output_address:");
   for (auto v : output_address) { std::cout << v.address << "\n"; }*/
   inputs_l out2 = output_address;
   //out2.sort();
   //out2.unique();
   for (a=out2.begin(); a != out2.end(); ++a)
   {
      const int n = a->kernel.rank;
      graph_of << "kernel_" << n <<  " -> " << ADDRS << a->address <<
               std::endl;
   }
   graph_of << "}" << std::endl;
   graph_of.close();
   exit(0);
}

// *****************************************************************************
void Runtime::For_()
{
   if (!Get().record) {return;}

   const int rank = Get().rank++;
   std::stringstream name; name << rank << "_FOR";
   kernels.push_back({rank,"",0,"","",strdup(name.str().c_str())});
   const kernel_t &k = Get().kernels.back();
   for (address_l::const_iterator i=in.begin(); i != in.end(); ++i)
   { Get().input_address.push_back(address_t{*i, k}); }
   for (address_l::const_iterator i=out.begin(); i != out.end(); ++i)
   { Get().output_address.push_back(address_t{*i, k}); }
   for (address_l::const_iterator i=inout.begin(); i != inout.end(); ++i)
   {
      Get().input_address.push_back(address_t{*i, k});
      Get().output_address.push_back(address_t{*i, k});
   }
   InOutClear();
}

// *****************************************************************************
void Runtime::Loop_()
{
   if (ready) {return;}
   InOutClear();
   dbg("\033[7;1;37mRuntime::Loop");
   record = false; // end of recording
   ready = true; // loop should be ready
   kernels.push_back({-1,"",0,"","","<<<Loop>>>"});
   DumpGraph_();
}

// *****************************************************************************
void Runtime::Return_()
{
   dbg("");
   kernels.push_back({-1,"",0,"","","<<<Return>>>"});
}

// *****************************************************************************
void Runtime::Break_()
{
   if (ready) {return;}
   dbg("\033[7;1;37mRuntime::Break");
   MFEM_VERIFY(!ready && !record,"");
   kernels.push_back({-1,"",0,"","","<<<Break>>>"});
}

// *****************************************************************************
void Runtime::Cond_(const char *test)
{
   if (ready) {return;}
   dbg("\033[7;1;37mRuntime::Cond: %s",test);
   MFEM_VERIFY(record,"");
   const int rank = Get().rank++;
   std::stringstream cond;
   cond << rank << "_COND(" << test << ")";
   kernels.push_back({rank,"",0,"","", strdup(cond.str().c_str())});
   const kernel_t &k = kernels.back();
   for (address_l::const_iterator i=in.begin(); i != in.end(); ++i)
   { Get().input_address.push_back(address_t{*i, k}); }
   for (address_l::const_iterator i=out.begin(); i != out.end(); ++i)
   { Get().output_address.push_back(address_t{*i, k}); }
   for (address_l::const_iterator i=inout.begin(); i != inout.end(); ++i)
   {
      Get().input_address.push_back(address_t{*i, k});
      Get().output_address.push_back(address_t{*i, k});
   }
   InOutClear();
}

// *****************************************************************************
void Runtime::Stop_()
{
   if (ready) {return;}
   dbg("\033[7;1;37mRuntime::Stop");
   kernels.push_back({-1,"",0,"","","<<<Stop>>>"});
}


// *****************************************************************************
void Runtime::Print(std::ostream &out) { dbg(""); }

// *****************************************************************************
void Runtime::RW_(void *p, const bool use_dev, int m)
{
   if (!use_dev) { return; }
   if (!Device::IsEnabled()) { return; }
   switch (m)
   {
      case 0:
      {
         Get().in.push_back({p});
         //printf("\n\033[32m%p:%d \033[m", p, m);
         break;
      }
      case 1:
      {
         Get().out.push_back({p});
         //printf("\n\033[35m%p:%d \033[m", p, m);
         //mfem_backtrace();
         break;
      }
      case 2:
      {
         Get().inout.push_back({p});
         //printf("\n\033[31m%p:%d \033[m", p, m);
         break;
      }
      default: mfem_error();
   }
   //fflush(0);
}

void Runtime::RW_(const void *p, const bool use_dev, int m)
{
   return RW_(const_cast<void*>(p), use_dev, m);
}


void Runtime::Sync() { printf("\033[33mS\033[m"); fflush(0); }



// *****************************************************************************
void Runtime::Kernel(const bool use_dev,
                     const char *file, const int line,
                     const char *function, const char *s_body)
{
   if (!use_dev || !Get().record) { InOutClear(); return; }
   if (!Device::IsEnabled()) { InOutClear(); return; }
   //printf("\n\033[33m[kernel] '%s' (%s:%d)\033[m", function, file, line);
   //printf("\n\033[33m[kernel] %s\033[m", s_body);

   address_l &in = Get().in;
   address_l &out = Get().out;
   address_l &inout = Get().inout;
   kernel_l &kernels = Get().kernels;

   // Add kernel if we haven't seen him yet
   std::stringstream kernal_ss;
   const int rank = Get().rank;

   kernal_ss << rank << "_" << file << ":" << line << ":" << function;
   std::string kernel_s = kernal_ss.str();
   const char *hash = kernel_s.c_str();

   auto ik = std::find(kernels.begin(), kernels.end(), hash);
   if (ik == kernels.end())
   {
      //printf("\n\033[33m[kernel] New kernel (\033[1;33m%s)\033[m", hash);
      kernel_t k{rank, file, line, function, s_body, strdup(hash)};
      kernels.push_back(k);
   }
   const kernel_t &k = Get().kernels.back();

   address_l::const_iterator i;
   for (i=in.begin(); i != in.end(); ++i)
   {
      //printf("\n\t\033[33m[kernel] \033[32m%p\033[m", rank_address);
      Get().input_address.push_back(address_t{*i, k});
   }
   for (i=out.begin(); i != out.end(); ++i)
   {
      //printf("\n\t\033[33m[kernel] \033[35m%p\033[m", rank_address);
      Get().output_address.push_back(address_t{*i, k});
   }
   for (i=inout.begin(); i != inout.end(); ++i)
   {
      //printf("\n\t\033[33m[kernel] \033[31m%p\033[m", rank_i_address);
      Get().input_address.push_back(address_t{*i, k});
      Get().output_address.push_back(address_t{*i, k});
   }
   Get().rank++;
   InOutClear();
}

// *****************************************************************************
void Runtime::Memcpy(const char *function, void *dst, const void *src,
                     size_t bytes)
{
   /*printf("\n\033[33m[%s] \033[32m%p\033[m => \033[35m%p\033[m (%ld bytes)\033[m",
          function, src, dst, bytes);*/
   if (!Get().record) {return;}

   const int rank = Get().rank++;
   std::stringstream kernal_ss;
   kernal_ss << rank << "_" << function;
   std::string kernel_s = kernal_ss.str();
   const char *hash = kernel_s.c_str();
   Get().kernels.push_back(kernel_t{rank,"",0,"","",strdup(hash)});
   const kernel_t &k = Get().kernels.back();
   Get().input_address.push_back(address_t{src, k});
   Get().output_address.push_back(address_t{dst, k});
   InOutClear();
}

} // mfem
