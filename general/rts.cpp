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

// *****************************************************************************
Runtime::~Runtime()
{
   if (i_am_this) { /* only *this will do something here */}
}

// *****************************************************************************
void Runtime::Setup_()
{
   InOutClear();
   rank = 1;
   kernels.clear();
   input_address.clear();
   output_address.clear();
   // Copy all data members from the global 'runtime_singleton' into '*this'.
   std::memcpy(this, &RTS(), sizeof(Runtime));
   i_am_this = true;
   names.clear();
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
   // Kernels
   //for (auto K : kernels) { dbg("%d:%s", K.rank, K.hash); }
   int number_of_kernels = 0;
   for (k_it k=kernels.begin(); k != kernels.end(); ++k)
   {
      if (k->rank < 0) { continue; }
      graph_of << "kernel_" << k->rank
               << " [label=\"" << (k->hash+0) << "\" color=\"#CCDDCC\"]"
               << std::endl;
      number_of_kernels++;
   }

   // In/Out
   graph_of <<  std::endl;
   inputs_l in = input_address;
   inputs_l out = output_address;
   inputs_l all = in;
   all.merge(out);
   all.sort();
   all.unique();
   const char *ADDRS = "ptr_";
   // All inputs and outputs
   for (a_it a=all.begin(); a != all.end(); ++a)
   {
      graph_of << ADDRS << a->rank << "_" << a->address
               << " [label=<<B>" //<< a->rank << "_"
               << this->Name_(a->address/*, a->rank*/)
               << "</B>> color=\"#DD8888\"]"
               << std::endl;
   }
   // Inputs
   graph_of << std::endl;
   /*dbg("input_address:");
   for (auto v : input_address) { std::cout << v.address << "\n"; }*/
   inputs_l in2 = input_address;
   in2.sort();
   //in2.unique();
   for (a_it a=in2.begin(); a != in2.end(); ++a)
   {
      const int n = a->kernel.rank;
      graph_of << ADDRS << a->rank << "_" << a->address
               << " -> " << "kernel_" << n
               << std::endl;
   }

   // Outputs
   graph_of <<  std::endl;
   /*dbg("output_address:");
   for (auto v : output_address) { std::cout << v.address << "\n"; }*/
   inputs_l out2 = output_address;
   out2.sort();
   //out2.unique();
   for (a_it o=out2.begin(); o != out2.end(); ++o)
   {
      graph_of << "kernel_" << o->kernel.rank <<  " -> "
               << ADDRS << o->rank << "_"<< o->address
               << std::endl;
   }

   // Outputs => future alone Inputs
   for (a_it o=out2.begin(); o != out2.end(); ++o)
   {
      for (a_it i=in2.begin(); i != in2.end(); ++i)
      {
         // Test for future
         if (o->address != i->address) { continue; }
         if (o->rank >= i->rank) { continue; }
         // Test if the candidate in alone
         bool alone = true;
         for (a_it o2=out2.begin(); o2 != out2.end(); ++o2)
         {
            if (o->address != o2->address) { continue; }
            if (o->rank >= o2->rank) { continue; }
            alone = false;
            break;
         }
         if (alone)
         {
            graph_of << ADDRS << o->rank << "_"<< o->address
                     << " -> "
                     << ADDRS << i->rank << "_"<< i->address
                     << std::endl;
         }
         break; // only next one
      }
   }

   // Outputs => future hanging Outputs
   for (a_it o=out2.begin(); o != out2.end(); ++o)
   {
      for (a_it o2=out2.begin(); o2 != out2.end(); ++o2)
      {
         if (o->address != o2->address) { continue; }
         if (o->rank >= o2->rank) { continue; }
         bool hanging = true;
         for (a_it i=in2.begin(); i != in2.end(); ++i)
         {
            if (o->address != i->address) { continue; }
            if (o->rank >= i->rank) { continue; }
            hanging = false;
            break;
         }
         if (hanging)
         {
            graph_of << ADDRS << o->rank << "_"<< o2 ->address
                     << " -> "
                     << ADDRS << o2->rank << "_"<< o2->address
                     << std::endl;
         }
         break; // only next one
      }
   }

   // Same ranks
   for (int n = 1; n <= number_of_kernels; n++)
   {
      graph_of << "\t{ rank=same; ";
      for (a_it a = all.begin(); a != all.end(); ++a)
      {
         if (2*n-1 == a->rank)
         {
            graph_of << ADDRS << a->rank << "_"<< a ->address << " ";
         }
      }
      graph_of << "}" << std::endl;
   }

   graph_of << "}" << std::endl;
   graph_of.close();
   //exit(0);
}

// *****************************************************************************
void Runtime::InOutClear_()
{
   in.clear();
   out.clear();
}

// *****************************************************************************
int Runtime::GetRank_() { return 2 * rank; }
void Runtime::IncRank_() { rank++; }

// *****************************************************************************
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
void Runtime::InOutPushBack_()
{
   dbg("rank=%d",rank);
   const kernel_t &k = kernels.back();
   for (p_it i=in.begin(); i != in.end(); ++i)
   { input_address.push_back(address_t{2*rank-1, *i, k}); }
   for (p_it i=out.begin(); i != out.end(); ++i)
   { output_address.push_back(address_t{2*rank+1, *i, k}); }
   InOutClear();
}
// *****************************************************************************
void Runtime::For_()
{
   if (!RTS().record) {return;}
   const int rank = GetRank_();
   std::stringstream name; name << rank << "_FOR";
   kernels.push_back({rank,"",0,"","",strdup(name.str().c_str())});
   InOutPushBack_();
   IncRank_();
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
   const int rank = GetRank_();
   std::stringstream cond;
   cond << rank << "_COND(" << test << ")";
   kernels.push_back({rank,"",0,"","", strdup(cond.str().c_str())});
   InOutPushBack_();
   IncRank_();
}

// *****************************************************************************
void Runtime::Stop_()
{
   if (ready) {return;}
   dbg("\033[7;1;37mRuntime::Stop");
   kernels.push_back({-1,"",0,"","","<<<Stop>>>"});
}

// *****************************************************************************
void Runtime::RW_(void *p, const bool use_dev, int m)
{
   if (!use_dev || !Device::IsEnabled()) { return; }
   if (m==0 || m==2) { RTS().in.push_back({p}); }
   if (m==1 || m==2) { RTS().out.push_back({p}); }
}

// *****************************************************************************
void Runtime::RW_(const void *p, const bool use_dev, int m)
{ return RW_(const_cast<void*>(p), use_dev, m); }

// *****************************************************************************
const char* Runtime::Name_(const void *adrs/*, const int rank*/)
{

   //const void *a = (void*) (((char*)adrs)+rank);
   auto name_it = names.find(adrs);
   if (name_it != names.end())
   {
      dbg("\033[7;1;37mRuntime::Name: Get %p %s", adrs, name_it->second);
      return strdup(name_it->second);
   }
   return "???";
}

// *****************************************************************************
void Runtime::Kernel(const bool use_dev,
                     const char *file, const int line,
                     const char *function, const char *s_body,
                     const int N, const int X, const int Y, const int Z)
{
   if (!Device::IsEnabled()) { InOutClear(); return; }
   if (!use_dev || !RTS().record) { InOutClear(); return; }
   //printf("\n\033[33m[kernel] '%s' (%s:%d)\033[m", function, file, line);
   //printf("\n\033[33m[kernel] %s\033[m", s_body);

   const int rank = RTS().GetRank_();
   kernel_l &kernels = RTS().kernels;

   // Add kernel if we haven't seen him yet
   std::stringstream kernal_ss;
   kernal_ss << rank << "_" << file << ":" << line << ":" << function;
   kernal_ss << "_"<< N << "_"<< X << "_"<< Y << "_"<< Z;
   std::string kernel_s = kernal_ss.str();
   const char *hash = kernel_s.c_str();

   auto ik = std::find(kernels.begin(), kernels.end(), hash);
   if (ik == kernels.end())
   {
      dbg("\n\033[33m[kernel] New kernel (\033[1;33m%s)\033[m", hash);
      kernel_t k{rank, file, line, function, s_body, strdup(hash)};
      kernels.push_back(k);
   }
   Runtime::InOutPushBack();
   RTS().IncRank_();
}

// *****************************************************************************
void Runtime::Memcpy(const char *function, void *dst, const void *src,
                     size_t bytes)
{
   /*printf("\n\033[33m[%s] \033[32m%p\033[m => \033[35m%p\033[m (%ld bytes)\033[m",
          function, src, dst, bytes);*/
   if (!RTS().record) { return;}
   //if (!Device::IsEnabled()) { InOutClear(); return; }
   const int rank = RTS().GetRank_();
   std::stringstream kernal_ss;
   kernal_ss << rank << "_" << function;
   std::string kernel_s = kernal_ss.str();
   const char *hash = kernel_s.c_str();
   RTS().kernels.push_back(kernel_t{rank,"",0,"","",strdup(hash)});
   const kernel_t &k = RTS().kernels.back();
   RTS().input_address.push_back(address_t{rank-1, src, k});
   RTS().output_address.push_back(address_t{rank+1, dst, k});
   InOutClear();
   RTS().IncRank_();
}

} // mfem
