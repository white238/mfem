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
#ifndef MFEM_RTS_HPP
#define MFEM_RTS_HPP

#include "dbg.hpp"
#include "globals.hpp"

#include <cassert>
#include <list>
#include <algorithm>

namespace mfem
{
class Runtime
{

private:
   // Kernel struct
   struct kernel_t
   {
      const int rank;
      const char *file;
      const int line;
      const char *function;
      const char *body;
      const char *hash;
      bool operator<(const kernel_t &k)
      {
         assert(hash);
         assert(k.hash);
         return strcmp(hash,k.hash);
      }
      bool operator==(const char *h)
      {
         assert(h);
         assert(hash);
         return strcmp(h,hash)==0;
      }
      bool operator==(const kernel_t &k) const
      {
         assert(hash);
         assert(k.hash);
         return strcmp(hash,k.hash)==0;
      }
   };

   // Address struct
   struct address_t
   {
      const void *address;
      const kernel_t &kernel;
      bool operator<(address_t &that)
      {
         //if (rank != that.rank) { return this->rank < that.rank; }
         return this->address < that.address;
      }
      bool operator==(address_t &that)
      {
         return (this->address == that.address);
      }
   };

   typedef std::list<void*> address_l;
   typedef std::list<kernel_t> kernel_l;
   typedef std::list<address_t> inputs_l;
   address_l in, inout, out;
   inputs_l input_address, output_address;
   int rank;
   kernel_l kernels;
   static Runtime runtime_singleton;
   bool ready = false;
   bool record = false;
   bool i_am_this = false;
   Runtime(Runtime const&);
   void operator=(Runtime const&);
   static Runtime& Get() { return runtime_singleton; }
   void Setup();
   void Start_();
   void Stop_();
   void For_();
   void Loop_();
   void Break_();
   void Return_();
   void DumpGraph_();
   void Cond_(const char *test);
   void RW_(const void *p, const bool use_dev, int m=0);
   void RW_(void *p, const bool use_dev, int m=0);
public:
   Runtime():ready(false), i_am_this(false) { Get().Setup(); }
   ~Runtime();
   void Print(std::ostream &out = mfem::out);
   static inline bool IsReady() { return Get().ready; }
   static inline void Start() { Get().Start_(); }
   static inline void For() { Get().For_(); }
   static inline void Loop() { Get().Loop_(); }
   static inline void Stop() { Get().Stop_(); }
   static inline void Break() { Get().Break_(); }
   static inline void Return() { Get().Return_(); }
   static inline void Cond(const char *test) { Get().Cond_(test); }

   static inline void InOutClear()
   {
      Get().in.clear();
      Get().out.clear();
      Get().inout.clear();
   }

   template <typename T> static const T *R(const T *p, const bool use_dev)
   { Get().RW_(p,use_dev,0); return p; }
   template <typename T> static T *W(T *p, const bool use_dev)
   { Get().RW_(p,use_dev,1); return p; }
   template <typename T> static T *RW(T *p, const bool use_dev)
   { Get().RW_(p,use_dev,2); return p; }
   static void Sync();
   static void Kernel(const bool use_dev,
                      const char *file, const int line,
                      const char *function, const char *s_body);
   static void Memcpy(const char *function, void *dst, const void *src,
                      size_t bytes);
};

} // mfem

#endif // MFEM_RTS_HPP
