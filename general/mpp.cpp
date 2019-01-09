#include <list>
#include <string>
#include <ciso646>
#include <cassert>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <string.h>
using namespace std;

// *****************************************************************************
#include "incbin.hpp"
INCBIN(Okrtc, "general/okrtc.hpp");

// *****************************************************************************
// * STRUCTS: context, error & args
// *****************************************************************************
struct argument {
  string type, name;
  bool star, is_const, restrict;
  argument(): star(false), is_const(false), restrict(false) {}
};

// *****************************************************************************
struct kernel{
   string xcc;
   string dirname;
   string name;
   string static_format;
   string static_args;
   string static_tmplt;
   string any_pointer_params;
   string any_pointer_params_;
   string any_pointer_args;
   string double_uint64_t;
   string uint64_t_double;
};

// *****************************************************************************
struct context {
   int line;
   int block;
   string& file;
   istream& in;
   ostream& out;
   std::list<argument*> args;
   kernel k;
public:
   context(istream& i, ostream& o, string &f)
      : line(1), block(-2), file(f), in(i), out(o){}
};

// *****************************************************************************
struct error {
   int line;
   string file;
   error(int l, string f): line(l), file(f) {}
};

// *****************************************************************************
static const char* strrnchr(const char* s, const unsigned char c, int n=1) {
   size_t len = strlen(s);
   char* p = (char*)s+len-1;
   for (; n; n--,p--,len--) {
      for (; len; p--,len--)
         if (*p==c) break;
      if (not len) return NULL;
      if (n==1) return p;
   }
   return NULL;
}

// *****************************************************************************
static inline void check(context &pp, const bool test){
   if (not test) throw error(pp.line,pp.file);
}

// *****************************************************************************
static inline int help(char* argv[]) {
   cout << "MFEM preprocessor:";
   cout << argv[0] << " -o output input" << endl;
   return ~0;
}

// *****************************************************************************
static inline bool is_newline(const char ch) {
   return static_cast<unsigned char>(ch) == '\n';
}

// *****************************************************************************
static inline char get(context &pp) {
   char ch;
   pp.in.get(ch);
   return ch;
}

// *****************************************************************************
static inline char put(context &pp) {
   const char c = get(pp);
   pp.out << c;
   return c;
}

// *****************************************************************************
static inline void skip_space(context &pp) {
   while (isspace(pp.in.peek())) {
      if (pp.in.peek() == '\n') pp.line++;
      put(pp);
   }
}

// *****************************************************************************
static inline void drop_space(context &pp) {
   while (isspace(pp.in.peek())) {
      if (pp.in.peek() == '\n') pp.line++;
      pp.in.get();
   }
}

// *****************************************************************************
static inline bool is_comment(context &pp) {
   if (pp.in.peek() != '/') return false;
   pp.in.get();
   const char c = pp.in.peek();
   pp.in.unget();
   if (c == '/' or c == '*') return true;
   return false;
}

// *****************************************************************************
static inline void singleLineComment(context &pp) {
   while (pp.in.peek()!=EOF and pp.in.peek()!='\n') put(pp);
   pp.line++;
}

// *****************************************************************************
static inline void blockComment(context &pp) {
   while (not pp.in.eof()) {
      const char c = put(pp);
      if (c == '\n') pp.line++;
      if (c == '*' and pp.in.peek() == '/') {
         put(pp);
         skip_space(pp);
         return;
      }
   }
}

// *****************************************************************************
static inline void comments(context &pp) {
   const char c1 = put(pp); check(pp,c1=='/');
   const char c2 = put(pp); check(pp,c2=='/' or c2=='*');
   if (c2 == '/') return singleLineComment(pp);
   return blockComment(pp);
}

// *****************************************************************************
static inline bool is_alnum(context &pp) {
   const int c = pp.in.peek();
   return isalnum(c) || c == '_';
}

// *****************************************************************************
static inline string get_name(context &pp) {
   string str;
   check(pp,is_alnum(pp));
   while ((not pp.in.eof()) and (pp.in.peek()!=EOF) and
          (isalnum(pp.in.peek()) or pp.in.peek()=='_'))
      str += pp.in.get();
   return str;
}

// *****************************************************************************
static inline string peekn(context &pp, const int n) {
   char c[n+1];
   for (int k=0;k<=n;k+=1) c[k] = 0;
   int k = 0;
   for (; k<n; k+=1) {
      if (pp.in.peek()==EOF) break;
      c[k] = pp.in.get();
   }
   string rtn = c;
   for (int l=0; l<k; l+=1) pp.in.unget();
   return rtn;
}

// *****************************************************************************
static inline string peekID(context &pp) {
   const int n = 128;
   char c[n] = {0};
   int k = 0;
   for (; k<n; k+=1) {
      const char p = pp.in.peek();
      if (p==EOF) break;
      if (not is_alnum(pp)) break;
      c[k]=pp.in.get();
   }
   string rtn = c;
   for (int l=0; l<k; l+=1) pp.in.unget();
   return rtn;
}

// *****************************************************************************
static inline void drop_name(context &pp) {
   while ((not pp.in.eof()) and (pp.in.peek()!=EOF) and
          (isalnum(pp.in.peek()) or pp.in.peek()=='_'))
      pp.in.get();
}

// *****************************************************************************
static inline bool isvoid(context &pp) {
   skip_space(pp);
   const string void_peek = peekn(pp,4);
   if (void_peek == "void") return true;
   return false;
}

// *****************************************************************************
static inline bool is_star(context &pp) {
   skip_space(pp);
   if (pp.in.peek() == '*') return true;
   return false;
}

// *****************************************************************************
static inline bool is_coma(context &pp) {
   skip_space(pp);
   if (pp.in.peek() == ',') return true;
   return false;
}

// *****************************************************************************
static inline void goto_start_of_left_paren(context &pp) {
   for (; not pp.in.eof(); put(pp)) {
      const char c = pp.in.peek();
      check(pp,c != EOF);
      if (c == '\n') pp.line++;
      if (c == '(') return;
   }
}

// *****************************************************************************
static inline bool get_args(context &pp) {
   bool empty = true;
   argument *arg = new argument();
   for (int p=0; not pp.in.eof(); empty=false) {
      if (is_star(pp)){
         arg->star = true;
         put(pp);
         continue;
      }
      if (is_coma(pp)){
         put(pp);
         continue;
      }
      const string id = peekID(pp);
      drop_name(pp);
      if (id=="const") { pp.out << id; arg->is_const = true; continue; }
      if (id=="__restrict") { pp.out << id; arg->restrict = true; continue; }
      if (id=="char") { pp.out << id; arg->type = id; continue; }
      if (id=="int") { pp.out << id; arg->type = id; continue; }
      if (id=="short") { pp.out << id; arg->type = id; continue; }
      if (id=="unsigned") { pp.out << id; arg->type = id; continue; }
      if (id=="long") { pp.out << id; arg->type = id; continue; }
      if (id=="bool") { pp.out << id; arg->type = id; continue; }
      if (id=="float") { pp.out << id; arg->type = id; continue; }
      if (id=="double") { pp.out << id; arg->type = id; continue; }
      if (id=="size_t") { pp.out << id; arg->type = id; continue; }
      pp.out /*<< "_"*/ << id;
      // focus on the name, we should have qual & type
      arg->name = id;
      pp.args.push_back(arg);
      arg = new argument();
      const char c = pp.in.peek();
      check(pp,c != EOF);
      if (c == '(') p+=1;
      if (c == ')') p-=1;
      if (c == '\n') pp.line++;
      if (p<0) { return empty; }
      drop_space(pp);
      check(pp,pp.in.peek()==',');
      put(pp);
   }
   return empty;
}

// *****************************************************************************
static inline void rtcKernelRefresh(context &pp){
   pp.k.xcc = "g++ -O3 -Wall -std=c++11 ";
   pp.k.dirname = "/home/camier1/home/mfem/okina-jit";
   pp.k.static_args = "";
   pp.k.static_tmplt = "";
   pp.k.static_format = "";
   pp.k.any_pointer_args = "";
   pp.k.any_pointer_params = "";
   pp.k.any_pointer_params_ = "";
   pp.k.double_uint64_t = "";
   pp.k.uint64_t_double = "";
   
   for(std::list<argument*>::iterator ia = pp.args.begin();
       ia != pp.args.end() ; ia++) {
      const argument *a = *ia;
      const bool is_const = a->is_const;
      //const bool is_restrict = a->restrict;
      const bool is_pointer = a->star;
      const char *type = a->type.c_str();
      const char *name = a->name.c_str();
      if (is_const and not is_pointer){
         const bool dbl = strcmp(type,"double")==0;
         if (not pp.k.static_format.empty()) pp.k.static_format += ",";
         pp.k.static_format += dbl?"0x%lx":"%ld";
         if (not pp.k.static_args.empty()) pp.k.static_args += ",";
         pp.k.static_args += dbl?"u":"";
         pp.k.static_args += name;
         if (not pp.k.static_tmplt.empty()) pp.k.static_tmplt += ",";
         pp.k.static_tmplt += "const ";
         pp.k.static_tmplt += dbl?"uint64_t":type;
         pp.k.static_tmplt += " ";
         pp.k.static_tmplt += dbl?"t":"";
         pp.k.static_tmplt += name;
         if (dbl){
            {
               // const double alpha = (union {double d; uint64_t u;}){u:talpha}.d;
               pp.k.double_uint64_t += "const double ";
               pp.k.double_uint64_t += name;
               pp.k.double_uint64_t += " = (union {double d; uint64_t u;}){u:t";
               pp.k.double_uint64_t += name;
               pp.k.double_uint64_t += "}.d;";
            }
            {
               //const uint64_t ualpha = (union {double d; uint64_t u;}){alpha}.u;
               pp.k.uint64_t_double += "const uint64_t u";
               pp.k.uint64_t_double += name;
               pp.k.uint64_t_double += " = (union {double d; uint64_t u;}){";
               pp.k.uint64_t_double += name;
               pp.k.uint64_t_double += "}.u;";
            }
         }
      }
      if (is_const and is_pointer){
         if (not pp.k.any_pointer_args.empty()) pp.k.any_pointer_args += ",";
         //pp.k.any_pointer_args += "_";
         pp.k.any_pointer_args += name;
         if (not pp.k.any_pointer_params.empty()) {
            pp.k.any_pointer_params += ",";
            pp.k.any_pointer_params_ += ",";
         }
         {
            pp.k.any_pointer_params += "const ";
            pp.k.any_pointer_params += type;
            pp.k.any_pointer_params += " *";
            pp.k.any_pointer_params += name;
         }
         {
            pp.k.any_pointer_params_ += "const ";
            pp.k.any_pointer_params_ += type;
            pp.k.any_pointer_params_ += " *_";
            pp.k.any_pointer_params_ += name;
         }
      }
      if (not is_const and is_pointer){
         if (not pp.k.any_pointer_args.empty()) pp.k.any_pointer_args += ",";
         //pp.k.any_pointer_args += "_";
         pp.k.any_pointer_args += name;
         if (not pp.k.any_pointer_params.empty()){
            pp.k.any_pointer_params += ",";
            pp.k.any_pointer_params_ += ",";
         }
         {
            pp.k.any_pointer_params += type;
            pp.k.any_pointer_params += " *";
            pp.k.any_pointer_params += name;
         }
         {
            pp.k.any_pointer_params_ += type;
            pp.k.any_pointer_params_ += " *_";
            pp.k.any_pointer_params_ += name;
         }
      }
   }
   //cout << "\nstatic_args:" << pp.k.static_args;
   //cout << "\nstatic_tmplt:" << pp.k.static_tmplt;
   //cout << "\nstatic_format:" << pp.k.static_format;
   //cout << "\nany_pointer_args:" << pp.k.any_pointer_args;
   //cout << "\nany_pointer_params:" << pp.k.any_pointer_params;
   //fflush(0);
   //while(true);   
}

// *****************************************************************************
static inline void __kernel(context &pp) {
   //        "__kernel "
   pp.out << "         ";
   drop_space(pp);
   check(pp,isvoid(pp)); // we need this for now
   const string void_return_type = get_name(pp);
   pp.out << void_return_type;
   // Get kernel's name
   skip_space(pp);
   const string name = get_name(pp);
   pp.out << name;
   pp.k.name = name;
   
   skip_space(pp);
   //goto_start_of_left_paren(pp);
   // check we are at the left parenthesis
   check(pp,pp.in.peek()=='(');
   put(pp); // put '('
   // Go to first possible argument
   skip_space(pp);
   if (isvoid(pp)) { // if it is 'void' don't add any coma
      drop_name(pp);
   } else {
      pp.args.clear();
      const bool empty = get_args(pp);
      rtcKernelRefresh(pp);
      check(pp,pp.in.peek()==')');
      //if (not empty) pp.out << ", ";
   }
   // __kernel((CPU, GPU & JIT)) will add more options than the '0'
   //pp.out << "const unsigned int __kernel =0";
}

// *****************************************************************************
static inline void rtcKernelPrefix(const context &pp){
   const char *xcc = pp.k.xcc.c_str();
   const char *dirname = pp.k.dirname.c_str();   
   const char *kernel_name = pp.k.name.c_str();   
   const char *static_format = pp.k.static_format.c_str();
   const char *static_args = pp.k.static_args.c_str();
   const char *static_tmplt = pp.k.static_tmplt.c_str();   
   const char *any_pointer_params = pp.k.any_pointer_params.c_str();
   const char *any_pointer_params_ = pp.k.any_pointer_params_.c_str();
   const char *any_pointer_args = pp.k.any_pointer_args.c_str();
   const char *double_uint64_t = pp.k.double_uint64_t.c_str();
   const char *uint64_t_double = pp.k.uint64_t_double.c_str();
      
   pp.out << "\n\ttypedef void (*kernel_t)("<<any_pointer_params<<");";
   pp.out << "\n\tstatic std::unordered_map<size_t,ok::okrtc<kernel_t>*> __kernels;";
   pp.out << "\n\t" << uint64_t_double;
   pp.out << "\n\tconst char *src=R\"_(";
   pp.out << "\n#include <cstdint>";
   pp.out << "\n#include <stdbool.h>";
   pp.out << "\n#include \"general/okina.hpp\"";
   pp.out << "\ntemplate<"<< static_tmplt <<">";
   pp.out << "\nvoid rtc_"<<kernel_name<<"("<< any_pointer_params_ <<"){";
   pp.out << "\n\t" << double_uint64_t;
}

// *****************************************************************************
static inline void rtcKernelPostfix(context &pp){
   const char *xcc = pp.k.xcc.c_str();
   const char *dirname = pp.k.dirname.c_str();   
   const char *kernel_name = pp.k.name.c_str();   
   const char *static_format = pp.k.static_format.c_str();
   const char *static_args = pp.k.static_args.c_str();
   const char *static_tmplt = pp.k.static_tmplt.c_str();   
   const char *any_pointer_params = pp.k.any_pointer_params.c_str();
   const char *any_pointer_args = pp.k.any_pointer_args.c_str();
   
   pp.out << "\nextern \"C\" void k%016lx("<<any_pointer_params<<"){";
	pp.out << "\n\trtc_"<<kernel_name<<"<"<<static_format<<">("<<any_pointer_args<<");";
   pp.out << "\n})_\";";
   pp.out << "\n\tconst char *xcc = \"" << xcc << "\";";
   //pp.out << "\n\tprintf(\"\\n\033[33m[rtcKernelPostfix] args_seed & args_hash\033[m\");";
   pp.out << "\n\tconst size_t args_seed = std::hash<size_t>()(0);";
   pp.out << "\n\tconst size_t args_hash = ok::hash_args(args_seed,"<<static_args<<");";
   pp.out << "\n\tif (!__kernels[args_hash]){";
   //pp.out << "\n\tprintf(\"\\n\033[33m[rtcKernelPostfix] new ok::okrtc<kernel_t>\033[m\");";
   pp.out << "\n\t\t__kernels[args_hash] = new ok::okrtc<kernel_t>"
          << "(xcc,src," << "\"-I" << dirname << "\"," << static_args << ");";
   pp.out << "}\n\t(__kernels[args_hash]->operator_void("<< any_pointer_args <<"));";
   pp.out << "\n}";
   pp.block--;
}

// *****************************************************************************
// * '__' was hit, now fetch its 'id'
// *****************************************************************************
static inline void __id(context &pp) {
   const string id = get_name(pp);
   if (id=="__kernel"){
      // Get arguments of this kernel
      __kernel(pp);
      check(pp,pp.in.peek()==')');
      put(pp);
      skip_space(pp);
      check(pp,pp.in.peek()=='{');
      put(pp);

      // dump RTC stuff for this kernel
      pp.out << "\n\t// Now RTC this kernel ******************************************************";
      rtcKernelPrefix(pp);

      for(std::list<argument*>::iterator ia = pp.args.begin();
          ia != pp.args.end() ; ia++) {
         const argument *a = *ia;
         const bool is_const = a->is_const;
         //const bool is_restrict = a->restrict;
         const bool is_pointer = a->star;
         const char *type = a->type.c_str();
         const char *name = a->name.c_str();
         if (is_const and not is_pointer){
            //pp.out << "\n\t//GET_CONST_T_("<<name<<", "<<type<<");" << " // could be JIT'ed";
         }
         if (is_const and is_pointer){
            pp.out << "\n\tGET_CONST_ADRS_T_("<<name<<", "<<type<<");";
         }
         if (not is_const and is_pointer){
            pp.out << "\n\tGET_ADRS_T_("<<name<<", "<<type<<");";
         }/*
            dbg("%s%s %s%s%s",
            is_const?"const ":"",
            type,
            is_restrict?" __restrict ":"",
            is_pointer?"*":"",
            name);*/
      }
      pp.block = 0;
      return;
   }
   pp.out << id;
}

// *****************************************************************************
static inline void dumpOKRTC(context &pp){
   // Assert gOkrtcData, gOkrtcEnd, gOkrtcSize
   assert(&gOkrtcData[gOkrtcSize] == (const unsigned char*) &gOkrtcEnd);
   const size_t okrtc_sz = gOkrtcSize;
   const char *okrtc_h = (char*) gOkrtcData;
   pp.out << okrtc_h;
}

// *****************************************************************************
static inline int process(context &pp) {
   dumpOKRTC(pp);
   while (not pp.in.eof()) {
      if (is_comment(pp)) comments(pp);
      if (pp.in.peek() != EOF) put(pp);
      if (peekn(pp,2) == "__") __id(pp);
      if (pp.block==-1) { rtcKernelPostfix(pp); }
      if (pp.block>=0 and pp.in.peek() == '{') { pp.block++; }
      if (pp.block>=0 and pp.in.peek() == '}') { pp.block--; }
      if (is_newline(pp.in.peek())) { pp.line++;}
   }
   return 0;
}

// *****************************************************************************
int main(const int argc, char* argv[]) {
   string input, output, file;   
   if (argc<=1) return help(argv);
   for (int i=1; i<argc; i+=1) {
      // -h lauches help
      if (argv[i] == string("-h"))
         return help(argv);
      // -o fills output
      if (argv[i] == string("-o")) {
         output = argv[i+1];
         i+=1;
         continue;
      }
      // should give input file
      const char* last_dot = strrnchr(argv[i],'.');
      const size_t ext_size = last_dot?strlen(last_dot):0;
      if (last_dot && ext_size>0) {
         assert(file.size()==0);
         file = input = argv[i];
      }
   }
   assert(not input.empty());
   const bool output_file = not output.empty();
   ifstream in(input.c_str(), ios::in);
   ofstream out(output.c_str(),ios::out);
   assert(in.is_open());
   if (output_file) {assert(out.is_open());}
   context pp(in,output_file?out:cout,file);
   try {
      process(pp);
   } catch (error err) {
      cerr << err.file << ":" << err.line << ":"
           << " parser error" << endl;
      unlink(output.c_str());
      exit(-1);
   }
   return 0;
}
